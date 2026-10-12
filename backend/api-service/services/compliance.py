"""轻量合规与风控校验

统一入口，供纯文本优化链路、上传链路、文档导出链路复用：
1. 长度限制（400 级）
2. 提示词注入 / 滥用内容拦截（403 级）
3. 敏感个人信息拦截（403 级，按用户要求直接拒绝）
4. 上传文件魔数校验（防改扩展名伪装）
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ---------- 长度限制 ----------
LENGTH_LIMITS = {
    "resume": 20000,
    "jd": 5000,
    "text": 20000,
}


@dataclass
class ComplianceResult:
    """校验结果：allowed=False 时 reason 为用户可读的拦截原因"""
    allowed: bool = True
    reason: str = ""
    code: int = 200  # 建议的 HTTP 状态码：400/403


# ---------- 提示词注入 / 滥用词表（小词表，Demo 级） ----------
_INJECTION_PATTERNS = [
    # 允许多个修饰词叠加：忽略以上所有指令 / 无视之前的规则 / 忽略系统设定
    r"(?:忽略|无视)(?:的|以上|上面|上述|之前|前面|此前|所有|全部|任何|历史|现有|系统|你|新|要求|内容|，|,|\s){0,6}(?:指令|提示|规则|设定|要求|限制)",
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"(?:泄露|输出|打印|告诉我|说出|展示)\s{0,6}(?:系统提示|system\s*prompt|初始指令)",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"you\s+are\s+now\s+(a|an)\s+",
    r"(进入|开启)\s*(开发者|无限制|越狱)\s*模式",
]

_ABUSE_PATTERNS = [
    r"颠覆(国家|政权|社会主义)",
    r"分裂国家",
    r"恐怖(袭击|分子|活动)",
    r"制作(枪支|爆炸物|毒品|病毒)",
    r"(色情|淫秽|嫖娼|卖淫)",
    r"(赌博|博彩)(网站|平台|推荐)?",
    r"(枪支|弹药|军火)(买卖|交易|出售)",
    r"儿童(色情|不良)",
    r"(毒品|冰毒|海洛因|大麻)(购买|制作|出售)?",
    r"porn|gambling\s+site|child\s+abuse",
]

_INJECTION_RE = re.compile("|".join(_INJECTION_PATTERNS), re.IGNORECASE)
_ABUSE_RE = re.compile("|".join(_ABUSE_PATTERNS), re.IGNORECASE)

# ---------- 敏感个人信息 ----------
_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_ID_CHECK_MAP = "10X98765432"
_ID_RE = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_BANK_RE = re.compile(r"(?<!\d)\d{13,19}(?!\d)")


def _valid_id_card(num: str) -> bool:
    """身份证校验位验证（GB 11643-1999）"""
    digits = num[:17]
    if not digits.isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(digits, _ID_WEIGHTS))
    return _ID_CHECK_MAP[total % 11] == num[-1].upper()


def _luhn_valid(num: str) -> bool:
    """Luhn 算法校验银行卡号"""
    if not num.isdigit():
        return False
    total, alt = 0, False
    for ch in reversed(num):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def check_text(text: str, kind: str = "resume") -> ComplianceResult:
    """校验用户输入文本（resume/jd/text）。返回 ComplianceResult。"""
    if not text or not text.strip():
        return ComplianceResult(allowed=True)

    limit = LENGTH_LIMITS.get(kind, LENGTH_LIMITS["text"])
    if len(text) > limit:
        return ComplianceResult(
            allowed=False,
            reason=f"内容过长（{len(text)} 字），最大允许 {limit} 字，请精简后重试",
            code=400,
        )

    if _INJECTION_RE.search(text):
        return ComplianceResult(
            allowed=False,
            reason="内容包含疑似提示词注入语句，已拦截。请删除相关内容后重试",
            code=403,
        )
    if _ABUSE_RE.search(text):
        return ComplianceResult(
            allowed=False,
            reason="内容包含违规或不当信息，已拦截",
            code=403,
        )

    if _PHONE_RE.search(text):
        return ComplianceResult(
            allowed=False,
            reason="检测到疑似手机号，为保护隐私已拦截，请脱敏（如 138****0000）后重试",
            code=403,
        )
    for m in _ID_RE.finditer(text):
        if _valid_id_card(m.group()):
            return ComplianceResult(
                allowed=False,
                reason="检测到疑似身份证号，为保护隐私已拦截，请脱敏后重试",
                code=403,
            )
    # 银行卡：先排除已被身份证命中的长数字串，再做 Luhn
    id_spans = [m.span() for m in _ID_RE.finditer(text)]
    for m in _BANK_RE.finditer(text):
        span = m.span()
        if any(s <= span[0] and span[1] <= e for s, e in id_spans):
            continue
        if _luhn_valid(m.group()):
            return ComplianceResult(
                allowed=False,
                reason="检测到疑似银行卡号，为保护隐私已拦截，请脱敏后重试",
                code=403,
            )

    return ComplianceResult(allowed=True)


# ---------- 上传文件魔数校验 ----------
# 扩展名 → 魔数前缀列表（命中任一即认为真实）
_MAGIC_SIGNS = {
    ".pdf": [b"%PDF"],
    ".docx": [b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"],
    ".pptx": [b"PK\x03\x04"],
    ".xlsx": [b"PK\x03\x04"],
    ".doc": [b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".png": [b"\x89PNG\r\n\x1a\n"],
    ".webp": [b"RIFF"],
    ".bmp": [b"BM"],
    ".wav": [b"RIFF"],
    ".mp3": [b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"],
    ".m4a": [b"ftyp"],
    ".txt": [],  # 纯文本无魔数，跳过
    ".md": [],
}


def check_file_magic(content: bytes, filename: str) -> ComplianceResult:
    """校验上传文件真实格式与扩展名是否一致（防改扩展名伪装）"""
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in (filename or "") else ""
    signs = _MAGIC_SIGNS.get(ext)
    if signs is None:
        return ComplianceResult(allowed=True)  # 未知扩展名交给上层白名单
    if not signs:
        return ComplianceResult(allowed=True)  # 纯文本类型不校验
    # OLE2(doc) 与 zip(docx) 与 m4a(ftyp 偏移 4) 特殊处理
    if ext == ".m4a":
        ok = content[4:8] == b"ftyp"
    elif ext == ".webp":
        ok = content[:4] == b"RIFF" and content[8:12] == b"WEBP"
    else:
        ok = any(content.startswith(s) for s in signs)
    if not ok:
        return ComplianceResult(
            allowed=False,
            reason=f"文件实际内容与扩展名 {ext} 不符，已拒绝处理",
            code=400,
        )
    return ComplianceResult(allowed=True)
