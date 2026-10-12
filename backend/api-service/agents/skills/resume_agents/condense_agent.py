"""简历精简 Agent

面向内容过长的简历（>2800 字 / 约 2 页）：压缩篇幅、保留核心信息，
输出可直接用于润色的精简稿。可通过用户指令跳过（"不要精简"等）。
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from logger import get_logger
from services.tracing import start_span, update_current_generation
from .textutil import split_text, strip_llm_artifacts

logger = get_logger(__name__)

# 用户明确要求不精简的关键词
_NO_CONDENSE_RE = re.compile(
    r"不?(要|用|需|必|能)精简|别精简|不要删减|无需删减|不要删|"
    r"保留(全部|所有|完整|原有)|全文保留|"
    r"keep\s+all|don'?t\s+condense|no\s+condense|do\s+not\s+condense",
    re.IGNORECASE,
)

# 触发阈值：简历正文字数
CONDENSE_THRESHOLD = 2800
# 单次精简的输入上限（超出则分段 map-reduce，杜绝截断丢内容）
_CONDENSE_SEGMENT_CHARS = 3800
# 合并稿再次压缩的输入上限
_REDUCE_SEGMENT_CHARS = 5000
# map-reduce 最大轮数（防极端长文下无限压缩）
_MAX_REDUCE_ROUNDS = 2
# 精简结果最低有效长度（低于此值视为 LLM 失败，回退原文）
_MIN_CONDENSED_CHARS = 100


def should_condense(resume: str) -> bool:
    """判断是否需要精简：超长 且 用户未明确拒绝"""
    if not resume:
        return False
    if _NO_CONDENSE_RE.search(resume):
        logger.info("[ResumeCondenseAgent] 用户明确要求不精简，跳过")
        return False
    return len(resume.strip()) > CONDENSE_THRESHOLD


class ResumeCondenseAgent:
    """简历精简 Agent（零流式，阻塞生成）"""

    def __init__(self, llm: Optional[Any] = None):
        if llm is None:
            # 与 score_agent 对齐：未显式传入时回落到默认本地 agent
            from agents.registry import get_agent
            llm = get_agent(provider="local")
        self.llm = llm

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        resume = (state.get("resume") or "").strip()
        jd = (state.get("jd") or "").strip()
        jd_hint = f"\n目标岗位：{jd[:200]}（优先保留与岗位相关的经历与技能）" if jd else ""

        if len(resume) <= _CONDENSE_SEGMENT_CHARS:
            # 常规长度：单次精简
            condensed = self._condense_once(resume, jd_hint)
        else:
            # 超长简历：分段精简（map）→ 合并 → 必要时再压缩（reduce）
            parts = split_text(resume, _CONDENSE_SEGMENT_CHARS)
            logger.info(f"[ResumeCondenseAgent] 简历 {len(resume)} 字，分 {len(parts)} 段 map-reduce")
            mapped: list[str] = []
            for idx, part in enumerate(parts, 1):
                piece = self._condense_once(
                    part, jd_hint,
                    part_hint=f"（这是简历的第 {idx}/{len(parts)} 部分，请只精简本部分，不要输出总起或总结）",
                )
                if not piece:
                    # 任一段精简失败 → 整体回退原文，不阻断流程
                    # （必须写回 state：executor 从传入 state 取结果，
                    #   返回新 dict 会被误判为步骤失败而触发反思重试）
                    logger.warning("[ResumeCondenseAgent] 分段精简失败，回退原文")
                    state["condensed_resume"] = resume
                    state["skipped"] = True
                    return state
                mapped.append(piece)

            condensed = "\n\n".join(mapped)
            rounds = 0
            while len(condensed) > _REDUCE_SEGMENT_CHARS and rounds < _MAX_REDUCE_ROUNDS:
                # 合并稿仍过长：切块再次压缩
                chunks = split_text(condensed, _REDUCE_SEGMENT_CHARS)
                reduced: list[str] = []
                for chunk in chunks:
                    piece = self._condense_once(
                        chunk, jd_hint,
                        part_hint="（以下是已精简简历的合并稿，请继续去重与压缩，保持分节结构）",
                    )
                    if not piece:
                        break
                    reduced.append(piece)
                if not reduced:
                    break
                condensed = "\n\n".join(reduced)
                rounds += 1

        if not condensed or len(condensed) < _MIN_CONDENSED_CHARS:
            # 精简失败：返回原文（视为未精简），不阻断流程（写回 state，同上）
            logger.warning("[ResumeCondenseAgent] 精简结果为空或过短，回退原文")
            state["condensed_resume"] = resume
            state["skipped"] = True
            return state

        state["condensed_resume"] = condensed
        state["original_length"] = len(resume)
        state["condensed_length"] = len(condensed)
        state["current_step"] = "condensed"
        logger.info(
            f"[ResumeCondenseAgent] 精简完成: {len(resume)} → {len(condensed)} 字"
        )
        return state

    def _condense_once(
        self, resume_text: str, jd_hint: str = "", part_hint: str = ""
    ) -> Optional[str]:
        """对一段文本执行一次精简；LLM 失败/结果无效时返回 None"""
        prompt = f"""请精简以下过长简历{part_hint}，输出精简后的完整简历文本。

【简历原文】
{resume_text}
{jd_hint}
精简要求：
1. 篇幅压缩至原文的 60% 左右
2. 保留：核心工作经历、量化成果（数字/百分比/QPS 等）、关键技术栈、教育与证书
3. 删除：重复表述、空泛套话、与目标无关的细节描述
4. 保持原有分节结构（个人信息/工作经历/项目经历/教育经历/技能等）与条目顺序
5. 每条经历保留 2-3 个最有价值的要点
6. 只输出精简后的简历正文，不要任何解释、前后缀和 markdown 代码块"""

        try:
            with start_span("llm-resume-condense", as_type="generation", input=prompt[:500]):
                result = self.llm.generate(prompt, deepThinking=False, max_new_tokens=2048)
                update_current_generation(output=result[:500])
        except Exception as e:
            logger.warning(f"[ResumeCondenseAgent] LLM 精简调用失败: {e}")
            return None

        condensed = self._clean_result(result)
        return condensed if condensed and len(condensed) >= _MIN_CONDENSED_CHARS else None

    @staticmethod
    def _clean_result(result: str) -> str:
        """清理 LLM 输出：去 think 标签、代码块围栏（公共内核）、解释性前后缀"""
        text = strip_llm_artifacts(result)
        # 去掉常见的解释性前缀行
        lines = text.strip().split("\n")
        while lines and re.match(
            r"^(好的|以下是|这是|精简后|如下|注：|说明[:：])", lines[0].strip()
        ):
            lines.pop(0)
        return "\n".join(lines).strip()
