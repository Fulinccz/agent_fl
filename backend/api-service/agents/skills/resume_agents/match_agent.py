"""JD匹配 Agent"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from agents.registry import get_agent
from logger import get_logger
from services.tracing import start_span, update_current_generation
from .state import ResumeState

logger = get_logger(__name__)


@dataclass
class JDMatchAgent:
    """JD 匹配 Agent - 分析 JD 并提供优化建议"""

    name: str = "jd_match_agent"
    llm: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        if self.llm is None:
            self.llm = get_agent(provider="local")

    def run(self, state: ResumeState) -> ResumeState:
        """执行 JD 匹配分析"""
        logger.info(f"[{self.name}] 开始 JD 匹配分析")

        jd = state.get("jd")
        resume = state.get("resume") or ""
        scores = state.get("score_result") or {}
        if not jd:
            logger.info(f"[{self.name}] 无 JD，跳过匹配分析")
            state["match_result"] = {
                "match_score": 70,
                "matched_keywords": [],
                "missing_keywords": [],
                "suggestions": ["建议提供目标岗位 JD 以获得更精准的优化建议"]
            }
            state["suggestions"] = self._generate_default_suggestions(scores, resume)
            state["current_step"] = "match_completed"
            return state

        try:
            prompt = f"""分析简历与 JD 的匹配度，提供优化建议。

【JD】
{jd[:1500]}

【简历】
{resume[:1500]}

请输出：
1. 匹配度分数（0-100）
2. 简历中已有的关键词（3-5个）
3. 缺失的关键词（3-5个）
4. 3条针对性优化建议：每条必须引用简历原文中的具体经历或表述，指出它相对 JD 差在哪里，并基于原文给出改写示例。禁止输出"增加量化数据""使用专业动词"这类不看内容的空泛建议。

只输出 JSON 格式：
{{
    "match_score": 分数,
    "matched_keywords": ["关键词1", "关键词2"],
    "missing_keywords": ["关键词3", "关键词4"],
    "suggestions": [{{"category": "类别（2-4字）", "suggestion": "针对简历原文具体内容的建议", "example": "基于简历原文的改写示例"}}]
}}"""

            with start_span("llm-jd-match", as_type="generation", input=prompt[:500]):
                # 3 条建议（含改写示例）约 300-450 token，默认 128 会截断 JSON
                result = self.llm.generate(prompt, deepThinking=False, max_new_tokens=512)
                update_current_generation(output=result[:500])
            match_result = self._parse_match_result(result)
            state["match_result"] = match_result

            suggestions = self._generate_suggestions(scores, match_result, resume)
            state["suggestions"] = suggestions
            state["current_step"] = "match_completed"

            logger.info(f"[{self.name}] 匹配分析完成: {match_result.get('match_score', 0)}")

        except Exception as e:
            logger.error(f"[{self.name}] 匹配分析失败: {e}")
            state["error"] = f"匹配分析失败: {str(e)}"
            state["match_result"] = {
                "match_score": 70,
                "matched_keywords": [],
                "missing_keywords": [],
                "suggestions": ["分析过程中出现错误，请重试"]
            }
            state["suggestions"] = self._generate_default_suggestions(scores, resume)

        return state

    def _parse_match_result(self, result: str) -> Dict[str, Any]:
        """解析匹配结果"""
        try:
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                parsed = json.loads(match.group())
                # 归一化 suggestions：兼容字符串数组与对象数组两种返回
                raw_suggestions = parsed.get("suggestions")
                if isinstance(raw_suggestions, list):
                    normalized = []
                    for item in raw_suggestions:
                        if isinstance(item, dict):
                            text = str(item.get("suggestion", "")).strip()
                            if text:
                                normalized.append({
                                    "category": str(item.get("category", "")).strip(),
                                    "suggestion": text,
                                    "example": str(item.get("example", "")).strip(),
                                })
                        elif isinstance(item, str) and item.strip():
                            normalized.append({"category": "", "suggestion": item.strip(), "example": ""})
                    parsed["suggestions"] = normalized
                return parsed
        except Exception as e:
            logger.warning(f"解析匹配结果失败: {e}")

        return {
            "match_score": 70,
            "matched_keywords": [],
            "missing_keywords": [],
            "suggestions": []
        }

    def _generate_suggestions(
        self,
        scores: Dict[str, int],
        match_result: Dict[str, Any],
        resume: str = ""
    ) -> List[Dict[str, Any]]:
        """生成优化建议：LLM 匹配建议优先（贴合简历原文），弱维度规则建议补位"""
        suggestions: List[Dict[str, Any]] = []

        # 1) LLM 产出的针对简历原文的建议
        for item in match_result.get("suggestions", []):
            text = item.get("suggestion", "").strip()
            if not text:
                continue
            # 过滤空泛建议（不看简历内容的套话）
            if self._is_generic_advice(text):
                continue
            suggestions.append({
                "priority": len(suggestions) + 1,
                "category": item.get("category") or "匹配度",
                "suggestion": text,
                "example": item.get("example", "")
            })
            if len(suggestions) >= 3:
                return suggestions

        # 2) 规则建议补位：基于真实弱维度 + JD 缺失关键词 + 简历原文证据
        suggestions.extend(
            self._rule_suggestions(scores, match_result, resume, limit=3 - len(suggestions))
        )
        return suggestions[:3]

    def _is_generic_advice(self, text: str) -> bool:
        """识别不看简历内容的空泛套话"""
        generic_patterns = (
            "增加量化数据", "使用更专业的动词", "突出项目成果", "优化简历结构",
            "使用专业动词", "注意排版", "控制篇幅"
        )
        return any(p in text and len(text) < 25 for p in generic_patterns)

    def _rule_suggestions(
        self,
        scores: Dict[str, int],
        match_result: Dict[str, Any],
        resume: str,
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """基于弱维度、JD 缺失关键词与简历原文证据生成规则建议"""
        suggestions: List[Dict[str, Any]] = []
        if limit <= 0:
            return suggestions

        THRESHOLD = 80  # 弱维度阈值：低于此分触发对应建议
        statements = self._split_statements(resume)

        # 1) JD 缺失关键词（真实差距）
        missing = [k for k in match_result.get("missing_keywords", []) if str(k).strip()][:3]
        if missing and len(suggestions) < limit:
            suggestions.append({
                "priority": len(suggestions) + 1,
                "category": "匹配度",
                "suggestion": f"JD 要求的关键词简历中未体现：{', '.join(missing)}，可在技能或项目描述中补入相关实践",
                "example": ""
            })

        # 2) 量化弱：无数字的语句
        quant = int(scores.get("quantification", 70))
        if quant < THRESHOLD and len(suggestions) < limit:
            plain_lines = [
                s for s in statements if not any(c.isdigit() for c in s)
            ][:2]
            if plain_lines:
                quoted = plain_lines[0][:30]
                suggestions.append({
                    "priority": len(suggestions) + 1,
                    "category": "量化",
                    "suggestion": f"「{quoted}」这条经历没有任何数据佐证，建议补充结果指标",
                    "example": f"改写为：{quoted}，实现 XX% 提升 / 覆盖 XX 用户"
                })

        # 3) 专业度弱：弱动词开头
        prof = int(scores.get("professionalism", 70))
        weak_verbs = ("负责", "参与", "做了", "协助", "完成")
        if prof < THRESHOLD and len(suggestions) < limit:
            weak_lines = [
                s for s in statements if s.startswith(weak_verbs)
            ][:1]
            if weak_lines:
                quoted = weak_lines[0][:30]
                suggestions.append({
                    "priority": len(suggestions) + 1,
                    "category": "专业度",
                    "suggestion": f"「{quoted}」以弱动词开头，建议替换为体现主导性的表述",
                    "example": f"改写为：主导 / 设计 / 落地 + 具体成果"
                })

        # 4) 完整性弱：内容最单薄的语句
        comp = int(scores.get("completeness", 70))
        if comp < THRESHOLD and len(suggestions) < limit and statements:
            shortest = min(statements, key=len)
            quoted = shortest[:30]
            suggestions.append({
                "priority": len(suggestions) + 1,
                "category": "完整性",
                "suggestion": f"「{quoted}」内容偏单薄，建议补充背景、职责与产出细节",
                "example": f"改写为：{quoted}，负责 XX，产出 XX"
            })

        # 5) 匹配度弱（无 JD 时给方向建议，引用原文避免空泛）
        match = int(scores.get("matching", 70))
        if match < THRESHOLD and len(suggestions) < limit and not missing and statements:
            quoted = statements[0][:30]
            suggestions.append({
                "priority": len(suggestions) + 1,
                "category": "匹配度",
                "suggestion": f"建议提供目标岗位 JD，以便针对「{quoted}」等经历做精准匹配分析",
                "example": ""
            })

        return suggestions[:limit]

    @staticmethod
    def _split_statements(resume: str) -> List[str]:
        """切分简历为语句级证据：兼容无换行的单段文本

        用户粘贴的简历常为整段无换行文本，splitlines() 切不出证据行导致
        规则建议全部落空；按句号/分号等标点切分后逐句判断。
        """
        parts = re.split(r"[。；;！？\n]+", resume or "")
        return [p.strip() for p in parts if len(p.strip()) > 10]

    def _generate_default_suggestions(self, scores: Dict[str, int], resume: str = "") -> List[Dict[str, Any]]:
        """无 JD 时的默认建议：基于评分弱维度与简历原文证据"""
        suggestions = self._rule_suggestions(scores, {"missing_keywords": []}, resume, limit=3)
        if suggestions:
            return suggestions
        # 简历太短无证据时的兜底
        return [{
            "priority": 1,
            "category": "完善",
            "suggestion": "简历内容较少，建议补充项目经历、技能清单与可量化的工作成果",
            "example": ""
        }]
