"""JD匹配 Agent"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

from agents.registry import get_agent
from logger import get_logger
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
        if not jd:
            logger.info(f"[{self.name}] 无 JD，跳过匹配分析")
            state["match_result"] = {
                "match_score": 70,
                "matched_keywords": [],
                "missing_keywords": [],
                "suggestions": ["建议提供目标岗位 JD 以获得更精准的优化建议"]
            }
            state["suggestions"] = self._generate_default_suggestions(state["score_result"])
            state["current_step"] = "match_completed"
            return state

        try:
            resume = state["resume"]

            prompt = f"""分析简历与 JD 的匹配度，提供优化建议。

【JD】
{jd[:1500]}

【简历】
{resume[:1500]}

请输出：
1. 匹配度分数（0-100）
2. 简历中已有的关键词（3-5个）
3. 缺失的关键词（3-5个）
4. 3条优化建议

只输出 JSON 格式：
{{
    "match_score": 分数,
    "matched_keywords": ["关键词1", "关键词2"],
    "missing_keywords": ["关键词3", "关键词4"],
    "suggestions": ["建议1", "建议2", "建议3"]
}}"""

            result = self.llm.generate(prompt, deepThinking=False)
            match_result = self._parse_match_result(result)
            state["match_result"] = match_result

            suggestions = self._generate_suggestions(state["score_result"], match_result)
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
            state["suggestions"] = self._generate_default_suggestions(state["score_result"])

        return state

    def _parse_match_result(self, result: str) -> Dict[str, Any]:
        """解析匹配结果"""
        try:
            match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"解析匹配结果失败: {e}")

        return {
            "match_score": 70,
            "matched_keywords": [],
            "missing_keywords": [],
            "suggestions": ["建议增加量化数据", "使用更专业的动词", "突出项目成果"]
        }

    def _generate_suggestions(
        self,
        scores: Dict[str, int],
        match_result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """生成优化建议"""
        suggestions = []

        suggestions.append({
            "priority": 1,
            "category": "量化",
            "suggestion": "增加成果数据，如提升XX%、完成XX个",
            "example": "优化系统性能，提升30%响应速度"
        })

        suggestions.append({
            "priority": 2,
            "category": "专业度",
            "suggestion": "使用更专业的动词和术语",
            "example": "主导微服务架构设计 vs 做了系统开发"
        })

        if match_result.get("missing_keywords"):
            missing = match_result.get("missing_keywords", [])[:3]
            suggestions.append({
                "priority": 3,
                "category": "匹配度",
                "suggestion": f"补充JD关键词：{', '.join(missing)}",
                "example": "在技能栏添加缺失的技术栈"
            })
        else:
            suggestions.append({
                "priority": 3,
                "category": "结构",
                "suggestion": "优化简历结构，突出重点经历",
                "example": "将最相关的项目经验放在前面"
            })

        return suggestions[:3]

    def _generate_default_suggestions(self, scores: Dict[str, int]) -> List[Dict[str, Any]]:
        """生成默认建议"""
        return [
            {
                "priority": 1,
                "category": "量化",
                "suggestion": "增加成果数据，如提升XX%、完成XX个",
                "example": "优化系统性能，提升30%响应速度"
            },
            {
                "priority": 2,
                "category": "专业度",
                "suggestion": "使用更专业的动词和术语",
                "example": "主导微服务架构设计 vs 做了系统开发"
            },
            {
                "priority": 3,
                "category": "结构",
                "suggestion": "优化简历结构，突出重点经历",
                "example": "将最相关的项目经验放在前面"
            }
        ]
