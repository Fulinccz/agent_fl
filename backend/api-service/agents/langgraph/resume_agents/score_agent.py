"""简历评分 Agent"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from agents.registry import get_agent
from logger import get_logger
from .state import ResumeState

logger = get_logger(__name__)


@dataclass
class ResumeScoreAgent:
    """简历评分 Agent - 专注于简历多维度评分"""

    name: str = "resume_score_agent"
    llm: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        if self.llm is None:
            self.llm = get_agent(provider="local")

    def run(self, state: ResumeState) -> ResumeState:
        """执行简历评分"""
        logger.info(f"[{self.name}] 开始简历评分")

        try:
            resume = state["resume"]

            prompt = f"""请对以下简历进行评分：

【简历内容】
{resume[:2000]}

请从以下4个维度评分（1-100分），必须输出纯 JSON 格式：
{{
    "completeness": 75,
    "professionalism": 80,
    "quantification": 65,
    "matching": 70
}}

要求：
1. 只输出 JSON，不要任何解释文字
2. 不要输出 markdown 代码块
3. 确保是有效的 JSON 格式"""

            result = self.llm.generate(prompt, deepThinking=False)

            scores = self._parse_scores(result)
            overall_score = sum(scores.values()) / len(scores)

            state["score_result"] = scores
            state["overall_score"] = {
                "score": round(overall_score, 1),
                "rating": self._get_rating(overall_score),
                "description": self._get_description(overall_score)
            }
            state["current_step"] = "score_completed"

            logger.info(f"[{self.name}] 评分完成: {overall_score:.1f}")

        except Exception as e:
            logger.error(f"[{self.name}] 评分失败: {e}")
            state["error"] = f"评分失败: {str(e)}"
            state["score_result"] = {
                "completeness": 70,
                "professionalism": 70,
                "quantification": 60,
                "matching": 70
            }
            state["overall_score"] = {
                "score": 67.5,
                "rating": "中等",
                "description": "简历基本合格"
            }

        return state

    def _parse_scores(self, result: str) -> Dict[str, int]:
        """解析评分结果"""
        logger.debug(f"解析评分结果: {result[:200]}...")

        try:
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            if match:
                json_str = match.group()
                scores = json.loads(json_str)

                if "scores" in scores and isinstance(scores["scores"], dict):
                    scores = scores["scores"]

                default_scores = {
                    "completeness": 70,
                    "professionalism": 70,
                    "quantification": 60,
                    "matching": 70
                }

                valid_scores = {}
                for k, v in scores.items():
                    if isinstance(v, (int, float)):
                        valid_scores[k] = int(v)
                    elif isinstance(v, str) and v.isdigit():
                        valid_scores[k] = int(v)

                default_scores.update(valid_scores)
                logger.info(f"解析评分成功: {default_scores}")
                return default_scores
        except Exception as e:
            logger.warning(f"解析评分 JSON 失败: {e}")

        try:
            scores = {}
            patterns = {
                "completeness": r"(?:完整性|completeness)[:\s]*(\d+)",
                "professionalism": r"(?:专业度|professionalism)[:\s]*(\d+)",
                "quantification": r"(?:量化程度|quantification)[:\s]*(\d+)",
                "matching": r"(?:匹配度|matching)[:\s]*(\d+)"
            }

            for key, pattern in patterns.items():
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    scores[key] = int(match.group(1))

            if len(scores) >= 4:
                logger.info(f"通过正则提取评分: {scores}")
                return scores
        except Exception as e:
            logger.warning(f"正则提取评分失败: {e}")

        numbers = re.findall(r'(\d+)', result)
        if len(numbers) >= 4:
            logger.info(f"提取数字作为评分: {numbers[:4]}")
            return {
                "completeness": int(numbers[0]),
                "professionalism": int(numbers[1]),
                "quantification": int(numbers[2]),
                "matching": int(numbers[3])
            }

        logger.warning("无法解析评分，使用默认值")
        return {
            "completeness": 70,
            "professionalism": 70,
            "quantification": 60,
            "matching": 70
        }

    def _get_rating(self, score: float) -> str:
        if score >= 90: return "优秀"
        elif score >= 80: return "良好"
        elif score >= 70: return "中等"
        elif score >= 60: return "及格"
        else: return "需改进"

    def _get_description(self, score: float) -> str:
        if score >= 90: return "简历质量很高，专业度强"
        elif score >= 80: return "简历整体不错，有小部分可以优化"
        elif score >= 70: return "简历基本合格，建议按建议改进"
        elif score >= 60: return "简历需要较多改进"
        else: return "简历问题较多，建议重新梳理"
