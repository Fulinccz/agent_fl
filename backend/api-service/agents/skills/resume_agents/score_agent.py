"""简历评分 Agent"""
import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from agents.registry import get_agent
from logger import get_logger
from services.tracing import start_span, update_observation
from .state import ResumeState

logger = get_logger(__name__)


# 复杂度分档：按简历长度自适应分析深度（输入越多，分析越细）
COMPLEXITY_PROFILES = {
    "simple": {
        "a_cap": 40, "s_cap": 60, "tokens": 1024,
        "a_req": "指出该维度最主要的问题，必须引用简历原文佐证，不超过40字",
        "s_req": "给出一条具体可执行的改进建议，不超过60字",
    },
    "medium": {
        "a_cap": 70, "s_cap": 100, "tokens": 1536,
        "a_req": "指出该维度1-2个主要问题，须引用简历原文佐证，不超过70字",
        "s_req": "给出具体可执行的改进建议，指出应补充的内容，不超过100字",
    },
    "complex": {
        "a_cap": 110, "s_cap": 150, "tokens": 2048,
        "a_req": "指出该维度2-3个问题点（用分号分隔），每点须引用简历原文佐证，不超过110字",
        "s_req": "给出具体改进建议并附改写示例（示例应体现量化数据或更专业的表述），不超过150字",
    },
}


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
            jd = (state.get("jd") or "").strip()
            jd_section = f"\n【目标岗位 JD】\n{jd[:500]}\n" if jd else ""
            profile = COMPLEXITY_PROFILES[self._assess_complexity(resume)]

            prompt = f"""请对以下简历进行评分：

【简历内容】
{resume[:2000]}
{jd_section}
请从以下4个维度评分（1-100分），并为每个维度给出"现状评估 + 改进建议"，必须输出纯 JSON 格式：
{{
    "completeness": 75,
    "professionalism": 80,
    "quantification": 65,
    "matching": 70,
    "analysis": {{
        "completeness": {{"assessment": "现状评估，必须引用简历原文", "suggestion": "具体改进建议"}},
        "professionalism": {{"assessment": "现状评估，必须引用简历原文", "suggestion": "具体改进建议"}},
        "quantification": {{"assessment": "现状评估，必须引用简历原文", "suggestion": "具体改进建议"}},
        "matching": {{"assessment": "现状评估{'，并结合JD要求' if jd else ''}", "suggestion": "具体改进建议，针对差距给出补强方向"}}
    }}
}}

要求：
1. 只输出 JSON，不要任何解释文字和 markdown 代码块
2. assessment：{profile['a_req']}
3. suggestion：{profile['s_req']}
4. 确保是有效的 JSON 格式"""

            with start_span("llm-resume-score", as_type="generation", input=prompt[:500]) as obs:
                result = self.llm.generate(prompt, max_new_tokens=profile["tokens"])
                update_observation(obs, output=result[:500])

            scores = self._parse_scores(result)
            analysis = self._parse_analysis(result, profile["a_cap"], profile["s_cap"])
            overall_score = sum(scores.values()) / len(scores)

            state["score_result"] = scores
            state["dimension_analysis"] = {
                k: analysis.get(k) or self._fallback_analysis(scores.get(k, 0))
                for k in scores
            }
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
            state["dimension_analysis"] = {
                k: self._fallback_analysis(v)
                for k, v in state["score_result"].items()
            }
            state["overall_score"] = {
                "score": 67.5,
                "rating": "中等",
                "description": "简历基本合格"
            }

        return state

    @staticmethod
    def _extract_json_object(text: str) -> Optional[str]:
        """提取文本中第一个平衡的 JSON 对象（薄包装，复用 services.json_utils）"""
        from services.json_utils import extract_json_object
        return extract_json_object(text)

    def _parse_scores(self, result: str) -> Dict[str, int]:
        """解析评分结果"""
        logger.debug(f"解析评分结果: {result[:200]}...")

        try:
            json_str = self._extract_json_object(result)
            if json_str:
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

    @staticmethod
    def _assess_complexity(resume: str) -> str:
        """按简历长度评估复杂度：输入越多，分析要求越细"""
        n = len((resume or "").strip())
        if n >= 1200:
            return "complex"
        if n >= 400:
            return "medium"
        return "simple"

    def _parse_analysis(self, result: str, a_cap: int = 80, s_cap: int = 120) -> Dict[str, Dict[str, str]]:
        """解析 LLM 输出的各维度分析（{assessment, suggestion} 结构）；缺失/解析失败返回空 dict"""
        json_str = self._extract_json_object(result)
        if not json_str:
            return {}
        try:
            data = json.loads(json_str)
        except Exception:
            return {}

        raw = data.get("analysis")
        if isinstance(raw, str) and raw.strip():
            # analysis 为 JSON 字符串形态时归一化
            try:
                inner = json.loads(raw)
                raw = inner if isinstance(inner, dict) else {}
            except Exception:
                return {}
        if not isinstance(raw, dict):
            return {}

        key_map = {
            "completeness": "completeness", "完整性": "completeness",
            "professionalism": "professionalism", "专业度": "professionalism",
            "quantification": "quantification", "量化程度": "quantification", "量化": "quantification",
            "matching": "matching", "匹配度": "matching",
        }
        analysis: Dict[str, Dict[str, str]] = {}
        for k, v in raw.items():
            key = key_map.get(str(k).strip().lower()) or key_map.get(str(k).strip())
            if not key:
                continue
            item = self._normalize_analysis_item(v, a_cap, s_cap)
            if item:
                analysis[key] = item
        return analysis

    @staticmethod
    def _normalize_analysis_item(v: Any, a_cap: int = 80, s_cap: int = 120) -> Dict[str, str]:
        """将单条分析归一化为 {assessment, suggestion}；兼容字符串与对象两种形态"""
        if isinstance(v, str) and v.strip():
            # 字符串形态：可能是二次编码的 JSON 对象，否则整体作为 assessment
            try:
                inner = json.loads(v)
                if isinstance(inner, dict):
                    return ResumeScoreAgent._normalize_analysis_item(inner, a_cap, s_cap)
            except Exception:
                pass
            return {"assessment": v.strip()[:a_cap], "suggestion": ""}
        if isinstance(v, dict):
            assessment = str(v.get("assessment") or v.get("现状") or v.get("analysis") or "").strip()
            suggestion = str(v.get("suggestion") or v.get("建议") or v.get("advice") or "").strip()
            if assessment or suggestion:
                return {"assessment": assessment[:a_cap], "suggestion": suggestion[:s_cap]}
        if isinstance(v, (int, float)) and str(v).strip():
            return {"assessment": str(v).strip()[:a_cap], "suggestion": ""}
        return {}

    @staticmethod
    def _fallback_analysis(score: int) -> Dict[str, str]:
        """LLM 分析缺失时按分数档位生成兜底分析（现状 + 建议）"""
        if score >= 85:
            return {"assessment": "该维度表现出色，内容充分且表述专业。", "suggestion": ""}
        if score >= 70:
            return {"assessment": "该维度表现良好，仍有小幅提升空间。", "suggestion": "可补充量化数据或关键成果，进一步增强说服力。"}
        if score >= 60:
            return {"assessment": "该维度偏弱，相关内容不够充分。", "suggestion": "对照目标岗位要求，补充缺失模块或具体实践细节。"}
        return {"assessment": "该维度明显不足，需要重点改进。", "suggestion": "建议重写相关内容，补充具体项目、数据与成果。"}

    def _get_rating(self, score: float) -> str:
        if score >= 90:
            return "优秀"
        elif score >= 80:
            return "良好"
        elif score >= 70:
            return "中等"
        elif score >= 60:
            return "及格"
        else:
            return "需改进"

    def _get_description(self, score: float) -> str:
        if score >= 90:
            return "简历质量很高，专业度强"
        elif score >= 80:
            return "简历整体不错，有小部分可以优化"
        elif score >= 70:
            return "简历基本合格，建议按建议改进"
        elif score >= 60:
            return "简历需要较多改进"
        else:
            return "简历问题较多，建议重新梳理"
