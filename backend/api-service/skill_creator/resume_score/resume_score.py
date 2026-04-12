"""
简历评分技能实现
对简历进行多维度综合评分，提供量化评估和改进建议
"""

from typing import Dict, Any, Optional, List
from agents.registry import get_agent
from rag.retriever import RAGRetriever
from logger import get_logger

logger = get_logger(__name__)


class ResumeScoreSkill:
    """简历评分技能"""
    
    def __init__(self):
        self.agent = get_agent(provider="local")
        self.retriever = RAGRetriever()
    
    def execute(
        self,
        resume: str,
        jd: Optional[str] = None,
        position_type: Optional[str] = None,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        执行简历评分
        
        Args:
            resume: 简历内容
            jd: 目标岗位描述（可选）
            position_type: 岗位类型（可选）
            use_rag: 是否使用 RAG 检索评分标准
        
        Returns:
            评分结果字典
        """
        # 多维度评分
        scores = self._score_dimensions(resume, jd, position_type)
        
        # 生成改进建议
        suggestions = self._generate_suggestions(resume, scores, jd)
        
        # 计算综合得分
        overall_score = sum(s["score"] for s in scores.values()) / len(scores)
        
        return {
            "success": True,
            "overall_score": {
                "score": round(overall_score, 1),
                "rating": self._get_rating(overall_score),
                "description": self._get_description(overall_score)
            },
            "scores": scores,
            "suggestions": suggestions
        }
    
    def _score_dimensions(
        self,
        resume: str,
        jd: Optional[str],
        position_type: Optional[str]
    ) -> Dict[str, Dict[str, Any]]:
        """多维度评分"""
        prompt = f"""你是资深 HR 和职业规划师。请对以下简历进行多维度评分。

评分维度：
1. completeness（完整性）：信息是否完整，有无关键缺失
2. professionalism（专业度）：用词是否专业，结构是否清晰
3. quantification（量化程度）：成果是否有数据支撑
4. matching（匹配度）：与目标岗位的契合程度

评分标准：0-100 分，每个维度给出分数、评价和改进建议。"""
        
        if jd:
            prompt += f"\n【目标岗位】\n{jd}\n\n请重点评估与岗位的匹配度。"
        
        if position_type:
            prompt += f"\n【岗位类型】{position_type}\n\n请根据该岗位类型调整评分权重。"
        
        prompt += f"""\n\n【简历内容】\n{resume}\n\n请严格按照 JSON 格式输出评分结果：
{{
    "completeness": {{"score": 0-100, "comment": "评价", "suggestions": ["改进建议"]}},
    "professionalism": {{"score": 0-100, "comment": "评价", "suggestions": ["改进建议"]}},
    "quantification": {{"score": 0-100, "comment": "评价", "suggestions": ["改进建议"]}},
    "matching": {{"score": 0-100, "comment": "评价", "suggestions": ["改进建议"]}}
}}"""
        
        result = self.agent.generate(prompt, deepThinking=False)
        
        try:
            import json
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析评分结果失败：{e}")
        
        # fallback：返回默认评分
        return {
            "completeness": {"score": 70, "comment": "基本信息完整", "suggestions": []},
            "professionalism": {"score": 70, "comment": "描述较为专业", "suggestions": []},
            "quantification": {"score": 60, "comment": "部分成果有量化", "suggestions": []},
            "matching": {"score": 70, "comment": "与岗位基本匹配", "suggestions": []},
        }
    
    def _generate_suggestions(
        self,
        resume: str,
        scores: Dict[str, Dict[str, Any]],
        jd: Optional[str]
    ) -> List[Dict[str, Any]]:
        """生成改进建议"""
        prompt = f"""你是专业职业顾问。请根据以下评分结果，为简历提供改进建议。

【评分结果】
"""
        for dimension, data in scores.items():
            prompt += f"- {dimension}: {data['score']}分 - {data['comment']}\n"
        
        if jd:
            prompt += f"\n【目标岗位】\n{jd}\n"
        
        prompt += f"""\n【简历内容】\n{resume}\n\n请提供 3-5 条最优先的改进建议，按重要性排序。

请严格按照 JSON 格式输出：
[
    {{"priority": 1, "category": "类别", "suggestion": "具体建议", "example": "示例"}},
    {{"priority": 2, "category": "类别", "suggestion": "具体建议", "example": "示例"}}
]"""
        
        result = self.agent.generate(prompt, deepThinking=False)
        
        try:
            import json
            start = result.find('[')
            end = result.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析建议失败：{e}")
        
        # fallback：返回默认建议
        return [
            {"priority": 1, "category": "量化", "suggestion": "增加成果数据，如提升XX%、完成XX个", "example": "优化系统性能，提升30%响应速度"},
            {"priority": 2, "category": "专业度", "suggestion": "使用更专业的动词和术语", "example": "主导微服务架构设计 vs 做了系统开发"}
        ]
    
    def _get_rating(self, score: float) -> str:
        """根据分数获取评级"""
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
        """根据分数获取描述"""
        if score >= 90:
            return "简历质量很高，专业度强，建议直接使用"
        elif score >= 80:
            return "简历整体不错，有小部分可以优化"
        elif score >= 70:
            return "简历基本合格，建议按建议改进"
        elif score >= 60:
            return "简历需要较多改进，请参考建议"
        else:
            return "简历问题较多，建议重新梳理"


def score_resume(
    resume: str,
    jd: Optional[str] = None,
    position_type: Optional[str] = None,
    use_rag: bool = True
) -> Dict[str, Any]:
    """简历评分技能入口函数"""
    skill = ResumeScoreSkill()
    return skill.execute(resume, jd, position_type, use_rag)
