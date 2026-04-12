"""
JD 关键词匹配技能实现
分析招聘 JD 关键词，优化简历提高 ATS 通过率
"""

from typing import Dict, Any, List, Optional
from agents.registry import get_agent
from rag.retriever import RAGRetriever
from logger import get_logger

logger = get_logger(__name__)


class JDKeywordMatchSkill:
    """JD 关键词匹配技能"""
    
    def __init__(self):
        self.agent = get_agent(provider="local")
        self.retriever = RAGRetriever()
    
    def execute(
        self,
        resume: str,
        jd: str,
        use_rag: bool = True
    ) -> Dict[str, Any]:
        """
        执行 JD 关键词匹配
        
        Args:
            resume: 简历内容
            jd: 招聘 JD 内容
            use_rag: 是否使用 RAG 检索
        
        Returns:
            优化后的简历和匹配分析
        """
        logger.debug(f"执行 JD 关键词匹配，简历长度：{len(resume)}")
        
        # 1. RAG 检索相关关键词
        rag_context = ""
        if use_rag:
            try:
                keywords = self.retriever.retrieve("技术关键词 技能要求 岗位职责", top_k=5)
                if keywords:
                    rag_context = "\n\n参考关键词库：\n" + "\n".join([k.get('content', '') for k in keywords])
            except Exception as e:
                logger.warning(f"RAG 检索失败：{e}")
        
        # 2. 提取 JD 关键词
        extracted_keywords = self._extract_keywords(jd, rag_context)
        
        # 3. 分析匹配度
        match_analysis = self._analyze_match(resume, extracted_keywords)
        
        # 4. 优化简历
        optimized_resume = self._optimize_resume(resume, extracted_keywords)
        
        return {
            "success": True,
            "optimized_resume": optimized_resume,
            "match_analysis": match_analysis,
            "extracted_keywords": extracted_keywords,
        }
    
    def _extract_keywords(self, jd: str, rag_context: str) -> Dict[str, List[str]]:
        """提取 JD 关键词"""
        prompt = f"""你是专业的 JD 关键词提取专家。请从以下招聘描述中提取关键词，分为三类：

【技术关键词】
- 编程语言：Python、Java、Go、C++ 等
- 框架：Spring Boot、Django、React、Vue 等
- 工具：Docker、Kubernetes、Git、Jenkins 等
- 数据库：MySQL、Redis、MongoDB、Elasticsearch 等
- 云平台：AWS、Azure、阿里云、腾讯云等

【能力关键词】
- 软技能：团队协作、沟通能力、领导力等
- 方法论：敏捷开发、Scrum、DevOps 等
- 业务：高并发、分布式、微服务等

【职责关键词】
- 常见动词：设计、开发、优化、主导、推动等
- 职责描述：系统设计、架构规划、性能优化等

{rag_context}

【JD 内容】
{jd}

请严格按照 JSON 格式输出：
{{
    "technical_keywords": ["技术关键词列表"],
    "capability_keywords": ["能力关键词列表"],
    "responsibility_keywords": ["职责关键词列表"]
}}
"""
        
        result = self.agent.generate(prompt, deepThinking=False)
        
        # 简单解析 JSON（实际应该用 json.loads）
        try:
            import json
            # 提取 JSON 部分
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析关键词失败：{e}")
        
        # fallback：返回空列表
        return {
            "technical_keywords": [],
            "capability_keywords": [],
            "responsibility_keywords": [],
        }
    
    def _analyze_match(
        self,
        resume: str,
        keywords: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """分析简历与 JD 的匹配度"""
        prompt = f"""请分析简历与 JD 关键词的匹配度。

【JD 关键词】
技术关键词：{', '.join(keywords['technical_keywords'])}
能力关键词：{', '.join(keywords['capability_keywords'])}
职责关键词：{', '.join(keywords['responsibility_keywords'])}

【简历内容】
{resume}

请分析：
1. 简历中已经包含哪些关键词？
2. 缺失哪些关键词？
3. 匹配度评分（0-100）
4. 建议如何补充缺失的关键词？

请严格按照 JSON 格式输出：
{{
    "match_score": 0-100,
    "matched_keywords": ["已匹配的关键词"],
    "missing_keywords": ["缺失的关键词"],
    "suggestions": ["补充建议"]
}}
"""
        
        result = self.agent.generate(prompt, deepThinking=False)
        
        try:
            import json
            start = result.find('{')
            end = result.rfind('}') + 1
            if start >= 0 and end > start:
                json_str = result[start:end]
                return json.loads(json_str)
        except Exception as e:
            logger.warning(f"解析匹配分析失败：{e}")
        
        return {
            "match_score": 0,
            "matched_keywords": [],
            "missing_keywords": [],
            "suggestions": [],
        }
    
    def _optimize_resume(
        self,
        resume: str,
        keywords: Dict[str, List[str]]
    ) -> str:
        """优化简历，融入缺失的关键词"""
        prompt = f"""你是 ATS（简历筛选系统）优化专家。请将 JD 关键词自然融入到简历中。

【需要融入的关键词】
技术关键词：{', '.join(keywords['technical_keywords'])}
能力关键词：{', '.join(keywords['capability_keywords'])}
职责关键词：{', '.join(keywords['responsibility_keywords'])}

【优化原则】
1. 自然融入，不要生硬堆砌
2. 优先融入到项目经历和工作经历中
3. 保持简历的真实性和可读性
4. 使用 STAR 法则描述

【原始简历】
{resume}

请输出优化后的简历："""
        
        return self.agent.generate(prompt, deepThinking=False)


def match_jd_keywords(
    resume: str,
    jd: str,
    use_rag: bool = True
) -> Dict[str, Any]:
    """JD 关键词匹配技能入口函数"""
    skill = JDKeywordMatchSkill()
    return skill.execute(resume, jd, use_rag)
