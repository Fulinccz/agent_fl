"""
简历优化多 Agent 工作流
基于 LangGraph 实现 3 个 Agent 的协作：
1. ResumeScoreAgent - 简历评分
2. JDMatchAgent - JD 关键词匹配与优化建议
3. ResumePolishAgent - 简历润色
"""

from typing import Dict, Any, Optional, List, TypedDict, Annotated
import operator
from dataclasses import dataclass, field

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.registry import get_agent
from logger import get_logger

logger = get_logger(__name__)


# ==================== 状态定义 ====================

class ResumeState(TypedDict):
    """工作流状态"""
    # 输入
    resume: str
    jd: Optional[str]
    position_type: Optional[str]
    
    # 中间结果
    score_result: Optional[Dict[str, Any]]
    match_result: Optional[Dict[str, Any]]
    
    # 输出
    overall_score: Optional[Dict[str, Any]]
    suggestions: Optional[List[Dict[str, Any]]]
    optimized_resume: Optional[str]
    
    # 控制
    error: Optional[str]
    current_step: str


# ==================== Agent 1: 简历评分 Agent ====================

@dataclass
class ResumeScoreAgent:
    """简历评分 Agent - 专注于简历多维度评分"""
    
    name: str = "resume_score_agent"
    
    def __post_init__(self):
        self.llm = get_agent(provider="local")
    
    def run(self, state: ResumeState) -> ResumeState:
        """执行简历评分"""
        logger.info(f"[{self.name}] 开始简历评分")
        
        try:
            resume = state["resume"]
            
            # 简化的评分 Prompt，适合小模型
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
            
            # 解析评分
            scores = self._parse_scores(result)
            
            # 计算综合得分
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
            # 使用默认评分
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
        import json
        import re
        
        logger.debug(f"解析评分结果: {result[:200]}...")
        
        # 尝试提取 JSON - 支持嵌套结构
        try:
            # 查找 JSON 部分（支持嵌套大括号）
            match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', result, re.DOTALL)
            if match:
                json_str = match.group()
                logger.debug(f"提取到 JSON: {json_str}")
                scores = json.loads(json_str)
                
                # 处理可能嵌套的结构
                if "scores" in scores and isinstance(scores["scores"], dict):
                    scores = scores["scores"]
                
                # 确保所有维度都有分数
                default_scores = {
                    "completeness": 70,
                    "professionalism": 70,
                    "quantification": 60,
                    "matching": 70
                }
                
                # 只保留数值类型的字段
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
        
        # 尝试从文本中提取数字（按维度名称匹配）
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
        
        # 最后尝试：提取任意数字
        numbers = re.findall(r'(\d+)', result)
        if len(numbers) >= 4:
            logger.info(f"提取数字作为评分: {numbers[:4]}")
            return {
                "completeness": int(numbers[0]),
                "professionalism": int(numbers[1]),
                "quantification": int(numbers[2]),
                "matching": int(numbers[3])
            }
        
        # 返回默认评分
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


# ==================== Agent 2: JD 匹配 Agent ====================

@dataclass
class JDMatchAgent:
    """JD 匹配 Agent - 分析 JD 并提供优化建议"""
    
    name: str = "jd_match_agent"
    
    def __post_init__(self):
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
            
            # 简化的匹配分析 Prompt
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
            
            # 解析结果
            match_result = self._parse_match_result(result)
            state["match_result"] = match_result
            
            # 生成建议
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
        import json
        import re
        
        try:
            match = re.search(r'\{[^}]+\}', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.warning(f"解析匹配结果失败: {e}")
        
        # 返回默认结果
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
        
        # 建议1：量化成果
        suggestions.append({
            "priority": 1,
            "category": "量化",
            "suggestion": "增加成果数据，如提升XX%、完成XX个",
            "example": "优化系统性能，提升30%响应速度"
        })
        
        # 建议2：专业动词
        suggestions.append({
            "priority": 2,
            "category": "专业度",
            "suggestion": "使用更专业的动词和术语",
            "example": "主导微服务架构设计 vs 做了系统开发"
        })
        
        # 建议3：结构优化（基于匹配度）
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
        
        return suggestions[:3]  # 固定 3 条建议
    
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


# ==================== Agent 3: 简历润色 Agent ====================

@dataclass
class ResumePolishAgent:
    """简历润色 Agent - 优化简历表述"""
    
    name: str = "resume_polish_agent"
    
    def __post_init__(self):
        self.llm = get_agent(provider="local")
    
    def run(self, state: ResumeState) -> ResumeState:
        """执行简历润色"""
        logger.info(f"[{self.name}] 开始简历润色")
        
        try:
            resume = state["resume"]
            jd = state.get("jd")
            
            # 构建润色 Prompt
            jd_hint = f"目标岗位：{jd[:500]}\n\n" if jd else ""
            
            prompt = f"""{jd_hint}你是一位资深HR和简历优化专家。请将以下简历内容改写得更专业。

【原始简历】
{resume[:2000]}

【改写要求】
1. 使用专业动词：主导、设计、优化、实现、推动、架构、落地等
2. 添加量化数据：提升XX%、完成XX个、节省XX小时、服务XX用户等
3. 突出技术难点和个人贡献
4. 保持简洁，不要添加简历中没有的内容

【输出格式】
直接输出改写后的简历文本，不要包含任何解释、分析或思考过程。只输出优化后的内容。"""
            
            result = self.llm.generate(prompt, deepThinking=False)
            
            # 清理结果
            polished = self._clean_result(result)
            
            state["optimized_resume"] = polished
            state["current_step"] = "polish_completed"
            
            logger.info(f"[{self.name}] 润色完成，长度: {len(polished)}")
            
        except Exception as e:
            logger.error(f"[{self.name}] 润色失败: {e}")
            state["error"] = f"润色失败: {str(e)}"
            state["optimized_resume"] = state["resume"]  # 使用原始简历
        
        return state
    
    def _clean_result(self, result: str) -> str:
        """清理润色结果，移除prompt内容和格式标记"""
        import re
        
        # 移除思考标签及其内容
        result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL | re.IGNORECASE)
        
        # 移除代码块标记（保留内容）
        result = re.sub(r'```\w*\n?', '', result)
        result = re.sub(r'```', '', result)
        
        # 移除常见前缀（包括行首的）
        prefixes = [
            r'^改写后的简历：\s*',
            r'^优化后的简历：\s*',
            r'^润色结果：\s*',
            r'^以下是改写后的内容：\s*',
            r'^输出：\s*',
            r'^改写结果：\s*',
            r'^优化结果：\s*',
            r'^以下是优化后的简历：\s*',
            r'^以下是改写后的简历：\s*',
            r'^assistant\s*',
            r'^用户\s*',
            r'^Human:\s*',
            r'^Assistant:\s*',
        ]
        for pattern in prefixes:
            result = re.sub(pattern, '', result, flags=re.MULTILINE)
        
        # 移除包含prompt关键词的行
        prompt_keywords = [
            "【原始简历】",
            "【简历内容】",
            "改写要求：",
            "请将以下简历",
            "使用专业动词",
            "添加量化数据",
            "突出技术难点",
            "保持简洁",
            "直接输出",
            "不要解释",
            "目标岗位：",
        ]
        
        # 需要跳过的行模式（只跳过明确的思考过程标记，保留Markdown列表）
        skip_patterns = [
            r'^\s*$',  # 空行
            r'^\s*<think>\s*$',  # think标签行
            r'^\s*Thinking Process:\s*$',  # Thinking Process行
            r'^\s*\d+\s*\.\s*\*\*.*\*\*\s*$',  # 数字. **标题** 行 (如 1. **Analyze**)
            r'^\s*\*\s*Input Text:',  # * Input Text: 行
            r'^\s*\*\s*Task:',  # * Task: 行
            r'^\s*\*\s*Requirements:',  # * Requirements: 行
            r'^\s*\*\s*\*\s+',  # ** 加粗标题行 (如 **核心能力：**)
            r'^\s*assistant\s*$',  # 单独的assistant
            r'^\s*用户\s*$',  # 单独的用户
            r'^\s*Human:\s*$',
            r'^\s*Assistant:\s*$',
        ]
        
        lines = result.split('\n')
        filtered_lines = []
        for line in lines:
            line_stripped = line.strip()
            
            # 跳过匹配任何skip_patterns的行
            if any(re.match(pattern, line, re.IGNORECASE) for pattern in skip_patterns):
                continue
            
            # 跳过包含prompt关键词的行
            if any(keyword in line for keyword in prompt_keywords):
                continue
            
            # 跳过英文思考内容（包含特定关键词）
            if any(keyword in line_stripped.lower() for keyword in [
                'analyze the request', 'input text:', 'task:', 'requirements:',
                'use professional verbs', 'add quantifiable data'
            ]):
                continue
            
            filtered_lines.append(line)
        result = '\n'.join(filtered_lines)
        
        # 移除多余的空行
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()


# ==================== LangGraph 工作流构建 ====================

def create_resume_workflow() -> StateGraph:
    """创建简历优化工作流"""
    
    # 创建 Agent 实例
    score_agent = ResumeScoreAgent()
    match_agent = JDMatchAgent()
    polish_agent = ResumePolishAgent()
    
    # 定义工作流
    workflow = StateGraph(ResumeState)
    
    # 添加节点
    workflow.add_node("score", score_agent.run)
    workflow.add_node("match", match_agent.run)
    workflow.add_node("polish", polish_agent.run)
    
    # 定义边 - 顺序执行
    workflow.set_entry_point("score")
    workflow.add_edge("score", "match")
    workflow.add_edge("match", "polish")
    workflow.add_edge("polish", END)
    
    return workflow.compile()


# ==================== 对外接口 ====================

class ResumeOptimizationWorkflow:
    """简历优化工作流 - 对外接口"""
    
    def __init__(self):
        self.workflow = create_resume_workflow()
    
    def optimize(
        self,
        resume: str,
        jd: Optional[str] = None,
        position_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行简历优化工作流
        
        Args:
            resume: 简历内容
            jd: 目标岗位 JD（可选）
            position_type: 岗位类型（可选）
        
        Returns:
            优化结果
        """
        # 初始状态
        initial_state: ResumeState = {
            "resume": resume,
            "jd": jd,
            "position_type": position_type,
            "score_result": None,
            "match_result": None,
            "overall_score": None,
            "suggestions": None,
            "optimized_resume": None,
            "error": None,
            "current_step": "started"
        }
        
        # 执行工作流
        final_state = self.workflow.invoke(initial_state)
        
        # 构建返回结果
        return {
            "success": final_state.get("error") is None,
            "overall_score": final_state.get("overall_score"),
            "scores": final_state.get("score_result"),
            "suggestions": final_state.get("suggestions"),
            "optimized_resume": final_state.get("optimized_resume"),
            "match_analysis": final_state.get("match_result"),
            "error": final_state.get("error")
        }
    
    def optimize_stream(
        self,
        resume: str,
        jd: Optional[str] = None,
        position_type: Optional[str] = None
    ):
        """
        流式执行简历优化工作流
        
        Yields:
            每个步骤的结果
        """
        logger.info(f"[optimize_stream] 开始流式执行")
        
        # 初始状态
        state: ResumeState = {
            "resume": resume,
            "jd": jd,
            "position_type": position_type,
            "score_result": None,
            "match_result": None,
            "overall_score": None,
            "suggestions": None,
            "optimized_resume": None,
            "error": None,
            "current_step": "started"
        }
        
        # Step 1: 评分
        logger.info(f"[optimize_stream] Step 1: 简历评分")
        score_agent = ResumeScoreAgent()
        state = score_agent.run(state)
        
        if state.get("error"):
            logger.error(f"[optimize_stream] 评分失败: {state['error']}")
            yield {"type": "error", "message": state["error"]}
            return
        
        logger.info(f"[optimize_stream] 评分完成: {state.get('overall_score')}")
        yield {
            "type": "score",
            "data": {
                "overall_score": state.get("overall_score"),
                "scores": state.get("score_result")
            }
        }
        
        # Step 2: JD 匹配（如果有 JD）
        logger.info(f"[optimize_stream] Step 2: JD 匹配分析")
        match_agent = JDMatchAgent()
        state = match_agent.run(state)
        
        if state.get("error"):
            logger.error(f"[optimize_stream] 匹配分析失败: {state['error']}")
        
        logger.info(f"[optimize_stream] 匹配分析完成")
        yield {
            "type": "suggestions",
            "data": {
                "suggestions": state.get("suggestions"),
                "match_analysis": state.get("match_result")
            }
        }
        
        # Step 3: 润色
        logger.info(f"[optimize_stream] Step 3: 简历润色")
        polish_agent = ResumePolishAgent()
        state = polish_agent.run(state)
        
        if state.get("error"):
            logger.error(f"[optimize_stream] 润色失败: {state['error']}")
        
        logger.info(f"[optimize_stream] 润色完成")
        yield {
            "type": "polished",
            "data": {
                "optimized_resume": state.get("optimized_resume")
            }
        }
        
        # 最终完成
        logger.info(f"[optimize_stream] 全部完成")
        yield {
            "type": "complete",
            "data": {
                "overall_score": state.get("overall_score"),
                "scores": state.get("score_result"),
                "suggestions": state.get("suggestions"),
                "optimized_resume": state.get("optimized_resume"),
                "match_analysis": state.get("match_result")
            }
        }


# 全局工作流实例
_resume_workflow: Optional[ResumeOptimizationWorkflow] = None


def get_resume_workflow() -> ResumeOptimizationWorkflow:
    """获取简历优化工作流实例"""
    global _resume_workflow
    if _resume_workflow is None:
        _resume_workflow = ResumeOptimizationWorkflow()
    return _resume_workflow
