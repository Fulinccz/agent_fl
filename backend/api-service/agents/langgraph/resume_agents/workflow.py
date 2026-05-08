"""工作流定义和对外接口"""
from typing import Dict, Any, Optional, Generator
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, END

from agents.registry import get_agent
from logger import get_logger
from .state import ResumeState
from .score_agent import ResumeScoreAgent
from .match_agent import JDMatchAgent
from .polish_agent import ResumePolishAgent

logger = get_logger(__name__)

# 全局共享的LLM实例，避免重复加载模型
_shared_llm_instance: Optional[Any] = None
_model_loading = False
_model_loaded = False


def preload_model():
    """后台预加载模型（在 lifespan 中调用）"""
    global _shared_llm_instance, _model_loading, _model_loaded
    if _model_loaded or _model_loading:
        return
    _model_loading = True
    try:
        logger.info("[workflow] 后台预加载LLM模型...")
        _shared_llm_instance = get_agent(provider="local")
        _model_loaded = True
        logger.info("[workflow] LLM模型预加载完成")
    except Exception as e:
        logger.error(f"[workflow] 模型预加载失败: {e}")
    finally:
        _model_loading = False


def get_shared_llm():
    """获取全局共享的LLM实例（如未加载则同步加载）"""
    global _shared_llm_instance
    if _shared_llm_instance is None:
        logger.info("[workflow] 初始化全局共享LLM实例")
        _shared_llm_instance = get_agent(provider="local")
    return _shared_llm_instance


def create_resume_workflow():
    """创建简历优化工作流 - 共享同一个LLM实例"""
    shared_llm = get_shared_llm()

    score_agent = ResumeScoreAgent(llm=shared_llm)
    match_agent = JDMatchAgent(llm=shared_llm)
    polish_agent = ResumePolishAgent(llm=shared_llm)

    workflow = StateGraph(ResumeState)

    workflow.add_node("score", score_agent.run)
    workflow.add_node("match", match_agent.run)
    workflow.add_node("polish", polish_agent.run)

    workflow.set_entry_point("score")
    workflow.add_edge("score", "match")
    workflow.add_edge("match", "polish")
    workflow.add_edge("polish", END)

    return workflow.compile()


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
        """执行简历优化工作流"""
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

        final_state = self.workflow.invoke(initial_state)

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
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行简历优化工作流"""
        logger.info(f"[optimize_stream] 开始流式执行")

        # 使用全局共享的LLM实例
        shared_llm = get_shared_llm()

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

        # Step 1 & 2: 评分和 JD 匹配并行执行
        logger.info(f"[optimize_stream] Step 1&2: 评分和JD匹配并行执行")
        score_agent = ResumeScoreAgent(llm=shared_llm)
        match_agent = JDMatchAgent(llm=shared_llm)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            score_future = executor.submit(score_agent.run, state.copy())
            match_future = executor.submit(match_agent.run, state.copy())
            
            score_result = score_future.result()
            match_result = match_future.result()
        
        # 合并结果
        if score_result.get("error"):
            logger.error(f"[optimize_stream] 评分失败: {score_result['error']}")
            yield {"type": "error", "message": score_result["error"]}
            return
        
        state["score_result"] = score_result.get("score_result")
        state["overall_score"] = score_result.get("overall_score")
        
        if match_result.get("error"):
            logger.error(f"[optimize_stream] 匹配分析失败: {match_result['error']}")
        
        state["match_result"] = match_result.get("match_result")
        state["suggestions"] = match_result.get("suggestions")

        logger.info(f"[optimize_stream] 评分完成: {state.get('overall_score')}")
        yield {
            "type": "score",
            "data": {
                "overall_score": state.get("overall_score"),
                "scores": state.get("score_result")
            }
        }

        logger.info(f"[optimize_stream] 匹配分析完成")
        yield {
            "type": "suggestions",
            "data": {
                "suggestions": state.get("suggestions"),
                "match_analysis": state.get("match_result")
            }
        }

        # Step 3: 润色（流式）
        logger.info(f"[optimize_stream] Step 3: 简历润色（流式）")
        polish_agent = ResumePolishAgent(llm=shared_llm)

        accumulated_text = ""
        last_yielded_length = 0
        
        for chunk in polish_agent.run_stream(state):
            if chunk.get("type") == "token":
                accumulated_text += chunk.get("content", "")
                
                # 每累积50个字符才输出，减少渲染次数，不做清理（避免不完整内容被过滤）
                if len(accumulated_text) - last_yielded_length >= 50:
                    yield {
                        "type": "polished",
                        "data": {
                            "optimized_resume": accumulated_text,
                            "partial": True
                        }
                    }
                    last_yielded_length = len(accumulated_text)
                        
            elif chunk.get("type") == "error":
                logger.error(f"[optimize_stream] 润色失败: {chunk.get('message')}")
                state["error"] = chunk.get('message')

        # 最终清理后的结果
        logger.info(f"[optimize_stream] 完整生成内容长度: {len(accumulated_text)}")
        
        final_result = polish_agent._clean_result(accumulated_text, state["resume"])
        
        logger.info(f"[optimize_stream] 清理后内容长度: {len(final_result)}")
        
        if not final_result or len(final_result) < 50:
            logger.warning(f"[optimize_stream] 清理后内容太短({len(final_result)}字符)，使用原始简历")
            final_result = state["resume"][:2000]

        state["optimized_resume"] = final_result
        state["current_step"] = "polish_completed"

        logger.info(f"[optimize_stream] 润色完成，长度: {len(final_result)}")
        
        yield {
            "type": "polished",
            "data": {
                "optimized_resume": final_result,
                "partial": False
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


def optimize_stream(resume: str, jd: Optional[str] = None, position_type: Optional[str] = None):
    """流式优化简历的便捷函数"""
    workflow = get_resume_workflow()
    yield from workflow.optimize_stream(resume, jd, position_type)
