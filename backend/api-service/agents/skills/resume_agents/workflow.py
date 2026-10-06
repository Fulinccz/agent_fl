"""简历优化工作流（降级路径）

仅在 MAS 主路径失败时作为兜底。手动串行执行 score → match → polish，
不依赖 LangGraph StateGraph（主路径 MAS 直接调度单个 Agent）。
"""
from typing import Dict, Any, Optional, Generator

from agents.llm import get_shared_llm
from logger import get_logger
from .state import ResumeState
from .score_agent import ResumeScoreAgent
from .match_agent import JDMatchAgent
from .polish_agent import ResumePolishAgent

logger = get_logger(__name__)


def _initial_state(resume: str, jd: Optional[str], position_type: Optional[str]) -> ResumeState:
    return {
        "resume": resume,
        "jd": jd,
        "position_type": position_type,
        "score_result": None,
        "match_result": None,
        "overall_score": None,
        "suggestions": None,
        "optimized_resume": None,
        "error": None,
        "current_step": "started",
    }


class ResumeOptimizationWorkflow:
    """简历优化工作流 - 对外接口（降级兜底用）"""

    def optimize_stream(
        self,
        resume: str,
        jd: Optional[str] = None,
        position_type: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行简历优化（provider/model 请求级切换）"""
        logger.info(f"[optimize_stream] 开始流式执行")

        shared_llm = get_shared_llm(provider=provider, model=model)
        state = _initial_state(resume, jd, position_type)

        # Step 1 & 2: 评分和 JD 匹配（串行执行——LocalProvider 单实例不支持并发推理）
        logger.info(f"[optimize_stream] Step 1&2: 评分和JD匹配（串行）")
        score_agent = ResumeScoreAgent(llm=shared_llm)
        match_agent = JDMatchAgent(llm=shared_llm)

        score_result = score_agent.run(state.copy())
        match_result = match_agent.run(state.copy())

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
                "scores": state.get("score_result"),
            },
        }

        logger.info(f"[optimize_stream] 匹配分析完成")
        yield {
            "type": "suggestions",
            "data": {
                "suggestions": state.get("suggestions"),
                "match_analysis": state.get("match_result"),
            },
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
                            "partial": True,
                        },
                    }
                    last_yielded_length = len(accumulated_text)

            elif chunk.get("type") == "error":
                logger.error(f"[optimize_stream] 润色失败: {chunk.get('message')}")
                state["error"] = chunk.get("message")

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
                "partial": False,
            },
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
                "match_analysis": state.get("match_result"),
            },
        }


# 全局工作流实例
_resume_workflow: Optional[ResumeOptimizationWorkflow] = None


def get_resume_workflow() -> ResumeOptimizationWorkflow:
    """获取简历优化工作流实例"""
    global _resume_workflow
    if _resume_workflow is None:
        _resume_workflow = ResumeOptimizationWorkflow()
    return _resume_workflow


def optimize_stream(resume: str, jd: Optional[str] = None, position_type: Optional[str] = None,
                    provider: Optional[str] = None, model: Optional[str] = None):
    """模块级便捷入口（透传 provider/model）"""
    workflow = get_resume_workflow()
    yield from workflow.optimize_stream(resume, jd, position_type, provider, model)
