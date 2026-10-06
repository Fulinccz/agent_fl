"""
简历优化相关路由
/resume/*

主从多智能体协作简历优化流水线：
Master 规划（评分/匹配 并行步骤 → 润色）→ 执行器执行 → ReAct 反思 → 流式输出。

事件契约与旧版保持一致：score → suggestions → polished → complete，
另附带 intent / plan / step_start / step_result / reflection 过程事件。
编排器异常时自动降级到旧版 LangGraph 工作流。
"""

from __future__ import annotations
import json
import asyncio
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import iterate_in_threadpool

from logger import get_logger
from agents.mas.state import IntentResult, INTENT_RESUME_OPTIMIZE
from .common import ResumeOptimizeRequest

router = APIRouter(tags=["简历"])
logger = get_logger(__name__)


@router.post(
    "/optimize/stream",
    summary="流式简历优化",
    description="""
启动主从多智能体协作的简历优化流程，流式返回各阶段结果。

**执行过程：**
1. Master 规划：评分、JD 匹配（可并行）→ 简历润色
2. 各步骤由执行器执行，ReAct 反思器审查结果（失败自动重试/重规划）

**事件契约（与旧版一致）：**
- `score` - 简历评分（匹配度、完整性、专业度）
- `suggestions` - 优化建议（具体改进点）
- `polished` - 润色后的完整简历（partial=true 为流式中间结果）
- `complete` - 全部完成

**过程事件（新增，前端可忽略）：**
- `intent` / `plan` / `step_start` / `step_result` / `reflection`

**请求参数：**
- `resume`（必填）：简历原文
- `jd`（可选）：目标职位描述，用于针对性优化
- `position_type`（可选）：职位类型

**响应格式：** `application/json`（每行一个 JSON 对象）
    """,
    responses={
        200: {"description": "流式响应"},
        500: {"description": "优化流程错误"},
    },
)
async def resume_optimize_stream(request: Request, data: ResumeOptimizeRequest):
    call_time = datetime.now().strftime('%H:%M:%S')
    logger.info(f"[{call_time}] Resume optimize stream started")

    # 预置意图：resume 路由的场景意图明确，跳过意图识别
    preset_intent = IntentResult(
        name=INTENT_RESUME_OPTIMIZE,
        confidence=1.0,
        source="preset",
        reason="简历优化端点预置意图",
        is_complex=True,
    )

    async def event_stream():
        got_result = False
        mas_failed = False
        try:
            from api.routes.deps import get_master_orchestrator
            orchestrator = get_master_orchestrator()

            context = {
                "resume": data.resume,
                "jd": data.jd,
                "position_type": data.position_type,
            }
            # 模型切换：透传到 MAS 黑板，执行器按此选择 Provider
            if data.provider:
                context["provider"] = data.provider
            if data.model:
                context["model"] = data.model

            async for event in iterate_in_threadpool(
                orchestrator.run_stream("", context, preset_intent)
            ):
                if await request.is_disconnected():
                    logger.info(f"[{call_time}] Client disconnected")
                    break

                etype = event.get("type")
                if etype in {"score", "suggestions", "polished", "complete"}:
                    got_result = True
                yield json.dumps(event, ensure_ascii=False) + "\n"
                await asyncio.sleep(0)

            logger.info(f"[{call_time}] Resume optimize stream completed (MAS)")

        except Exception as e:
            logger.error(f"[{call_time}] MAS optimize error: {e}")
            mas_failed = True

        # 降级回退：旧版 LangGraph 工作流
        # 在 MAS 异常、或完整执行但未产出任何结果事件时回退，避免用户拿到空结果
        if not got_result:
            logger.warning(f"[{call_time}] Falling back to legacy resume workflow")
            try:
                from agents.skills.resume_agents import get_resume_workflow

                workflow = get_resume_workflow()
                for event in workflow.optimize_stream(
                    resume=data.resume,
                    jd=data.jd,
                    position_type=data.position_type,
                    provider=data.provider,
                    model=data.model
                ):
                    if await request.is_disconnected():
                        logger.info(f"[{call_time}] Client disconnected")
                        break

                    yield json.dumps(event, ensure_ascii=False) + "\n"
                    await asyncio.sleep(0)

                logger.info(f"[{call_time}] Resume optimize stream completed (legacy)")
            except Exception as e:
                logger.error(f"[{call_time}] Resume optimize stream error: {e}")
                yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/json")
