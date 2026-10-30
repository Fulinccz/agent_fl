# 面试攻略模块 Checklist

> 逐项核对 spec.md 的验收标准。每项通过后打勾，失败则在 tasks.md 追加修复任务。

## 后端能力

- [x] `InterviewPrepAgent.run(state)` 直接修改并返回传入的 state（含 `interview_guide`），不返回新 dict
- [x] 正常链路 `source="llm"`，维度 3-4、每题含 question / assess_point / answer_points / follow_up
- [x] LLM 不可用或 JSON 两次非法时降级为 `source="rule"`，`state` 不写 `error`（避免 Reflector 无谓重试）
- [x] 简历摘要用 `split_text` 取首末块（无 `resume[:N]` 截断）；JSON 清理复用 `strip_llm_artifacts`
- [x] JSON 输出显式传 `max_new_tokens`（维度规划 256 / 出题 1024）
- [x] tracing 只用持有式 `update_observation`，且包在 try/except 内
- [x] `InterviewPrepExecutor` 满足 BaseExecutor 契约（name/description/required_inputs），空简历 `success=False`
- [x] 增量流式：`streamable=True`，`run_stream` 按维度推 `interview(partial=True)`，等待期每 10s 发 `heartbeat`
- [x] `ExecutorRegistry` 已注册 `interview_prep`，catalog 可见
- [x] 兜底错误只放 `data["interview_fallback_error"]`，不进 `result.error`

## 意图与规划

- [x] `INTENT_INTERVIEW_PREP = "interview-prep"`，`_SIMPLE_SKILL_AGENTS` 映射到 `interview_prep`
- [x] `KeywordMatcher.match("帮我准备面试")` 命中 interview-prep，精确模式 confidence 1.0
- [x] 向量样本库重建完成且不含与 chat-general 冲突的「面试技巧」类样本
- [x] `planner` 模板为单步 `interview_prep`，`_is_valid_agent` 白名单包含它

## 事件契约

- [x] `_legacy_events` 对 interview_prep 产出 `{"type":"interview","data":{"interview_guide":...}}`
- [x] `run()` 聚合把 interview 事件写入 `final_state["interview_guide"]`
- [x] `complete` 事件携带 `interview_guide`（artifact_map）

## 请求契约

- [x] `ResumeOptimizeRequest.intent` 存在，粘贴链路透传到 `build_preset_intent`
- [x] 上传链路 `intent: str = Form("")` 并在 FormData 中透传
- [x] `intent="interview-prep"` → preset 意图 interview-prep、`is_complex=False`；缺省/未知值仍为 resume-optimize 全流程

## 前端

- [x] `types/index.ts` 含 interview 事件类型与 InterviewGuideData 等接口，请求类型含 `intent?`
- [x] `useStreamResponse` 有 interview state + case + reset，ref 与 state 同步更新
- [x] 快捷操作条含 💼面试攻略 / 📊简历评分 / 🎯JD 匹配，可 toggle，流式中禁用
- [x] 快捷操作命中时跳过 `/api/skill/route`，直接带 preset intent 发起生成
- [x] 生成期有渐进反馈：卡片维度数递增提示 + 空状态文案（"正在分析简历、规划考察维度…"）
- [x] 面试模式下隐藏评分卡、优化结果卡、导出卡
- [x] 卡片展示岗位、维度编号与考察重点、题目、考察点、答题要点、可能追问
- [x] `npx tsc --noEmit` 与 `npm run lint` 均 0 error
- [x] `/api` 代理超时足以覆盖 5 分钟以上的本地生成（900s）

## 端到端

- [x] 真实 POST `/api/resume/optimize/stream`（intent=interview-prep，经 Vite 代理）事件序列含 intent/plan/step_start/interview(partial ×3)/interview(final)/reflection/complete
- [x] 长任务保活：106.6s 全程心跳不断流（10s 一次），代理与网关均正常收尾（无 ERR_INCOMPLETE_CHUNKED_ENCODING）
- [x] 浏览器实测：点击「面试攻略」→ 填写简历与 JD → 提交 → 秒表递增（8s→17s→22s）、空状态文案与并行生成说明正常，无 error-banner
- [x] 生成秒表：面试卡片流式期间显示 `正在生成 · Ns`，完成后隐藏
- [x] 客户端在规划阶段断开：不再发起整批出题（日志「客户端已断开，跳过题库生成」），只落规则兜底
- [x] `pytest tests/ -q` 502 passed；`ruff check .` All checks passed
- [x] 未提交 git，未删除 spec 文档
