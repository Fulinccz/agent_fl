# 面试攻略模块 Tasks

> 对应 [spec.md](file:///f:/Working/FulinAI/specs/interview-prep/spec.md)。逐项完成，每项自验证。

## T1. 面试 Agent（后端）
- [x] 新建 `backend/api-service/agents/skills/resume_agents/interview_agent.py`
- 验收：`InterviewPrepAgent.run(state)` 返回同 dict；正常 JSON → `source="llm"` 且维度 3-4、每题含 question/assess_point/answer_points/follow_up；两次 JSON 失败 → `source="rule"`；异常 → `state["error"]` + 规则兜底
- 约束：`split_text` 取首末块（禁 `resume[:N]`）；`strip_llm_artifacts`；`max_new_tokens` 256/1024；tracing 只用 `update_observation`

## T2. 执行器 + 注册（后端）
- [x] 新建 `agents/mas/executors/interview.py`（`name="interview_prep"`，`required_inputs=["resume"]`，`streamable=True`）
- [x] 逐维度增量：agent 增加 `on_progress` 回调，执行器用后台线程 + 队列回传，每完成 1 个维度先推一份题卡；等待期间每 10s 发 `heartbeat` 保活 SSE
- [x] `executors/registry.py` 注册
- 验收：`InterviewPrepExecutor().run({"resume": ""})` → `success=False`；有简历 + fake llm → `success=True` 且 `data["interview_guide"]` 非空；`run_stream` 依次产出 1 维 → 2 维增量（均 `partial=True`）

## T3. 意图与关键词（后端）
- [x] `mas/state.py` 加 `INTENT_INTERVIEW_PREP`
- [x] `mas/intent.py` 加 `_SIMPLE_SKILL_AGENTS` 映射
- [x] `skillhub/keyword_matcher.py` 加 `SKILL_KEYWORDS` + `EXACT_PATTERNS`
- [x] `skillhub/skill_query_samples.json`：chat-general 移除两条冲突样本，新增 interview-prep 20 条
- [x] 重建向量库 `embed_skills.py --rebuild`（123 条样本入库，dim=512，collection=skill_router）
- 验收：`KeywordMatcher().match("帮我准备面试").skill == "interview-prep"`；`match("面试攻略").confidence == 1.0`

## T4. 规划器（后端）
- [x] `mas/planner.py`：`_template_plan` 加分支（单步）+ `_is_valid_agent` 白名单加 `interview_prep`
- 验收：`plan("面试攻略", IntentResult(INTENT_INTERVIEW_PREP)).steps == [{agent: interview_prep}]`

## T5. Master 事件（后端）
- [x] `mas/master.py`：`run()` 聚合、`_legacy_events`、`_finalize` artifact_map 三处加 `interview`
- [x] `_iter_streaming`：透传 `interview` 增量与 `heartbeat`，按最后一份增量聚合出 `StepResult`
- 验收：`_legacy_events` 对 `agent="interview_prep"` 产出 `{"type":"interview"}`；`complete` 带 `interview_guide`

## T6. 请求契约（后端）
- [x] `api/routes/common.py`：`ResumeOptimizeRequest.intent` + `build_preset_intent()`
- [x] `api/routes/resume.py`、`api/routes/upload.py` 改用它
- 验收：`intent="interview-prep"` → preset name=`interview-prep`、is_complex=False；不传 → `resume-optimize`（旧行为不变）

## T7. 前端
- [x] `types/index.ts`：事件类型 + 接口 + `intent` 字段
- [x] `services/api.ts`：上传 FormData 透传 intent（粘贴链路确认 JSON 已带）
- [x] `hooks/useStreamResponse.ts`：`interview` state/case/reset
- [x] `App.tsx`：快捷操作条（面试攻略/简历评分/JD 匹配）+ 面试攻略卡片 + 模式切换隐藏旧卡片
- [x] `App.tsx`：快捷操作已显式给出意图 → 跳过 `/api/skill/route`（本地链路该调用会占用推理槽位 20–30s）
- [x] `App.tsx`：面试卡片渐进提示（「N 个维度已完成，继续生成」+ 生成期空状态文案）
- [x] `App.css`：快捷条与卡片样式
- [x] `vite.config.ts`：/api 代理超时 300s → 900s（本地面试攻略单次生成超 5 分钟，300s 会在完成前截断 SSE）
- 验收：`npx tsc --noEmit` 与 `npm run lint` 0 error；点击「面试攻略」→ 发送 → 卡片渲染 dimension/questions

## T8. 测试与回归
- [x] `tests/test_interview_agent.py`（12 例，含跨维度去重、维度上限、增量流式、维度收敛、取消、模型缺失降级）
- [x] `tests/test_interview_flow.py`（6 例）
- [x] `pytest tests/ -q`（502 passed）、`ruff check .`（All checks passed）
- [x] `npx tsc --noEmit`、`npm run lint`（均 0 error）
- [x] 起服务实测（intent=interview-prep，经 Vite 代理）：225.6s 完成，88.4s 出第 1 维、154.5s 出第 2 维、225.6s 出第 3 维，心跳 10s 一次（共 21 次），3 维 9 题、source=llm
- [x] 浏览器实测：卡片维度数随时间递增 0 → 1（~120s）→ 2（~180s）→ 3（~240s），3 维 6 题，每题含考察点/答题要点/可能追问，无错误横幅

## T9. 收尾
- [x] 汇总改动文件与验证结果；未提交 git；征询是否删除 `specs/` 目录

## T10. 复盘加固
- [x] 维度收敛：LLM 规划出的同主题维度先按相似度合并，再用默认维度补齐到 3 个（避免多跑一次完整出题）
- [x] 增量推送去重：新维度的题目若被整维剔除，不重复推送同一份题库
- [x] 取消：客户端断开后执行器置位停止标志，agent 在维度之间提前收手（实测第 1 维后断开，仅再跑 1 维即结束；T11 改为并发出题后，改为规划阶段断开即整批跳过）
- [x] 降级：LLM/Provider 解析或加载失败不再让整步失败（避免 Reflector 无谓重试），直接落规则题库
- [x] 提示词统一为「简历 + JD 在前、任务在后」的公共前缀（连续调用的固定前缀一致）
- [x] 归一化兜底改用真实简历/JD 生成规则题，不再退化成通用占位题
- [x] `pytest tests/ -q`（502 passed）、`ruff check .`（All checks passed）

## T11. 性能优化（解码速度）
- [x] llama-server 开 3 个并行槽位：`--parallel 3 -c 12288`（每槽 4096，与改造前上下文持平）。CPU 上并发出题实测总吞吐 11.7 tok/s（串行 7.0），三维度并发出题 116s → 串行约 198s
- [x] 提示词要求紧凑 JSON（单行、无缩进、无围栏）：单维输出 token 465 → 约 320，答题要点固定 3 条（每条 ≤18 字）
- [x] 三维度改并发出题（`ThreadPoolExecutor`），仍按维度顺序推送增量；断开后不再发起整批出题
- [x] `MAX_ANSWER_POINTS` 4 → 3：挡住模型自纠时多写的第 4 条要点
- [x] 前端面试卡片加生成秒表（`正在生成 · Ns`），首屏增量前不再只有脉冲动画
- 实测（经 Vite 代理，真实 GGUF）：全流程 **106.6s**（优化前 225.6s，2.1×），102.6s 出第 1 维、106.6s 出第 3 维，3 维 9 题、`source=llm`，每题 3 条要点
- 验证：`pytest tests/ -q`（502 passed）、`ruff check .`、`npx tsc --noEmit`、`npm run lint` 全通过；浏览器实测秒表递增（8s → 17s → 22s）、无错误横幅

## T12. 冗余清理
- [x] 执行器 `_worker` 的 `state["error"] = str(e)` 是死写（流式路径无人读 state），删除
- [x] 执行器注释/文档从「逐维度串行推送」改为「三维度并发出题，完成后按序推送」，避免误导维护者
- [x] 前端流式提示合并：并发出题后「N 个维度已完成，继续生成」几乎不渲染，改为 `已出 N 个维度 · Ns` / `正在生成 · Ns`
- [x] 出题 `max_new_tokens` 1024 → 640（实测单维约 320 token，保留 2× 余量并给异常啰嗦输出封顶）
- [x] 保留但已确认非冗余：重试前的 `_stop_requested()`（断开后不再发起重试解码）、`_emitted_dims` 守卫（整维被去重不重复推送）、`_normalize_guide` 的二次去重（最终产物兜底，n≤9）
- 复测：107.7s 完成，3 维 9 题、每题 3 条要点、`source=llm`，无异常要点文本
