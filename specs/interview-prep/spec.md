# 面试攻略模块（interview-prep skill）Spec

> Spec Mode 产物（临时规划文档）。实现验收后可按用户要求删除 `specs/` 目录。
> 已确认决策：① 输出结构化卡片，不做 Markdown 流式；② 多轮模拟面试为二期，本期不建会话状态；③ 触发＝意图识别 + 输入框上方快捷操作条。

## 1. 目标

在 FulinAI（简历优化 Demo）新增「面试攻略」技能：基于简历（可选 JD）生成**分维度的面试题库卡片**，每题包含考察点、答题要点、可能追问。

## 2. 范围外（本期不做）

- 多轮模拟面试 / 追问对话 / 会话状态
- 面试攻略导出 docx
- 语音面试
- 新增 REST 路由或改动 Go 网关（沿用 `/api/resume/optimize/stream` 与 `/api/agent/upload_stream` 的 SSE 通道）

## 3. 关键现状（影响实现方式）

| 发现 | 位置 | 结论 |
| --- | --- | --- |
| 两个主流式端点硬编码 `preset_intent = INTENT_RESUME_OPTIMIZE` 且 `run_stream("", context, preset_intent)` | [resume.py](file:///f:/Working/FulinAI/backend/api-service/api/routes/resume.py#L81-L88) / [upload.py](file:///f:/Working/FulinAI/backend/api-service/api/routes/upload.py#L120-L127) | 关键词/向量/L3 意图识别对主链路**不生效**；快捷按钮必须通过请求显式透传 intent，否则永远走 resume-optimize |
| `_post_process` 用 `_SIMPLE_SKILL_AGENTS` 反推 known_skills | [intent.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/intent.py#L240-L251) | 必须同步加映射，否则 `interview-prep` 被归一化为 chat-general |
| planner 未知执行器白名单写死 6 个 | [planner.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/planner.py#L256-L263) | 需加 `interview_prep` |
| chat-general 向量样本含 `面试技巧`、`怎么准备面试` | [skill_query_samples.json](file:///f:/Working/FulinAI/backend/api-service/skillhub/skill_query_samples.json#L118-L124) | 会与新技能抢分，需移除 |
| `EXACT_PATTERNS` 命中即返回置信度 1.0 | [keyword_matcher.py](file:///f:/Working/FulinAI/backend/api-service/skillhub/keyword_matcher.py#L51-L70) | 快捷词可走精确模式 |
| `_legacy_events` 只映射 score/suggestions/condensed | [master.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/master.py#L574-L596) | 需加 `interview` 事件 |

## 4. 数据契约

### 4.1 面试攻略产物

```json
{
  "position": "后端开发工程师（示例，来自 JD 或简历推断）",
  "source": "llm | rule",
  "dimensions": [
    {
      "key": "project",
      "label": "项目经历深挖",
      "focus": "围绕简历中的核心项目追问技术选型与个人贡献",
      "questions": [
        {
          "question": "你在这个项目中负责哪一部分？为什么这样设计？",
          "assess_point": "判断候选人是否真正主导过模块设计",
          "answer_points": ["先讲业务背景与目标", "说明方案对比与取舍", "给出结果数据"],
          "follow_up": "如果流量再涨 10 倍，你会先改哪里？"
        }
      ]
    }
  ]
}
```

约束：`dimensions` 最多 4 个；每维度 2-3 题；`answer_points` 最多 4 条；`question` 为空则丢弃。

### 4.2 SSE 事件

```json
{"type": "interview", "data": {"interview_guide": "<4.1 结构>"}}
```

- 非流式 `MasterOrchestrator.run()` 聚合时写入 `final_state["interview_guide"]`
- `complete` 事件 data 中也带 `interview_guide`

## 5. 后端改动（精确到文件/函数）

### 5.1 新增 `backend/api-service/agents/skills/resume_agents/interview_agent.py`

`@dataclass class InterviewPrepAgent`，`name = "interview_prep_agent"`，`llm: Optional[Any]`，`__post_init__` 兜底 `get_agent(provider="local")`（与 match_agent 同构）。

`run(self, state) -> state`：

1. `resume = state.get("resume") or ""`，`jd = state.get("jd") or ""`
2. 简历摘要：`chunks = split_text(resume, 1200)`；`len(chunks) == 1` 用整块；否则取**首块 + 末块**并以 `【简历开头】/【简历结尾】` 标注（禁止 `resume[:N]`）
3. **阶段 A 维度规划**（1 次 LLM，`max_new_tokens=256`）：输出 `{"position": "...", "dimensions":[{"key","label","focus"}]}`，要求 3-4 个维度；`strip_llm_artifacts` → `extract_json_object` → `json.loads`；失败用 `_default_dimensions()`（项目经历深挖 / 技术基础与深度 / 岗位匹配度 / 职业规划与动机）
4. **阶段 B 逐维度出题**（每维度 1 次 LLM，`max_new_tokens=1024`）：`{"questions":[{question, assess_point, answer_points[], follow_up}]}`；解析失败**重试一次**，仍失败 → `_rule_questions(dim_key, resume_excerpt)` 规则题库（静态模板 + 从简历摘要提取的项目/技能关键词）
5. `_normalize_guide()` 按 4.1 约束裁剪
6. 全程 `start_span("llm-interview-dimensions"/"llm-interview-<key>", as_type="generation")` + `update_observation(output=raw[:500])`，只包 try/except（未配置 Langfuse 时 no-op）
7. 任何异常：`state["error"] = f"面试攻略生成失败: {e}"`；`state["interview_guide"] = 规则兜底 guide`；**写回传入 state dict 后 return state**

### 5.2 新增 `backend/api-service/agents/mas/executors/interview.py`

`InterviewPrepExecutor`：
- `name = "interview_prep"`，`description = "基于简历与目标岗位生成分维度面试题库与答题要点"`，`required_inputs = ["resume"]`，`streamable = False`
- `run()` 参考 [jd_match.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/executors/jd_match.py#L23-L67)：`validate` → `_resolve_llm(self._llm, task_input)` → `agent.run(state)` → 兜底错误只放 `data["interview_fallback_error"]`，**`success = True`**（agent 有规则兜底，总有产出）→ `data["interview_guide"]`，`summary = f"生成 {n} 个维度的面试题库"`

### 5.3 注册与意图

| 文件 | 改动 |
| --- | --- |
| [registry.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/executors/registry.py#L26-L35) | import + `_register_default` 在 `JDMatchExecutor` 后追加 `InterviewPrepExecutor(llm=self._llm)` |
| [state.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/state.py#L11-L18) | `INTENT_INTERVIEW_PREP = "interview-prep"` |
| [intent.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/intent.py#L39-L45) | `_SIMPLE_SKILL_AGENTS["interview-prep"] = "interview_prep"` |
| [keyword_matcher.py](file:///f:/Working/FulinAI/backend/api-service/skillhub/keyword_matcher.py#L20-L56) | `SKILL_KEYWORDS` 新增 `interview-prep`：`面试 / 面试题 / 面经 / 面试官 / 面试攻略 / 面试准备 / 怎么准备面试 / 会被问什么 / 问什么 / interview`；`EXACT_PATTERNS` 新增：`面试攻略 / 面试准备 / 面试问题 / 面试题库 / 面试技巧 / 面经` |
| [skill_query_samples.json](file:///f:/Working/FulinAI/backend/api-service/skillhub/skill_query_samples.json) | chat-general 删除 `面试技巧`、`怎么准备面试`；新增 `interview-prep` 20 条样本；执行 `.venv\Scripts\python.exe skillhub\embed_skills.py --rebuild`（`data/skill_vector_db/` 已 gitignore） |

### 5.4 规划器

[planner.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/planner.py#L106-L147)：
- import `INTENT_INTERVIEW_PREP`
- `_template_plan` 新增分支：
  ```python
  elif intent_name == INTENT_INTERVIEW_PREP:
      steps = [TaskStep(step_id="step_1", agent="interview_prep",
                        task="分析简历与目标岗位，生成分维度面试题库与答题要点", depends_on=[])]
  ```
- `_is_valid_agent` 内置白名单加 `"interview_prep"`

### 5.5 Master 事件

[master.py](file:///f:/Working/FulinAI/backend/api-service/agents/mas/master.py)：
- `run()`：新增 `elif etype == "interview": final_state["interview_guide"] = event.get("data", {}).get("interview_guide")`
- `_legacy_events()`：新增
  ```python
  elif step.agent == "interview_prep":
      yield {"type": "interview", "data": {
          "interview_guide": result.data.get("interview_guide"),
      }}
  ```
- `_finalize()`：`artifact_map` 增加 `"interview_guide": "interview_guide"`

### 5.6 请求契约（让快捷按钮真正生效）

| 文件 | 改动 |
| --- | --- |
| [common.py](file:///f:/Working/FulinAI/backend/api-service/api/routes/common.py#L130-L135) | `ResumeOptimizeRequest` 增加 `intent: Optional[str]`（快捷操作显式意图）；新增 `build_preset_intent(raw: Optional[str]) -> IntentResult`，白名单：`interview-prep` → `INTENT_INTERVIEW_PREP`（is_complex=False）、`resume-score` → `INTENT_RESUME_SCORE`、`jd-keyword-match` → `INTENT_JD_MATCH`，空/未知 → 现有 `resume-optimize`（is_complex=True，source="preset"） |
| [resume.py](file:///f:/Working/FulinAI/backend/api-service/api/routes/resume.py#L81-L88) | 改用 `build_preset_intent(data.intent)`，其余流程不变 |
| [upload.py](file:///f:/Working/FulinAI/backend/api-service/api/routes/upload.py#L55-L62) | 函数签名增加 `intent: str = Form("")`，preset 用 `build_preset_intent(intent)` |

> 说明：这是"意图识别 + 快捷按钮"的落地方式——文本链路维持原样（`/chat` 已有真实三层识别），优化链路用显式 intent 白名单，零 LLM 开销、行为确定。

## 6. 前端改动

| 文件 | 改动 |
| --- | --- |
| [types/index.ts](file:///f:/Working/FulinAI/frontend/src/types/index.ts) | `ResumeEventType` 加 `'interview'`；新增 `InterviewQuestion` / `InterviewDimension` / `InterviewGuideData` / `InterviewData`；`ResumeOptimizeRequest` 与 `UploadRequest` 加 `intent?: string` |
| services/api.ts | 粘贴链路 JSON body 自动透传（确认 `intent` 不被过滤）；上传链路 `FormData` 追加 `form.append('intent', ...)`（仅当有值） |
| [useStreamResponse.ts](file:///f:/Working/FulinAI/frontend/src/hooks/useStreamResponse.ts) | 新增 `interview` state + `interviewRef`；`case 'interview'` 读取 `data.interview_guide`；`resetOutput` 清空；返回 `interview` |
| [App.tsx](file:///f:/Working/FulinAI/frontend/src/App.tsx) | 新增 `quickAction: 'interview-prep' \| 'resume-score' \| 'jd-keyword-match' \| null`；输入卡片顶部 `.quick-actions` 三枚 chip（面试攻略 / 简历评分 / JD 匹配，点击 toggle）；提交时 `intent: quickAction ?? undefined` 传给 `startStream/startUpload`；`confirmNewChat` 清空 quickAction；`isInterviewMode = quickAction === 'interview-prep' \|\| !!interview` 时隐藏评分卡 / 优化结果卡 / 导出卡，渲染「面试攻略」卡（位置平级；卡片：position chip + 维度区块 → focus → 题目：题干 / 考察点 / 答题要点 ul / 可能追问） |
| App.css | 新增 `.quick-actions` `.quick-chip(.active)` `.interview-card` `.interview-dim` `.interview-question` `.iq-meta` `.iq-points`，复用既有 card/border/text CSS 变量 |

## 7. 测试

新增 `backend/api-service/tests/test_interview_agent.py`（mock LLM）：
1. 正常：维度 JSON + 每维度题目 JSON → 4.1 结构、question 字段齐全
2. JSON 非法：两次失败 → 规则题库兜底，`source == "rule"`，无异常
3. 空简历：`InterviewPrepExecutor().run({"resume": ""})` → `success is False`，`error` 含"缺少必需输入"
4. 执行器成功：注入 fake llm → `success is True`，`data["interview_guide"]["dimensions"]` 非空

新增 `backend/api-service/tests/test_interview_flow.py`：
5. `KeywordMatcher().match("帮我准备面试")` → `interview-prep`，置信度 ≥ 0.9
6. `MasterAgent(executor_catalog="")._is_valid_agent("interview_prep")` → True
7. `MasterAgent().plan("面试攻略", IntentResult(INTENT_INTERVIEW_PREP))` → 单步且 `agent == "interview_prep"`
8. `MasterOrchestrator._legacy_events`（构造 StepResult）→ 产出 `type == "interview"`
9. `build_preset_intent("interview-prep")` → name/intent 正确；`build_preset_intent(None)` → resume-optimize

## 8. 验证步骤

```powershell
cd backend/api-service; ..\..\.venv\Scripts\python.exe -m pytest tests\ -q   # 基线 484，预期 +9
cd backend/api-service; ..\..\.venv\Scripts\python.exe -m ruff check .
cd backend/api-service; ..\..\.venv\Scripts\python.exe skillhub\embed_skills.py --rebuild
cd frontend; npx tsc --noEmit; npm run lint
powershell -File .\start.ps1
# POST /api/resume/optimize/stream  {"resume":"...","intent":"interview-prep"} → 断言出现 "type":"interview"
# 浏览器 5173：点击「面试攻略」→ 发送 → 卡片渲染
```

不提交 git；除本 spec 目录外不新增文档。

## 9. 风险与取舍

- 本地 4B 模型 JSON 不稳 → 双层兜底（逐维度重试一次 + 规则题库），保证始终有卡片
- 最多 5 次 LLM 调用（1 维度规划 + 4 出题）→ 本地耗时偏长，前端复用现有 `isStreaming`/停止按钮
- `chat-general` 关键词含 `建议`/`指南`，`面试攻略` 走精确模式 1.0 优先，不会误判
- 快捷按钮覆盖简历评分 / JD 匹配，复用同一 `intent` 白名单，不新增端点

## 10. 待批准

请确认以上方案；确认后按 `tasks.md` 顺序实现。
