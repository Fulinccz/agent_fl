# 合并技能实现 + 清理冗余 + 目录优化 实施计划

## 仓库调研结论

系统存在两套并行的简历技能实现，功能重复：

| 维度 | skill_creator/（旧路径） | agents/skills/resume_agents/（MAS 路径） |
|---|---|---|
| 评分 | `resume_score.resume_score()` | `ResumeScoreAgent.run()` |
| 润色 | `resume_polishing.polish_resume()` | `ResumePolishAgent.run()` |
| 匹配 | `jd_keyword_match.match_jd_keywords()` | `JDMatchAgent.run()` |

旧路径被 `skill.py`/`upload.py` 路由 + 前端 `useSkillRouter.ts` 使用；MAS 路径被 `chat.py` 使用。

**合并策略**：保留 skill_creator 的函数签名（`score_resume`/`polish_resume`/`match_jd_keywords`）和注册机制不变，内部改为调用 `agents/skills/resume_agents/` 的 Agent，消除重复的 LLM 逻辑和 prompt。

返回结构适配（保持原签名，减少前端改动）：
- `score_resume` → 调 `ResumeScoreAgent`，`overall_score` 直接取自 state，`scores` 取自 `score_result`，`suggestions` 调 `JDMatchAgent` 取 `suggestions`
- `polish_resume` → 调 `ResumePolishAgent`，`polished_content` = `optimized_resume`
- `match_jd_keywords` → 调 `JDMatchAgent`，`match_analysis` = `match_result`，`extracted_keywords` 从 match_result 提取或留空，`optimized_resume` 留空

## 涉及文件

### 合并（第二类）
- `skill_creator/resume_score/resume_score.py` — 改为调用 ResumeScoreAgent
- `skill_creator/resume_polishing/resume_polishing.py` — 改为调用 ResumePolishAgent
- `skill_creator/jd_keyword_match/jd_keyword_match.py` — 改为调用 JDMatchAgent

### 死代码删除（第一类）
- `messaging/` 整个目录（kafka_client.py 零引用）
- `rag/db_based/` 整个目录（零外部引用）
- `rag/text_based/` 整个目录（空壳）
- `rag/graph/test_graph.py`（测试混在源码）

### 目录优化（第三类）
- `generate_knowledge.py` → `scripts/generate_knowledge.py`
- `manage_vector_db.py` → `scripts/manage_vector_db.py`
- `rag/graph/` 按职责拆子目录（可选，见风险）

## 实施步骤

### Step 1：合并 skill_creator 三个技能
1. `resume_score.py`：删除内部 LLM 逻辑，改为 `from agents.skills.resume_agents.score_agent import ResumeScoreAgent` + `from agents.skills.resume_agents.match_agent import JDMatchAgent`，构造 state 调 run，适配返回
2. `resume_polishing.py`：改为调 `ResumePolishAgent.run()`，取 `optimized_resume`
3. `jd_keyword_match.py`：改为调 `JDMatchAgent.run()`，取 `match_result`

### Step 2：删除死代码
- 删除 `messaging/`、`rag/db_based/`、`rag/text_based/`、`rag/graph/test_graph.py`

### Step 3：移动根目录脚本到 scripts/
- 移动 `generate_knowledge.py`、`manage_vector_db.py` 到 `scripts/`
- 无需改引用（都是命令行脚本）

### Step 4：rag/graph 目录拆分（低优先级）
- 拆为 `rag/graph/neo4j/`（graph_builder, neo4j_client, graph_retriever, config, hybrid_retriever）和 `rag/graph/data/`（import_train_data, train_data_builder）
- 需同步修改 import

### Step 5：回归验证
- 后端 pytest 全量
- 前端 tsc
- 验证 skill_creator 技能可正常注册和执行

## 依赖与注意事项
- `skill_creator` 的函数签名和注册名必须保持不变（`__init__.py` 的 skill_mapping 依赖）
- Agent 的 `run(state)` 需要 `ResumeState` 格式的 state，需构造完整 state
- `ResumePolishAgent` 有 `run()` 和 `run_stream()`，skill 走非流式用 `run()`
- 前端对 skill 返回结构耦合低（useSkillRouter 透传），但 App.tsx 可能有字段访问，需验证

## 验证
- pytest 全量通过
- tsc 0 errors
- 手动验证：`from skill_creator import init_skills; init_skills()` 能注册三个技能且执行不报错

## 风险
- **rag/graph 拆分的 import 联动**：拆分后 `rag.graph.graph_builder` 等路径变化，需全局改 import。若风险高可跳过，保持原目录。
- **返回结构差异**：skill_creator 原返回的 `scores` 是 `{dim: {score, comment, suggestions}}`，Agent 的 `score_result` 是 `{dim: int}`。前端若依赖 comment 字段会受影响。处理：适配层尽量补全结构，或前端同步调整。
