# 代码架构整理去冗余 实施计划

## 仓库调研结论

当前 `agents/` 目录存在 **3 类冗余/架构问题**：

### 问题 1：LLM 共享单例双重实现 + 双向依赖（核心问题）
- `agents/langgraph/resume_agents/workflow.py` 有一套：`get_shared_llm()`、`preload_model()`、`_shared_llm_instance`、`_llm_lock`
- `agents/mas/llm.py` 也有一套：`get_shared_llm()`、`_shared_llm`、`_extra_llms`、`_provider_enabled()`（成本开关）
- `mas/llm.py` 默认路径调用 `workflow.get_shared_llm()` → **上层依赖下层**
- `conversation_graph.py` 调用 `mas.llm.get_shared_llm` → **下层依赖上层**
- 形成 `mas ↔ langgraph` 双向依赖，层次混乱

### 问题 2：`resume_agents.py` 死代码
- `agents/langgraph/resume_agents.py` 与 `agents/langgraph/resume_agents/` **包同名**
- Python 包优先于同名模块，该文件被完全遮蔽，实际零引用
- 所有引用都走 `resume_agents.workflow` 或 `resume_agents` 包

### 问题 3：`create_resume_workflow()` LangGraph 图基本无用
- `workflow.py` 的 `create_resume_workflow()` 构建了 `score → match → polish` StateGraph
- 只在 `ResumeOptimizationWorkflow.__init__` 赋值给 `self.workflow`
- `self.workflow` 仅被非流式 `optimize()` 使用 `invoke()`
- **主降级路径 `optimize_stream()` 完全不用此图**，手动串行调三个 agent
- 主路径（MAS）也直接调单个 agent，不用此图
- 即 StateGraph 只服务于一个非流式方法，属于过度设计

### 问题 4：`workflow.py` 职责混乱
- 一个文件混合了：LLM 单例管理 + LangGraph 图编排 + 流式手动编排 + 全局工作流单例

## 目标架构（清晰分层）

```
agents/
├── llm.py              ← ★ 新建：统一 LLM 共享单例（合并 workflow.py + mas/llm.py）
├── registry.py         ← Provider 注册中心（不变）
├── providers/          ← Provider 实现（不变）
├── mas/                ← 多智能体调度层（主大脑）
│   ├── llm.py          ← 改为转发 agents.llm（保持向后兼容）
│   └── ...
└── langgraph/
    ├── resume_agents/  ← 专业技能 Agent（被 MAS 调度）
    │   ├── workflow.py ← 清理后只保留降级流式编排
    │   └── ...
    └── conversation_graph.py  ← 降级兜底对话
```

**依赖方向**：`mas → langgraph.resume_agents`（调度技能），`mas → agents.llm`，`langgraph → agents.llm`。消除双向依赖。

## 涉及文件

| 文件 | 改动 |
|---|---|
| `agents/llm.py` | **新建**：合并 LLM 单例逻辑（workflow.py + mas/llm.py） |
| `agents/mas/llm.py` | 改为 `from agents.llm import ...` 转发 |
| `agents/langgraph/resume_agents/workflow.py` | 删 LLM 单例代码、删 `create_resume_workflow()` StateGraph、`optimize()` 改手动编排 |
| `agents/langgraph/resume_agents.py` | **删除**（死代码） |
| `agents/langgraph/resume_agents/__init__.py` | 移除 `get_shared_llm` 导出 |
| `main.py` | `preload_model` 改从 `agents.llm` 导入 |
| `tests/test_workflow.py` | 更新 mock 路径（`workflow.get_shared_llm` → `agents.llm.get_shared_llm`） |

## 实施步骤（依赖顺序）

### Step 1：新建 `agents/llm.py`，合并 LLM 单例
- 合并 `workflow.py` 的本地单例（`_shared_llm_instance` + 锁 + `preload_model`）
- 合并 `mas/llm.py` 的 provider 缓存 + 成本开关 `_provider_enabled`
- 对外暴露：`get_shared_llm(provider, model)`、`preload_model()`
- 本地默认走 `get_agent(provider="local")` 单例；在线 provider 按 `(provider, model)` 缓存；未启用时回落本地

### Step 2：`mas/llm.py` 改为转发模块
- `from agents.llm import get_shared_llm, preload_model` 并重新导出
- 保持 `mas/__init__.py` 导出不变（向后兼容）

### Step 3：清理 `resume_agents/workflow.py`
- 删除：`_shared_llm_instance`、`_model_loading`、`_model_loaded`、`_llm_lock`、`get_shared_llm()`、`preload_model()`
- 删除：`create_resume_workflow()` 函数
- `ResumeOptimizationWorkflow.__init__`：删除 `self.workflow = create_resume_workflow()`
- `optimize()`：改为手动串行调三个 agent（与 `optimize_stream` 一致，不再用 StateGraph）
- `optimize_stream()`：`get_shared_llm` 改从 `agents.llm` 导入

### Step 4：删除 `resume_agents.py`
- 确认零引用后删除

### Step 5：更新 `resume_agents/__init__.py`
- 移除 `get_shared_llm` 从 `__all__` 和 import

### Step 6：更新 `main.py`
- `from agents.llm import preload_model`

### Step 7：更新 `tests/test_workflow.py`
- mock 路径从 `agents.langgraph.resume_agents.workflow.get_agent` 改为 `agents.llm.get_agent`
- 移除对 `wf_module._shared_llm_instance` 的重置（该变量已不存在）
- `test_workflow_initialization` 不再断言 `workflow.workflow`（StateGraph 已删）

### Step 8：回归验证
- 后端 pytest 全量
- 前端 tsc

## 依赖与注意事项
- `mas/llm.py` 必须保留转发，因为 `mas/__init__.py` 导出了 `get_shared_llm`，且可能有其他地方 import
- `agents.llm` 不能 import `mas` 或 `langgraph`（避免循环），只能 import `agents.registry.get_agent` 和 `services.app_config`
- `conversation_graph.py` 里 `from agents.mas.llm import get_shared_llm` 可保留（转发模块），或改 `agents.llm`（更直接）
- 流式 `optimize_stream` 的事件契约（score → suggestions → polished → complete）必须保持不变

## 验证
- `pytest -q` 全量通过（重点：test_workflow.py、test_mas.py、test_mas_routes.py）
- `grep` 确认无残留引用 `workflow.get_shared_llm`、`workflow.preload_model`、`workflow.create_resume_workflow`
- 确认 `agents.langgraph.resume_agents.py` 已删除且 import 不报错

## 风险与处理
- **测试 mock 路径变更**：test_workflow.py 大量 patch `workflow.get_agent`，需同步改 `agents.llm.get_agent`；若漏改会导致测试失败 → 已列入 Step 7
- **`optimize()` 行为变化**：从 StateGraph.invoke 改为手动串行，输出结构应一致（都是三个 agent 的 state 合并）→ 验证 optimize 返回字段不变
- **循环 import**：`agents.llm` 只依赖 `agents.registry` + `services.app_config`，不依赖 mas/langgraph，确保无循环
