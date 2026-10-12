# backend/api-service 代码整理方案

## Context

上一轮完成了 provider 链路拆分（providers/local/ + providers/cloud/ 物理隔离）。用户要求继续检查 backend/api-service 下类似的"拆分/整理"机会。经排查发现 4 项问题，用户已确认全部处理。

---

## 1. Memory 双存储统一（保留 SQLite，删除 SimpleMemory）

**问题**：`deps.py:74` 实际用 `SQLiteMemoryStore`（持久化），但 `MemoryManager` 默认 fallback 到 `SimpleMemoryStore`（死代码，且缺 `cleanup_old_sessions` 方法）。`__init__.py` 只导出 Simple 不导出 SQLite。

**改动**：

| 文件 | 操作 |
|------|------|
| `memory/simple_memory.py` | **删除**（死代码，无外部引用） |
| `memory/__init__.py` | 改为导出 `SQLiteMemoryStore` + `get_memory_store` + `ConversationMemory` + `MemoryManager` + `get_memory_manager`；更新 docstring |
| `memory/memory_manager.py` | import 从 `simple_memory` 改为 `sqlite_memory`；类型标注 `SimpleMemoryStore` → `SQLiteMemoryStore`；默认 fallback 从 `get_simple_memory_store()` 改为 `get_memory_store()`；`ConversationMemory` import 路径改 |
| `api/routes/deps.py` | `from memory.sqlite_memory import get_memory_store` → `from memory import get_memory_store`（统一入口） |

---

## 2. Resume 降级路径清理（删除 workflow.py，MAS 异常直接报错）

**问题**：`resume.py` 在 MAS 异常时降级到 `workflow.py`（170 行重复串行逻辑）。MAS 已稳定，降级路径是冗余兜底代码。

**改动**：

| 文件 | 操作 |
|------|------|
| `agents/skills/resume_agents/workflow.py` | **删除** |
| `agents/skills/resume_agents/__init__.py` | 移除 `from .workflow import ...` 和 `__all__` 中的 3 个导出 |
| `api/routes/resume.py` | 删除降级逻辑（line 110-135），MAS 异常时直接 `yield error` 事件；删除 `got_result`/`mas_failed` 标志位和降级判断 |
| `tests/test_workflow.py` | **删除**（专门测试 workflow.py） |
| `tests/test_mas_routes.py` | 删除 `test_resume_mas_failure_falls_back_to_legacy`、`test_resume_legacy_inner_error_emits_error` 两个降级测试；更新 `test_resume_partial_results_no_fallback` 移除降级 mock |

**resume.py 简化后的事件流逻辑**：
```python
async def event_stream():
    try:
        orchestrator = get_master_orchestrator()
        context = {...}
        async for event in iterate_in_threadpool(orchestrator.run_stream(...)):
            if await request.is_disconnected():
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"
            await asyncio.sleep(0)
    except Exception as e:
        logger.error(f"[{call_time}] MAS optimize error: {e}")
        yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"
```

---

## 3. executors.py 拆分为子包

**问题**：`agents/mas/executors.py` 437 行，5 个执行器 + ExecutorRegistry + BaseExecutor 全在一个文件。

**改动**：拆为 `agents/mas/executors/` 子包，每执行器一文件，`__init__.py` re-export 全部（**保持 `from agents.mas.executors import ...` 向后兼容**）：

```
agents/mas/executors/
├── __init__.py          # re-export 全部（BaseExecutor, ExecutorRegistry, 5 个执行器, _resolve_llm）
├── base.py              # BaseExecutor + _resolve_llm + 公共 import（logger, start_span, get_shared_llm, StepResult, prompts）
├── resume_score.py      # ResumeScoreExecutor
├── jd_match.py           # JDMatchExecutor
├── resume_polish.py     # ResumePolishExecutor
├── resume_parse.py      # ResumeParseExecutor
├── chat.py              # ChatExecutor
└── registry.py          # ExecutorRegistry
```

**关键约束**：
- `mas/__init__.py:29` 的 `from .executors import (...)` 不需改（子包 `__init__.py` re-export）
- `master.py:14` 的 `from .executors import ExecutorRegistry, BaseExecutor` 不需改
- `tests/test_mas.py:34` 的 `from agents.mas.executors import (ExecutorRegistry, BaseExecutor, ...)` 不需改
- `tests/test_deepseek_provider.py:14` 的 `from agents.mas.executors import _resolve_llm` 不需改
- 删除原 `agents/mas/executors.py`

---

## 4. docstring / 小清理

| 文件 | 操作 |
|------|------|
| `agents/__init__.py` | 更新过时 docstring（"starting point for building a pluggable inference stack"）为反映现状：MAS 主从编排 + skill agents + provider 抽象 |
| `memory/__init__.py` | docstring 同步更新（第 1 项一并处理） |

---

## 验证

1. **后端测试**：`python -m pytest tests/ -x -q`（预期 305 项，删 test_workflow.py 后约 297 项全过 + 1 项 langfuse 环境问题跳过）
2. **重启后端**：带 `$env:FULIN_MODELS__DEEPSEEK__ENABLED='true'` 重启 8080
3. **浏览器复测**：http://localhost:5173 本地 Qwen + DeepSeek 全链路（简历优化 + 闲聊）
4. **memory 持久化**：重启后端后会话不丢失
