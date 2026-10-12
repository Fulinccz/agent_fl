# Provider 链路拆分方案

## Context

当前"线上 API 链路（DeepSeek）"与"本地模型链路（GGUF）"在代码上耦合在一起，改任一条链路都要动共享入口，互相影响、难维护：

1. `agents/registry.py` 的 `get_provider` 用 if-elif 分支处理三类 provider，且**顶部硬 import** 在线 Provider 类——本地链路启动时被迫加载在线 SDK。
2. `agents/providers/` 目录混放：本地 `local/` 与在线 `deepseek.py`、`online.py` 平级，无物理隔离。
3. `BaseProvider` 缺 `generate_stream`（实际被 executors.py / polish_agent.py 调用），契约不一致。
4. `OnlineProvider` + `OpenAIAgent` 是死代码（无任何调用方），config 的 openai section 也没被读取。

目标：把两条链路在目录与 registry 入口上物理隔离，改一条不影响另一条；同时清理死代码。用户已确认删除 OnlineProvider。

## 改动清单

### 1. 目录重组（本地 / 在线物理隔离）
- 新建 `agents/providers/cloud/__init__.py`
- 移动 `agents/providers/deepseek.py` → `agents/providers/cloud/deepseek.py`（内容不变，仅调整相对 import：`..core.base_provider` → `...core.base_provider`）
- `agents/providers/local/` 保持不变

### 2. 删除死代码 OnlineProvider
- 删除 `agents/providers/online.py`
- 删除 `agents/providers/__init__.py` 中对 online 的引用（若有）
- `registry.py`：删除顶部 `from .providers.online import OnlineProvider, OpenAIAgent`、`get_provider` 中 `online/openai/cloud` 分支、`__all__` 移除 `OnlineProvider`/`OpenAIAgent`
- `config.yaml`：删除 `models.openai` section
- `services/app_config.py`：删除 `OpenAIModelConfig` 类、`ModelsConfig.openai` 字段

### 3. registry 拆分（本地 / 在线入口分离）
`registry.py` 改动：
- 删除顶部硬 import（`from .providers.deepseek import DeepSeekProvider`），改为延迟 import
- 拆 `get_provider` 为两个内部方法：
  - `_get_local_provider(model, **kwargs)`：本地 GGUF，按 gguf_path 缓存，延迟 import `GGUFProvider`（已有逻辑搬迁）
  - `_get_cloud_provider(normalized, model, **kwargs)`：在线 DeepSeek，延迟 import `DeepSeekProvider`
- `get_provider` 保持兼容入口：`deepseek` → `_get_cloud_provider`；其余 → `_get_local_provider`
- `__all__` 仅保留 `AgentRegistry`、`get_agent`、`DeepSeekProvider`（DeepSeekProvider 仅作类型提示导出，不再顶部 import）
- 辅助方法 `_is_gguf_engine_enabled`/`_get_gguf_path` 保留在 `_get_local_provider` 路径

### 4. BaseProvider 契约统一
- `agents/core/base_provider.py`：新增 `generate_stream` 抽象方法（签名与 GGUF/DeepSeek 现有实现一致：`prompt, images=None, **kwargs` → `Generator[Dict[str,Any], None, None]`）
- GGUFProvider / DeepSeekProvider 已有实现，无需改动

### 5. 更新测试 import 路径
- `tests/test_deepseek_provider.py`：`from agents.providers.deepseek import DeepSeekProvider` → `from agents.providers.cloud.deepseek import DeepSeekProvider`

### 6. llm.py（保持现状）
registry 拆分后，`get_shared_llm` 调用 `get_agent` 时本地与在线已走不同 registry 路径，已隔离。llm.py 仅清理 docstring 中关于 OpenAI 的过时描述，逻辑不动（改动风险高、收益低）。

## 不改动的部分
- `agents/llm.py` 核心逻辑（本地单例 + 在线缓存分支清晰，registry 已隔离）
- 所有调用方（executors.py / planner.py / reflector.py / conversation_graph.py / workflow.py / llm_classifier.py / jd_keyword_match.py / resume_agents / agent_service / evaluation）——它们都走 `get_agent`/`get_shared_llm` 兼容入口，无需改动
- `agents/providers/local/` 目录及内部文件（token_processor.py / utils.py / gguf_provider.py）

## 验证
1. 后端测试：`cd backend/api-service && python -m pytest -q`（应 305 项全过）
2. 重启后端（带 `FULIN_MODELS__DEEPSEEK__ENABLED=true`）
3. 浏览器复测（http://localhost:5173）：
   - 本地 Qwen 模式：普通对话 + 简历优化全链路
   - DeepSeek 模式：对话流式渲染正常
4. 确认本地链路启动不再加载在线 SDK（registry 不再顶部 import DeepSeekProvider）
