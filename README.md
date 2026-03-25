# fulin_ai_agent
A private small ai agent

## 本地运行加速库配置（推荐）

为了避免大型模型在 CPU 上推理过慢，建议安装以下加速库：

```bash
cd backend/api-service
python -m pip install -r requirements.txt
python -m pip install git+https://github.com/Dao-AILab/causal-conv1d
```

以上包含：
- `flash-attn`: 高性能 Attention 内核
- `causal-conv1d`: 快速因果卷积实现

启动服务后，日志中不应出现 `fast path is not available`，否则说明加速路径仍不可用。

## 开发调试小贴士

- 首次运行 `Qwen3___5-4B` 模型体积大，加载时间会较长。
- 为了快速响应可在 `backend/api-service/agents/local.py` 中先设置 `max_new_tokens=32`。
- 请确保本机有可用 GPU + CUDA，或使用轻量模型（如 gpt2）做调试。
