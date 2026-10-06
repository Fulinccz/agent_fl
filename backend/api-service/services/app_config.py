"""应用业务配置加载器（config.yaml + 环境变量覆盖）

配置分层（业界惯例，优先级从低到高）：
    1. 代码默认值（本文件各 Section 的字段默认值）
    2. config.yaml（随代码版本管理的非敏感行为参数）
    3. 环境变量 / 仓库根目录 .env（敏感凭证 + 环境差异）
       规则：前缀 FULIN_，双下划线分层，全大写
       例：FULIN_MODELS__DEFAULT_PROVIDER=deepseek
           FULIN_RAG__TOP_K=5

基础设施连接（MySQL/Redis/Kafka/Neo4j/JWT/服务端口）由 services.config.AppSettings
从 .env 加载，本模块只负责业务行为参数。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from logger import get_logger

logger = get_logger(__name__)

# 仓库根目录（backend/api-service/services/app_config.py → 上三级）
ROOT_DIR = Path(__file__).resolve().parents[3]
# 默认 yaml 路径：backend/api-service/config.yaml
DEFAULT_CONFIG_FILE = Path(__file__).resolve().parents[1] / "config.yaml"
# 根目录 .env（同时服务于 AppSettings 与本模块的 env 覆盖）
ENV_FILE = ROOT_DIR / ".env"

# 将 .env 注入进程环境（override=False：真实环境变量优先），
# 使 os.getenv("DEEPSEEK_API_KEY") 等直接读取生效
load_dotenv(dotenv_path=ENV_FILE, override=False)


class LocalModelConfig(BaseModel):
    """本地模型配置"""
    name: str = "Qwen3___5-4B"
    max_new_tokens: int = 128
    temperature: float = 0.35
    chat_max_new_tokens: int = 512


class DeepSeekConfig(BaseModel):
    """DeepSeek OpenAPI 配置（OpenAI 兼容协议）"""
    # 成本开关：false 时 MAS 链路对 deepseek 的请求自动回落本地模型
    enabled: bool = False
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    timeout: float = 120.0
    max_retries: int = 2


class OpenAIModelConfig(BaseModel):
    """OpenAI 兼容在线模型配置"""
    model: str = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    timeout: float = 120.0
    max_retries: int = 2


class ModelsConfig(BaseModel):
    """模型与 Provider 配置"""
    default_provider: str = "local"
    local: LocalModelConfig = Field(default_factory=LocalModelConfig)
    deepseek: DeepSeekConfig = Field(default_factory=DeepSeekConfig)
    openai: OpenAIModelConfig = Field(default_factory=OpenAIModelConfig)


class ReflectionConfig(BaseModel):
    """ReAct 反思器配置"""
    use_llm: bool = True
    max_retries: int = 1


class MasConfig(BaseModel):
    """多智能体编排配置"""
    max_total_rounds: int = 12
    max_replans: int = 1
    stream_chunk_chars: int = 50
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)


class RagConfig(BaseModel):
    """RAG 检索配置"""
    top_k: int = 3
    hybrid_top_k: int = 5
    vector_weight: float = 0.5
    graph_weight: float = 0.5
    default_mode: str = "hybrid"


class MemoryConfig(BaseModel):
    """会话记忆配置"""
    max_context_length: int = 4000


class AppConfig(BaseSettings):
    """业务配置根模型

    配置优先级（高 → 低）：
        进程环境变量/.env（FULIN_ 前缀） > config.yaml > 代码默认值
    环境变量规则：前缀 FULIN_，双下划线分层，全大写，如
        FULIN_MODELS__DEFAULT_PROVIDER=deepseek
        FULIN_RAG__TOP_K=5
    """
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    mas: MasConfig = Field(default_factory=MasConfig)
    rag: RagConfig = Field(default_factory=RagConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    model_config = SettingsConfigDict(extra="ignore")

    @classmethod
    def load(cls, config_file: Optional[Path] = None) -> "AppConfig":
        """加载配置：yaml 提供默认值，FULIN_ 前缀环境变量（含 .env）深合并覆盖"""
        path = Path(config_file) if config_file else DEFAULT_CONFIG_FILE
        data = _read_yaml(path)
        merged = _deep_merge(data, _collect_env_overrides())
        return cls(**merged)


def _read_yaml(path: Path) -> dict:
    """读取 yaml；缺失/损坏时返回空 dict（降级到代码默认值）"""
    if not path.exists():
        logger.warning(f"配置文件 {path} 不存在，使用代码默认值")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"配置文件 {path} 解析失败，使用代码默认值: {e}")
        return {}


def _collect_env_overrides() -> dict:
    """收集 FULIN_ 前缀环境变量（进程 env 优先，其次 .env 已由 load_dotenv 注入），
    按 __ 拆分为嵌套 dict。值保持字符串，由 pydantic 做类型转换。"""
    overrides: dict = {}
    prefix = "FULIN_"
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        parts = [p.lower() for p in key[len(prefix):].split("__") if p]
        if not parts:
            continue
        node = overrides
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return overrides


def _deep_merge(base: dict, override: dict) -> dict:
    """override 深合并进 base（override 优先），返回新 dict"""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# ------------------------- 全局单例 -------------------------
_app_config: Optional[AppConfig] = None
_lock = threading.Lock()


def load_app_config(config_file: Optional[Path] = None) -> AppConfig:
    """获取全局业务配置单例（测试可通过 reset_app_config 重置）"""
    global _app_config
    if _app_config is None:
        with _lock:
            if _app_config is None:
                _app_config = AppConfig.load(config_file)
                logger.info(f"[AppConfig] 业务配置加载完成: {DEFAULT_CONFIG_FILE}")
    return _app_config


def reset_app_config():
    """重置单例（仅供测试使用）"""
    global _app_config
    with _lock:
        _app_config = None
