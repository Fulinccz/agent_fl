"""services/app_config.py 单元测试

覆盖：yaml 加载、环境变量覆盖、yaml 缺失/损坏降级、单例行为。
"""
import textwrap

import pytest

from services import app_config as app_config_module
from services.app_config import AppConfig, load_app_config, reset_app_config


@pytest.fixture(autouse=True)
def _reset_singleton():
    """每个用例前后重置全局单例，避免污染"""
    reset_app_config()
    yield
    reset_app_config()


class TestAppConfigLoad:
    def test_load_default_yaml(self):
        """加载仓库自带 config.yaml，字段与默认值节一致"""
        cfg = load_app_config()
        assert cfg.models.default_provider in {"local", "deepseek", "openai"}
        assert cfg.models.local.name
        assert cfg.models.deepseek.base_url == "https://api.deepseek.com"
        assert cfg.models.deepseek.model
        assert cfg.mas.max_total_rounds >= 1
        assert cfg.mas.max_replans >= 0
        assert cfg.rag.top_k >= 1
        assert cfg.memory.max_context_length > 0

    def test_load_missing_yaml_uses_defaults(self, tmp_path):
        """yaml 不存在 → 全部使用代码默认值，不抛异常"""
        cfg = AppConfig.load(config_file=tmp_path / "no_such.yaml")
        assert cfg.models.default_provider == "local"
        assert cfg.models.deepseek.base_url == "https://api.deepseek.com"
        assert cfg.mas.max_total_rounds == 12
        assert cfg.mas.stream_chunk_chars == 50

    def test_load_broken_yaml_falls_back_to_defaults(self, tmp_path):
        """yaml 损坏 → 降级到代码默认值"""
        bad = tmp_path / "broken.yaml"
        bad.write_text("models: [unclosed", encoding="utf-8")
        cfg = AppConfig.load(config_file=bad)
        assert cfg.models.default_provider == "local"
        assert cfg.rag.top_k == 3

    def test_custom_yaml_overrides(self, tmp_path):
        """自定义 yaml 覆盖默认值"""
        custom = tmp_path / "custom.yaml"
        custom.write_text(textwrap.dedent("""
            models:
              default_provider: deepseek
              deepseek:
                model: deepseek-v4-pro
                timeout: 30
            mas:
              max_total_rounds: 5
              reflection:
                use_llm: false
            rag:
              top_k: 7
        """), encoding="utf-8")
        cfg = AppConfig.load(config_file=custom)
        assert cfg.models.default_provider == "deepseek"
        assert cfg.models.deepseek.model == "deepseek-v4-pro"
        assert cfg.models.deepseek.timeout == 30
        assert cfg.mas.max_total_rounds == 5
        assert cfg.mas.reflection.use_llm is False
        assert cfg.rag.top_k == 7
        # 未覆盖字段保留默认
        assert cfg.mas.stream_chunk_chars == 50

    def test_env_overrides_yaml(self, tmp_path, monkeypatch):
        """FULIN_ 前缀 + 双下划线分层的环境变量覆盖 yaml"""
        monkeypatch.setenv("FULIN_MODELS__DEFAULT_PROVIDER", "deepseek")
        monkeypatch.setenv("FULIN_RAG__TOP_K", "9")
        cfg = AppConfig.load(config_file=tmp_path / "no_such.yaml")
        assert cfg.models.default_provider == "deepseek"
        assert cfg.rag.top_k == 9

    def test_env_overrides_explicit_yaml_value(self, monkeypatch):
        """环境变量必须能覆盖 yaml 中显式写的值（优先级 env > yaml）"""
        # 仓库自带 config.yaml 中 deepseek.enabled: false，用 env 打开
        monkeypatch.setenv("FULIN_MODELS__DEEPSEEK__ENABLED", "true")
        monkeypatch.setenv("FULIN_MAS__MAX_TOTAL_ROUNDS", "3")
        cfg = AppConfig.load()  # 读真实 config.yaml
        assert cfg.models.deepseek.enabled is True
        assert cfg.mas.max_total_rounds == 3
        # 未覆盖字段仍来自 yaml
        assert cfg.models.deepseek.base_url == "https://api.deepseek.com"


class TestAppConfigSingleton:
    def test_singleton_returns_same_instance(self):
        a = load_app_config()
        b = load_app_config()
        assert a is b

    def test_reset_creates_new_instance(self, tmp_path):
        a = load_app_config()
        reset_app_config()
        b = load_app_config(config_file=tmp_path / "no_such.yaml")
        assert a is not b
