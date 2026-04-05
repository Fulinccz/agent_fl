from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_provider: str = "local"
    model_name: str = "Qwen3___5-4B"

    model_config = {"extra": "ignore"}

    @classmethod
    def load(cls):
        return cls()
