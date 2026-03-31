from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    env: str = "dev"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    model_provider: str = "local"
    model_name: str = "Qwen3___5-4B"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def load(cls):
        return cls()
