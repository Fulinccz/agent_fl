from typing import Optional

from agents.registry import get_agent
from logger import get_logger
from .exceptions import ServiceError

logger = get_logger(__name__)


class AgentService:
    """业务层：对外提供 Agent 生成接口。"""

    def __init__(self):
        self.logger = logger

    def generate(self, prompt: str, provider: str = "local", model: Optional[str] = None) -> str:
        self.logger.debug("AgentService.generate called prompt=%s provider=%s model=%s", prompt, provider, model)

        try:
            agent = get_agent(provider=provider, model=model)
            result = agent.generate(prompt)
            self.logger.debug("Generated text length=%d", len(result) if isinstance(result, str) else 0)
            return result
        except Exception as err:
            self.logger.error("AgentService.generate failed: %s", err, exc_info=True)
            raise ServiceError("Agent generation failed", provider=provider, model=model) from err

    def generate_with_file(self, prompt: str, file_path: str, provider: str = "local", model: Optional[str] = None) -> str:
        self.logger.debug("AgentService.generate_with_file called prompt=%s file_path=%s provider=%s model=%s", prompt, file_path, provider, model)

        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
            
            # 构建包含文件内容的提示
            full_prompt = f"文件内容:\n{file_content}\n\n用户问题:\n{prompt}"
            
            agent = get_agent(provider=provider, model=model)
            result = agent.generate(full_prompt)
            self.logger.debug("Generated text length=%d", len(result) if isinstance(result, str) else 0)
            return result
        except Exception as err:
            self.logger.error("AgentService.generate_with_file failed: %s", err, exc_info=True)
            raise ServiceError("Agent generation with file failed", provider=provider, model=model) from err
