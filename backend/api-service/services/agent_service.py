from typing import Optional, Dict, Any

from agents.registry import get_agent
from logger import get_logger
from .exceptions import ServiceError
from skill_creator.registry import get_skill_executor
from skill_creator import init_skills
from rag.document_processor import parse_resume

logger = get_logger(__name__)


class AgentService:
    """业务层：对外提供 Agent 生成接口和技能调用接口。"""

    def __init__(self):
        self.logger = logger
        self.skill_executor = get_skill_executor()
        # 初始化技能注册表
        init_skills()

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
        self.logger.info(f"[generate_with_file] 开始处理文件: {file_path}")
        self.logger.info(f"[generate_with_file] 用户输入: {prompt[:100]}...")

        try:
            # 解析简历文件，提取技能栏和项目描述
            self.logger.info("[generate_with_file] 开始解析简历...")
            resume_data = parse_resume(file_path)
            skills = resume_data.get("skills", "")
            projects = resume_data.get("projects", "")
            
            self.logger.info(f"[generate_with_file] 简历解析完成：技能栏{len(skills) if skills else 0}字，项目描述{len(projects) if projects else 0}字")
            
            if skills:
                self.logger.debug(f"[generate_with_file] 技能栏内容: {skills[:200]}...")
            if projects:
                self.logger.debug(f"[generate_with_file] 项目描述内容: {projects[:200]}...")
            
            # 构建包含简历解析结果的提示
            full_prompt = f"""请优化这份简历：

【技能栏】
{skills or '无'}

【项目描述】
{projects or '无'}

【用户补充说明】
{prompt}

请重点优化技能栏和项目描述，使其更专业、更具吸引力。"""
            
            self.logger.info(f"[generate_with_file] 开始调用模型生成，prompt长度: {len(full_prompt)}")
            self.logger.debug(f"[generate_with_file] 完整prompt: {full_prompt[:500]}...")
            
            agent = get_agent(provider=provider, model=model)
            result = agent.generate(full_prompt)
            
            self.logger.info(f"[generate_with_file] 模型生成完成，结果长度: {len(result) if isinstance(result, str) else 0}")
            self.logger.debug(f"[generate_with_file] 生成结果: {result[:200]}...")
            return result
        except Exception as err:
            self.logger.error("AgentService.generate_with_file failed: %s", err, exc_info=True)
            raise ServiceError("Agent generation with file failed", provider=provider, model=model) from err
    
    def execute_skill(self, skill_name: str, **kwargs) -> Dict[str, Any]:
        """
        执行指定技能
        
        Args:
            skill_name: 技能名称
            **kwargs: 技能参数
        
        Returns:
            技能执行结果
        """
        self.logger.debug("AgentService.execute_skill called skill=%s", skill_name)
        
        try:
            result = self.skill_executor.execute(skill_name, **kwargs)
            self.logger.info(f"Skill {skill_name} executed successfully")
            return result
        except Exception as err:
            self.logger.error(f"Skill execution failed: {err}", exc_info=True)
            raise ServiceError(f"Skill execution failed: {skill_name}", skill=skill_name) from err
    
    def execute_skill_auto(self, user_input: str, context: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        自动识别并执行技能
        
        Args:
            user_input: 用户输入
            context: 上下文信息
            **kwargs: 技能参数
        
        Returns:
            技能执行结果
        """
        self.logger.debug("AgentService.execute_skill_auto called input=%s", user_input[:50] if len(user_input) > 50 else user_input)
        
        try:
            result = self.skill_executor.execute_auto(user_input, context, **kwargs)
            self.logger.info(f"Auto skill execution completed")
            return result
        except Exception as err:
            self.logger.error(f"Auto skill execution failed: {err}", exc_info=True)
            raise ServiceError("Auto skill execution failed", input=user_input) from err
    
    def list_skills(self) -> list:
        """列出所有可用技能"""
        return self.skill_executor.list_available_skills()
