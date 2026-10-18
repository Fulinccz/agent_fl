from typing import Dict, Any, Optional, List
from logger import get_logger
from agents.llm import get_shared_llm
from services.tracing import start_span, update_current_generation

logger = get_logger(__name__)


INTENT_CLASSIFICATION_PROMPT = """你是一个意图分类助手。请根据用户输入，判断用户想要使用哪个功能。

可选功能：
1. resume-polishing - 简历润色优化（用户想改进简历措辞、让简历更专业、重写某段描述）
2. jd-keyword-match - JD岗位匹配分析（用户想知道简历和某个岗位是否匹配、分析JD要求）
3. resume-score - 简历评分评估（用户想知道自己简历质量如何、能打多少分）
4. resume-parse - 简历解析提取（用户想从简历中提取结构化信息、识别技能）
5. chat-general - 通用对话（用户打招呼、询问功能、寻求求职建议）

向量检索候选结果（按相似度排序）：
{candidates}

用户输入：{user_input}

请分析：
1. 用户的核心意图是什么？
2. 最匹配的功能是哪个？
3. 置信度（0-1）是多少？
4. 是否需要额外参数（如JD文本、简历内容等）？

请严格按以下JSON格式输出，不要有任何其他内容：
{{
  "intent": "功能名称",
  "confidence": 0.95,
  "reason": "简要分析原因",
  "params": {{}}
}}
"""


class LLMIntentClassifier:
    """L3: LLM 意图分类层

    当关键词和向量都无法高置信决策时，调用本地LLM做最终判断。
    """

    def __init__(self, provider: str = "local", model: Optional[str] = None, llm: Optional[Any] = None):
        self.provider = provider
        self.model = model
        self._llm = llm
        self._agent = None
        self._agents: Dict[str, Any] = {}

    @property
    def agent(self):
        if self._agent is None:
            if self._llm is not None:
                self._agent = self._llm
            else:
                self._agent = get_shared_llm(provider=self.provider, model=self.model)
        return self._agent

    def _agent_for(self, provider: Optional[str]) -> Any:
        """按请求上下文的 provider 解析分类用 LLM

        用户显式选择在线模型（如 deepseek）时，意图分类跟随该模型：分类 prompt
        仅约 300 token，在线模型数秒完成，而本地 4B 模型需 60-90 秒，是整条请求
        的最大延迟项。经 get_shared_llm 解析以复用成本开关（deepseek 未启用时自动
        回落本地）与共享单例缓存。注入 LLM（测试/定制）时始终优先。
        """
        if self._llm is not None:
            return self.agent  # 注入的 LLM 优先（测试/定制场景）；走 property 懒加载
        normalized = (provider or "").strip().lower() or self.provider
        if normalized not in self._agents:
            try:
                self._agents[normalized] = get_shared_llm(provider=normalized, model=self.model)
            except Exception as e:
                logger.warning(f"provider={normalized} 解析分类 LLM 失败，回退默认：{e}")
                self._agents[normalized] = self.agent
        return self._agents[normalized]

    def classify(
        self,
        user_input: str,
        candidates: List[Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """使用 LLM 进行意图分类"""

        candidates_str = "\n".join([
            f"- {c.skill} (相似度: {c.score:.3f}): {c.text}"
            for c in candidates[:3]
        ]) if candidates else "无候选结果"

        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            user_input=user_input,
            candidates=candidates_str
        )

        try:
            # 用户选择了在线模型时分类跟随该模型（context.provider），否则用默认
            agent = self._agent_for((context or {}).get("provider"))
            logger.info(f"LLM intent classification for: {user_input[:50]}")
            with start_span("llm-intent-classify", as_type="generation", input=prompt[:500]):
                # max_new_tokens 必须显式给足：GGUF/transformers 默认 128，
                # 长输入（简历全文）时 reason 字段变长，JSON 被截断无闭合括号 → 解析失败
                result_text = agent.generate(prompt, max_new_tokens=256, deepThinking=False)
                update_current_generation(output=result_text[:300])

            import json
            from services.json_utils import extract_json_object

            json_str = extract_json_object(result_text)
            if json_str:
                result = json.loads(json_str)
            else:
                result = {"intent": "chat-general", "confidence": 0.5, "reason": "解析失败", "params": {}}

            logger.info(f"LLM classified intent: {result.get('intent')} (confidence: {result.get('confidence')})")
            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            return {
                "intent": "chat-general",
                "confidence": 0.3,
                "reason": f"分类失败: {str(e)}",
                "params": {}
            }
