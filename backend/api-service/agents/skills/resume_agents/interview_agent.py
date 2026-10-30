"""面试攻略 Agent

产物 = 一份「整体分析攻略」+ 若干个「最可能被问的问题」。

三阶段生成：
1. 维度规划：依据简历（可选 JD）确定 3 个面试考察维度（内部素材，用于保证提问覆盖面）
2. 并发出题 + 整体攻略：四个调用互不依赖，一起提交给推理服务（llama-server
   多槽位批处理，本地 CPU 上总吞吐约 1.4 倍），维度完成即按顺序推送增量
3. 合成：维度题目去重后按「每维首题优先」摊平，截取前 MAX_LIKELY_QUESTIONS 道
   作为最可能被问的问题，并附上该题所属考察维度

本地小模型 JSON 稳定性有限，每阶段都有规则兜底，保证始终产出可渲染的攻略卡片。
跨维度去重依赖"已覆盖维度"提示 + 题库后处理相似度过滤。
"""
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agents.registry import get_agent
from logger import get_logger
from services.json_utils import extract_json_object
from services.tracing import start_span, update_observation
from .state import ResumeState
from .textutil import split_text, strip_llm_artifacts

logger = get_logger(__name__)

MAX_DIMENSIONS = 3
MAX_QUESTIONS_PER_DIM = 3
MAX_ANSWER_POINTS = 3
# 攻略输出的高频问题条数：每维首题必然入选，再按轮次补齐
MAX_LIKELY_QUESTIONS = 6
MAX_STRATEGY_POINTS = 3
RESUME_CHUNK_LIMIT = 1200

# 题干 + 答题要点的字符二元组 Jaccard 相似度阈值：
# 实测同义重复题 ≈0.48，不同题 ≤0.20，取 0.35 留足间隔
DEDUP_SIMILARITY = 0.35
# 维度名 + 考察重点的相似度阈值：LLM 偶尔把同一主题拆成两个维度，
# 会导致多跑一次完整出题（CPU 约 1 分钟）且题目高度重合
DIM_SIMILARITY = 0.5

# LLM 不可用/解析失败时的默认考察维度
_DEFAULT_DIMENSIONS: List[Dict[str, str]] = [
    {"key": "project", "label": "项目经历深挖",
     "focus": "围绕简历中的核心项目追问技术选型、个人贡献与结果"},
    {"key": "skill", "label": "技术基础与深度",
     "focus": "验证简历所列技能的掌握深度与能力边界"},
    {"key": "match", "label": "岗位匹配度",
     "focus": "确认候选人与目标岗位要求的差距与迁移能力"},
]

# 规则兜底题库使用：从简历中提取可引用的技术关键词
_TECH_TERMS = (
    "Java", "Python", "Go", "C++", "React", "Vue", "TypeScript", "Node",
    "Spring", "MySQL", "Redis", "Kafka", "Flink", "Spark", "Docker",
    "Kubernetes", "PyTorch", "TensorFlow", "RAG", "LangChain",
)


@dataclass
class InterviewPrepAgent:
    """面试攻略 Agent - 生成分维度面试题库"""

    name: str = "interview_prep_agent"
    llm: Optional[Any] = field(default=None, repr=False)
    # 增量回调：每完成一个维度就推一份当前题库，前端可边生成边渲染
    on_progress: Optional[Callable[[Dict[str, Any]], None]] = field(default=None, repr=False)
    # 取消探测：客户端断开后执行器会置位，逐维度出题之间据此提前收手，
    # 避免无人消费时仍把剩余维度跑完（每维一次完整解码，CPU 约 1 分钟）
    should_stop: Optional[Callable[[], bool]] = field(default=None, repr=False)
    # 已推送过的维度数：新增维度被整维去重时不必重复推送同一份题库
    _emitted_dims: int = field(default=0, init=False, repr=False)
    # 已完成但可能尚未随增量推送的攻略：后续增量沿用同一份，避免前端闪空
    _strategy_so_far: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    # 页面入口指令（面试页：面试官视角出题）
    _directive: str = field(default="", init=False, repr=False)

    def __post_init__(self):
        if self.llm is None:
            try:
                self.llm = get_agent(provider="local")
            except Exception as e:
                # 模型不可用不应让整条链路失败：generate 会抛错，各阶段各自落规则兜底
                logger.warning(f"[{self.name}] 本地模型加载失败，改用规则题库: {e}")
                self.llm = None

    def run(self, state: ResumeState) -> ResumeState:
        """执行面试攻略生成（始终写回 state 并返回原 dict）"""
        logger.info(f"[{self.name}] 开始生成面试攻略")

        resume = str(state.get("resume") or "")
        jd = str(state.get("jd") or "")
        excerpt = self._resume_excerpt(resume)
        self._directive = str(state.get("directive") or "").strip()

        try:
            position, dimensions, plan_rule = self._plan_dimensions(excerpt, jd)
            labels = [d["label"] for d in dimensions]
            prepared: List[Dict[str, Any]] = []
            strategy: Dict[str, Any] = {}
            used_rule = plan_rule
            strategy_rule = False

            if self._stop_requested():
                # 客户端在规划阶段已断开：不再发起任何出题/攻略调用
                logger.info(f"[{self.name}] 客户端已断开，跳过题库生成")
            else:
                with ThreadPoolExecutor(max_workers=MAX_DIMENSIONS + 1) as pool:
                    # 整体攻略与逐维度出题并行，互不阻塞（攻略不依赖题目）
                    strategy_future = pool.submit(
                        self._build_strategy, position, excerpt, jd, labels
                    )
                    futures = [
                        pool.submit(
                            self._questions_for_dimension, dim, excerpt, jd, labels
                        )
                        for dim in dimensions
                    ]
                    for dim, future in zip(dimensions, futures):
                        questions, rule = future.result()
                        used_rule = used_rule or rule
                        prepared.append({**dim, "questions": questions})
                        # 攻略先回就先带上，前端可先渲染策略再等题目补齐
                        if strategy_future.done() and not strategy:
                            strategy, strategy_rule = strategy_future.result()
                        self._emit_progress(position, prepared, used_rule, strategy)
                    if not strategy:
                        strategy, strategy_rule = strategy_future.result()
                used_rule = used_rule or strategy_rule

            guide = {
                "position": position,
                "source": "rule" if used_rule else "llm",
                "strategy": strategy,
                "questions": self._flatten_likely(prepared),
            }
            state["interview_guide"] = self._normalize_guide(guide, excerpt, jd)
            state["current_step"] = "interview_completed"
            logger.info(
                f"[{self.name}] 面试攻略生成完成: "
                f"{len(state['interview_guide']['questions'])} 道高频题, source={guide['source']}"
            )
        except Exception as e:
            logger.error(f"[{self.name}] 面试攻略生成失败: {e}")
            state["error"] = f"面试攻略生成失败: {str(e)}"
            state["interview_guide"] = self._fallback_guide(excerpt, jd)
            state["current_step"] = "interview_fallback"

        return state

    def _emit_progress(
        self,
        position: str,
        dimensions: List[Dict[str, Any]],
        used_rule: bool,
        strategy: Optional[Dict[str, Any]] = None,
    ) -> None:
        """增量推送：把已完成的维度（去重后）+ 已有攻略交给调用方，边生成边渲染"""
        if self.on_progress is None:
            return
        if strategy:
            self._strategy_so_far = strategy
        dims = self._dedupe_questions(dimensions)
        # 新维度题目被整维去重（与已有题库同义）时，题库没有变化就不重复推送
        if len(dims) <= self._emitted_dims:
            return
        self._emitted_dims = len(dims)
        self.on_progress({
            "position": position,
            "source": "rule" if used_rule else "llm",
            "strategy": self._strategy_so_far,
            "questions": self._flatten_likely(dims),
        })

    def _stop_requested(self) -> bool:
        """是否已被要求提前结束（客户端断开）"""
        return bool(self.should_stop and self.should_stop())

    # ---------- 阶段 A：维度规划 ----------

    def _plan_dimensions(
        self, excerpt: str, jd: str
    ) -> Tuple[str, List[Dict[str, str]], bool]:
        """返回 (岗位名称, 维度列表, 是否使用规则兜底)"""
        prompt = f"""{self._context_block(excerpt, jd, self._directive)}
任务：规划 3 个互不重叠的面试考察维度。

要求：
1. 三个不同角度：简历核心项目的深挖、目标岗位最关键技术的深度、岗位匹配或协作软素质
2. 维度之间不能是同一主题的两种说法
3. 每个维度都要能直接从简历经历里找到追问点
4. 只输出紧凑 JSON（单行、无缩进、无换行），禁止 markdown 围栏和任何解释
{{"position": "岗位名称", "dimensions": [{{"key": "英文短标识", "label": "中文维度名", "focus": "该维度考察重点（20字内）"}}]}}"""

        try:
            raw = self._generate(prompt, "llm-interview-dimensions", 256)
            data = json.loads(extract_json_object(strip_llm_artifacts(raw)) or "")
            dims = self._clean_dimensions(data.get("dimensions"))
            if dims:
                dims = self._fill_dimensions(dims)
                position = str(data.get("position") or "").strip()
                return position or self._guess_position(jd), dims, False
            logger.warning(f"[{self.name}] 维度规划结果为空，使用默认维度")
        except Exception as e:
            logger.warning(f"[{self.name}] 维度规划解析失败，使用默认维度: {e}")

        return self._guess_position(jd), [dict(d) for d in _DEFAULT_DIMENSIONS], True

    @staticmethod
    def _context_block(excerpt: str, jd: str, directive: str = "") -> str:
        """所有 LLM 调用共用的固定前缀（简历 + JD + 入口指令）

        放在提示词最前面，连续调用时 llama-server 可复用这段 KV 前缀，
        省掉每次重新 prefill 简历的成本（本地 CPU 上一次 prefill 约数秒到十几秒）。
        """
        directive_block = f"\n【本次入口要求】{directive}\n" if directive else ""
        return (
            "你是资深技术面试官，正在为下面这位候选人准备面试题库。\n"
            f"{directive_block}\n"
            f"【简历】\n{excerpt}\n\n"
            f"【目标岗位 JD】\n{jd.strip() or '未提供'}\n"
        )

    @classmethod
    def _fill_dimensions(cls, dims: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """剔除同主题维度，并以默认维度补齐到 3 个

        LLM 偶尔把同一主题拆成两个维度（白跑一次完整出题），也可能只给 1-2 个；
        这里统一收敛为 3 个互不重叠的维度。
        """
        kept: List[Dict[str, str]] = []
        for dim in dims:
            if any(cls._dim_similarity(dim, k) >= DIM_SIMILARITY for k in kept):
                logger.info(f"[{cls.__name__}] 维度去重丢弃: {dim['label']}")
                continue
            kept.append(dim)

        for default in _DEFAULT_DIMENSIONS:
            if len(kept) >= MAX_DIMENSIONS:
                break
            if any(cls._dim_similarity(default, k) >= DIM_SIMILARITY for k in kept):
                continue
            kept.append(dict(default))
        return kept[:MAX_DIMENSIONS]

    @classmethod
    def _dim_similarity(cls, a: Dict[str, str], b: Dict[str, str]) -> float:
        """维度相似度：短维度名用重叠系数（只要一个名字涵盖另一个即视为同主题），
        名字不同再看「名字 + 考察重点」的 Jaccard。"""
        left, right = cls._bigrams(a["label"]), cls._bigrams(b["label"])
        label_sim = (
            len(left & right) / min(len(left), len(right)) if left and right else 0.0
        )
        focus_sim = cls._similarity(
            a["label"] + a.get("focus", ""), b["label"] + b.get("focus", "")
        )
        return max(label_sim, focus_sim)

    @staticmethod
    def _clean_dimensions(raw: Any) -> List[Dict[str, str]]:
        """清洗 LLM 返回的维度列表：兼容纯字符串数组"""
        dims: List[Dict[str, str]] = []
        if not isinstance(raw, list):
            return dims
        for item in raw:
            if isinstance(item, str) and item.strip():
                dims.append({"key": f"dim{len(dims) + 1}", "label": item.strip(), "focus": ""})
            elif isinstance(item, dict):
                label = str(item.get("label") or "").strip()
                if not label:
                    continue
                dims.append({
                    "key": str(item.get("key") or "").strip() or f"dim{len(dims) + 1}",
                    "label": label,
                    "focus": str(item.get("focus") or "").strip(),
                })
            if len(dims) >= MAX_DIMENSIONS:
                break
        return dims

    @staticmethod
    def _guess_position(jd: str) -> str:
        """从 JD 首行推断岗位名称；无 JD 时给通用占位"""
        for line in (jd or "").splitlines():
            line = line.strip()
            if line:
                return line[:40]
        return "目标岗位"

    # ---------- 阶段 A2：整体分析攻略 ----------

    def _build_strategy(
        self, position: str, excerpt: str, jd: str, labels: List[str]
    ) -> Tuple[Dict[str, Any], bool]:
        """返回 (整体攻略, 是否使用规则兜底)"""
        prompt = f"""{self._context_block(excerpt, jd, self._directive)}
任务：为这位候选人写一份面试整体攻略（面试官视角的预判，供候选人备战）。
考察维度：{'、'.join(labels)}

要求：
1. summary：候选人整体画像 + 面试主攻方向（80-120字，点明最值得深挖、最容易被问住的地方）
2. highlights：3 条核心亮点，每条不超过 18 字，必须来自简历
3. risks：3 条主要风险（简历短板或与岗位要求的差距），每条不超过 18 字
4. tips：3 条备战建议（怎么补、怎么讲），每条不超过 30 字
5. 只输出紧凑 JSON（单行、无缩进、无换行），禁止 markdown 围栏和任何解释
{{"summary": "整体画像与主攻方向", "highlights": ["亮点"], "risks": ["风险"], "tips": ["建议"]}}"""

        try:
            raw = self._generate(prompt, "llm-interview-strategy", 512)
            data = json.loads(extract_json_object(strip_llm_artifacts(raw)) or "")
            strategy = self._normalize_strategy(data)
            if strategy:
                return strategy, False
            logger.warning(f"[{self.name}] 整体攻略解析为空，使用规则兜底")
        except Exception as e:
            logger.warning(f"[{self.name}] 整体攻略生成失败，使用规则兜底: {e}")

        return self._rule_strategy(position, excerpt, jd), True

    @classmethod
    def _normalize_strategy(cls, data: Any) -> Dict[str, Any]:
        """攻略归一化：四个字段全空视为无效（交给规则兜底）"""
        if not isinstance(data, dict):
            return {}
        strategy = {
            "summary": str(data.get("summary") or "").strip(),
            "highlights": cls._str_list(data.get("highlights")),
            "risks": cls._str_list(data.get("risks")),
            "tips": cls._str_list(data.get("tips")),
        }
        if not any(strategy.values()):
            return {}
        return strategy

    @staticmethod
    def _str_list(raw: Any, limit: int = MAX_STRATEGY_POINTS) -> List[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        return [str(item).strip() for item in raw if str(item).strip()][:limit]

    def _rule_strategy(self, position: str, excerpt: str, jd: str) -> Dict[str, Any]:
        """规则攻略：用简历技术词与岗位信息拼出可用的备战建议"""
        terms = self._resume_terms(excerpt)
        tech = "、".join(terms) if terms else "简历中的核心技术栈"
        has_jd = bool((jd or "").strip())
        target = "JD" if has_jd else "岗位要求"
        return {
            "summary": (
                f"目标岗位是{position}，简历主线围绕{tech}展开。"
                "面试大概率从项目深挖切入，再验证技术深度"
                f"{'与岗位要求的匹配度' if has_jd else '与基础扎实程度'}；"
                "每个案例按「背景-动作-结果」组织，结果尽量给数字。"
            ),
            "highlights": [
                f"有 {tech} 的实战经历",
                "有完整项目落地经验",
                "技术方向与岗位一致",
            ],
            "risks": [
                "项目成果缺少量化数据",
                "技术深度可能被追问到原理",
                "岗位要求的短板未体现",
            ],
            "tips": [
                "每个项目准备 1 个可量化结果",
                "准备 2 个难题的排查过程",
                f"对照{target}逐条准备匹配案例",
            ],
        }

    # ---------- 阶段 B：逐维度出题 ----------

    def _questions_for_dimension(
        self,
        dim: Dict[str, str],
        excerpt: str,
        jd: str,
        labels: List[str],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        """返回 (题目列表, 是否使用规则兜底)"""
        prompt = f"""{self._context_block(excerpt, jd, self._directive)}
任务：为「{dim['label']}」维度生成 3 道面试题（素材确实不足时给 2 道）。

该维度考察重点：{dim.get('focus') or '结合简历与岗位要求'}
已覆盖的其他维度：{'、'.join(label for label in labels if label != dim['label']) or '无'}

要求：
1. 每题必须扣住简历里的具体项目/数据/技术，不要出通用八股
2. 同一维度内题目考察方向必须不同
3. answer_points 写答题结构（固定 3 条，每条不超过 18 字），不要写完整答案
4. follow_up 是追问，换一个角度，不要复述题干
5. follow_up 必须与题干前提自洽，不能出现与前提矛盾的假设
6. 只输出紧凑 JSON（单行、无缩进、无换行），禁止 markdown 围栏和任何解释
{{"questions": [{{"question": "面试官会问的问题", "assess_point": "这道题的考察点", "answer_points": ["答题要点1", "答题要点2", "答题要点3"], "follow_up": "可能的追问"}}]}}"""

        for attempt in range(2):
            if attempt and self._stop_requested():
                break
            try:
                raw = self._generate(prompt, f"llm-interview-{dim['key']}", 640)
                data = json.loads(extract_json_object(strip_llm_artifacts(raw)) or "")
                questions = self._normalize_questions(data.get("questions"))
                if questions:
                    return questions, False
                logger.warning(f"[{self.name}] 维度 {dim['key']} 题目为空（第 {attempt + 1} 次）")
            except Exception as e:
                logger.warning(
                    f"[{self.name}] 维度 {dim['key']} 出题失败（第 {attempt + 1} 次）: {e}"
                )

        return self._rule_questions(dim, excerpt), True

    def _generate(self, prompt: str, span_name: str, max_new_tokens: int) -> str:
        """单次 LLM 调用（统一埋点）"""
        if self.llm is None:
            raise RuntimeError("LLM 不可用")
        with start_span(span_name, as_type="generation", input=prompt[:500]) as obs:
            raw = self.llm.generate(prompt, deepThinking=False, max_new_tokens=max_new_tokens)
            update_observation(obs, output=(raw or "")[:500])
        return raw or ""

    # ---------- 归一化与兜底 ----------

    def _normalize_guide(
        self, guide: Dict[str, Any], excerpt: str = "", jd: str = ""
    ) -> Dict[str, Any]:
        """按契约裁剪产物；题目为空时回退规则兜底（用真实简历/JD 兜底，避免退化成通用题）"""
        questions = self._normalize_likely(guide.get("questions"))
        if not questions:
            return self._fallback_guide(excerpt, jd)

        position = str(guide.get("position") or "").strip() or self._guess_position(jd)
        strategy = self._normalize_strategy(guide.get("strategy")) \
            or self._rule_strategy(position, excerpt, jd)
        return {
            "position": position,
            "source": guide.get("source", "llm"),
            "strategy": strategy,
            "questions": questions,
        }

    # ---------- 高频问题合成 ----------

    @classmethod
    def _flatten_likely(
        cls, dims: List[Dict[str, Any]], limit: int = MAX_LIKELY_QUESTIONS
    ) -> List[Dict[str, Any]]:
        """维度题库 → 高频问题列表

        每维首题优先（面试官通常先用该维度的开场题切入），再按轮次补齐；
        每道题带上所属考察维度，前端可直接展示分类。
        不同维度的题也可能撞车（如两个维度都追问同一项目），摊平时同步去重，
        跳过后继续下一轮补齐，避免最终列表出现重复题。
        """
        dims = cls._dedupe_questions(dims)
        questions: List[Dict[str, Any]] = []
        fingerprints: List[str] = []
        for round_index in range(MAX_QUESTIONS_PER_DIM):
            for dim in dims:
                items = dim.get("questions") or []
                if round_index >= len(items):
                    continue
                item = items[round_index]
                fp = cls._fingerprint(item)
                if any(cls._similarity(fp, seen) >= DEDUP_SIMILARITY for seen in fingerprints):
                    logger.info(
                        f"[interview_prep_agent] 跨维度题目去重丢弃: {str(item.get('question'))[:30]}"
                    )
                    continue
                fingerprints.append(fp)
                questions.append({
                    **item,
                    "category": str(dim.get("label") or "").strip(),
                })
                if len(questions) >= limit:
                    return questions
        return questions

    @classmethod
    def _normalize_likely(cls, raw: Any) -> List[Dict[str, Any]]:
        """高频问题列表归一化：兼容字符串与对象，跨题去重后截断"""
        if not isinstance(raw, list):
            return []
        questions: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, str):
                item = {"question": item}
            if not isinstance(item, dict):
                continue
            normalized = cls._normalize_questions([item])
            if not normalized:
                continue
            questions.append({
                **normalized[0],
                "category": str(item.get("category") or "").strip(),
            })
            if len(questions) >= MAX_LIKELY_QUESTIONS:
                break
        return cls._dedupe_flat(questions)

    @classmethod
    def _dedupe_flat(cls, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """扁平题目列表去重（同义题只保留一次）"""
        kept: List[Dict[str, Any]] = []
        fingerprints: List[str] = []
        for question in questions:
            fp = cls._fingerprint(question)
            if any(cls._similarity(fp, seen) >= DEDUP_SIMILARITY for seen in fingerprints):
                logger.info(
                    f"[interview_prep_agent] 去重丢弃: {str(question.get('question'))[:30]}"
                )
                continue
            fingerprints.append(fp)
            kept.append(question)
        return kept

    # ---------- 跨维度去重 ----------

    @staticmethod
    def _fingerprint(question: Dict[str, Any]) -> str:
        """题干 + 答题要点拼接：同义题即使措辞不同，答题结构也高度重合"""
        points = "".join(str(p) for p in question.get("answer_points") or [])
        return str(question.get("question") or "") + points

    @staticmethod
    def _bigrams(text: str) -> set:
        compact = "".join(ch for ch in text if ch.isalnum())
        if len(compact) < 2:
            return {compact}
        return {compact[i:i + 2] for i in range(len(compact) - 1)}

    @classmethod
    def _similarity(cls, a: str, b: str) -> float:
        left, right = cls._bigrams(a), cls._bigrams(b)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    @classmethod
    def _dedupe_questions(cls, dims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """跨维度剔除同义题；某维度题目被全部剔除则丢弃该维度"""
        kept: List[Dict[str, Any]] = []
        fingerprints: List[str] = []
        for dim in dims:
            questions: List[Dict[str, Any]] = []
            for question in dim.get("questions") or []:
                fp = cls._fingerprint(question)
                if any(
                    cls._similarity(fp, seen) >= DEDUP_SIMILARITY
                    for seen in fingerprints
                ):
                    logger.info(
                        f"[interview_prep_agent] 去重丢弃: {str(question.get('question'))[:30]}"
                    )
                    continue
                fingerprints.append(fp)
                questions.append(question)
            if questions:
                kept.append({**dim, "questions": questions})
        return kept

    @staticmethod
    def _normalize_questions(raw: Any) -> List[Dict[str, Any]]:
        """题目归一化：兼容字符串数组与对象数组，丢弃空题干"""
        questions: List[Dict[str, Any]] = []
        if not isinstance(raw, list):
            return questions
        for item in raw:
            if isinstance(item, str):
                item = {"question": item}
            if not isinstance(item, dict):
                continue
            text = str(item.get("question") or "").strip()
            if not text:
                continue
            points = item.get("answer_points") or []
            if isinstance(points, str):
                points = [points]
            points = [str(p).strip() for p in points if str(p).strip()][:MAX_ANSWER_POINTS]
            questions.append({
                "question": text,
                "assess_point": str(item.get("assess_point") or "").strip(),
                "answer_points": points,
                "follow_up": str(item.get("follow_up") or "").strip(),
            })
            if len(questions) >= MAX_QUESTIONS_PER_DIM:
                break
        return questions

    def _fallback_guide(self, excerpt: str, jd: str) -> Dict[str, Any]:
        """全量规则兜底：默认维度题库 + 规则攻略"""
        position = self._guess_position(jd)
        dims = [
            {**dim, "questions": self._rule_questions(dim, excerpt)}
            for dim in _DEFAULT_DIMENSIONS
        ]
        return {
            "position": position,
            "source": "rule",
            "strategy": self._rule_strategy(position, excerpt, jd),
            "questions": self._flatten_likely(dims),
        }

    def _rule_questions(self, dim: Dict[str, str], excerpt: str) -> List[Dict[str, Any]]:
        """规则题库：按维度 key 给静态模板，并结合简历中的技术关键词"""
        key = (dim.get("key") or "").lower()
        terms = self._resume_terms(excerpt)
        term = terms[0] if terms else ""
        tech = "、".join(terms) if terms else "简历中的技术栈"

        if key in ("project", "项目"):
            questions = [
                {
                    "question": "简历里最有代表性的项目是哪一个？能说说你的具体职责和产出吗？",
                    "assess_point": "判断是否真实主导过项目，能否区分个人贡献与团队成果",
                    "answer_points": ["项目背景与目标", "你的角色与关键动作", "结果数据（性能/用户量/收益）"],
                    "follow_up": "这个项目里最棘手的问题是什么？你是怎么解决的？",
                },
                {
                    "question": "项目中的技术方案为什么这样选？有没有考虑过其他方案？",
                    "assess_point": "考察技术选型的思考过程与权衡能力",
                    "answer_points": ["当时约束条件", "候选方案对比", "最终结论与验证方式"],
                    "follow_up": "如果重做一次，你会怎么改？",
                },
            ]
        elif key in ("skill", "技术"):
            questions = [
                {
                    "question": f"你简历提到 {tech}，说说你用它在最复杂场景下的实践？",
                    "assess_point": "验证技能掌握深度，区分了解与实战",
                    "answer_points": ["具体场景与规模", "你负责的部分", "踩过的坑与解决方式"],
                    "follow_up": f"如果让你从零设计一个基于 {term or '该技术'} 的方案，你会怎么做？",
                },
                {
                    "question": "线上出现性能问题，你会怎么定位和优化？",
                    "assess_point": "考察排查思路的系统性与工具熟练度",
                    "answer_points": ["先看监控与指标定位瓶颈", "分层排查（应用/存储/网络）", "优化后回归验证"],
                    "follow_up": "如何证明优化真的有效？",
                },
            ]
        elif key in ("match", "匹配"):
            questions = [
                {
                    "question": "这个岗位最看重的哪项能力，你有对应的经历可以证明吗？",
                    "assess_point": "考察候选人对岗位要求的理解与自我匹配度",
                    "answer_points": ["拆解岗位核心要求", "对应经历与成果", "差距与补齐计划"],
                    "follow_up": "如果某项要求你暂时不具备，你打算怎么补？",
                },
                {
                    "question": "如果你入职，前三个月你会怎么开展工作？",
                    "assess_point": "考察业务理解与落地节奏",
                    "answer_points": ["先熟悉业务与团队", "找到可快速交付的小目标", "建立反馈与复盘机制"],
                    "follow_up": "你怎么判断自己前三个月是否成功？",
                },
            ]
        elif key in ("career", "规划"):
            questions = [
                {
                    "question": "为什么选择我们公司 / 这个岗位？",
                    "assess_point": "考察求职动机与稳定性",
                    "answer_points": ["对业务与产品的理解", "与自身方向的契合", "可贡献的具体价值"],
                    "follow_up": "如果同时拿到另一个 offer，你会怎么选？",
                },
                {
                    "question": "未来三年的职业规划是什么？",
                    "assess_point": "考察成长预期与目标感",
                    "answer_points": ["短期目标（能力补齐）", "中期目标（职责升级）", "与岗位发展路径的衔接"],
                    "follow_up": "为此你现在已经在做哪些准备？",
                },
            ]
        else:
            questions = [
                {
                    "question": f"围绕「{dim.get('label') or '岗位要求'}」，你有哪些可以举例的经历？",
                    "assess_point": "考察相关经验的真实性与可迁移性",
                    "answer_points": ["具体场景", "你的关键动作", "可量化的结果"],
                    "follow_up": "这件事如果重来，你会改哪一步？",
                },
                {
                    "question": "遇到没有做过的任务，你通常怎么推进？",
                    "assess_point": "考察学习能力与解决问题的路径",
                    "answer_points": ["拆解问题", "快速补齐信息与方法", "小步验证并复盘"],
                    "follow_up": "举一个你最近这么做的例子。",
                },
            ]
        return questions

    # ---------- 简历摘要 ----------

    @staticmethod
    def _resume_excerpt(resume: str) -> str:
        """简历摘要：按段落切块后取首块 + 末块，避免尾部内容被静默截断"""
        chunks = split_text(resume, RESUME_CHUNK_LIMIT)
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]
        return f"【简历开头】\n{chunks[0]}\n【简历结尾】\n{chunks[-1]}"

    @staticmethod
    def _resume_terms(text: str) -> List[str]:
        lowered = (text or "").lower()
        return [term for term in _TECH_TERMS if term.lower() in lowered][:3]
