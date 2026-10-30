"""面试攻略 Agent / 执行器测试（mock LLM，不加载真实模型）"""
import json
from unittest.mock import MagicMock

from agents.skills.resume_agents.interview_agent import InterviewPrepAgent
from agents.mas.executors.interview import InterviewPrepExecutor

SAMPLE_RESUME = (
    "张三，后端工程师。\n"
    "项目：订单系统，使用 Java + MySQL + Redis，支撑日均 10 万单。"
)
SAMPLE_JD = "招聘后端工程师，要求熟悉 Java、MySQL、Redis"

DIM_JSON = json.dumps({
    "position": "后端工程师",
    "dimensions": [
        {"key": "project", "label": "项目经历深挖", "focus": "追问个人贡献"},
        {"key": "skill", "label": "技术基础", "focus": "验证技术深度"},
    ],
}, ensure_ascii=False)

QUESTIONS_JSON = json.dumps({
    "questions": [
        {
            "question": "订单系统里你负责哪部分？",
            "assess_point": "是否真正主导过模块",
            "answer_points": ["讲背景", "讲方案", "讲结果"],
            "follow_up": "流量翻倍怎么改？",
        }
    ]
}, ensure_ascii=False)

QUESTIONS_JSON_SKILL = json.dumps({
    "questions": [
        {
            "question": "Redis 缓存穿透在你们系统里具体是怎么防的？",
            "assess_point": "缓存治理的实战深度",
            "answer_points": ["空值缓存", "布隆过滤器", "限流兜底"],
            "follow_up": "布隆过滤器误判怎么办？",
        }
    ]
}, ensure_ascii=False)

STRATEGY_JSON = json.dumps({
    "summary": "候选人有订单系统实战经验，面试主攻方向是并发与缓存治理。",
    "highlights": ["订单系统实战", "有量化结果", "技术栈匹配"],
    "risks": ["缺少跨团队协作案例", "技术原理深度待验证", "高并发经验偏窄"],
    "tips": ["补齐一致性方案细节", "准备压测数据", "梳理一次故障复盘"],
}, ensure_ascii=False)


def _mock_llm(dim_text: str = DIM_JSON, question_text: str = QUESTIONS_JSON):
    """按提示词特征路由：维度规划 / 整体攻略 / 逐维度出题（同维度返回同一批题）"""
    mock = MagicMock()

    def generate(prompt, **kwargs):
        if "dimensions" in prompt:
            return dim_text
        if "面试整体攻略" in prompt:
            return STRATEGY_JSON
        if "为「技术基础」" in prompt:
            return QUESTIONS_JSON_SKILL
        return question_text

    mock.generate.side_effect = generate
    return mock


def test_agent_generates_structured_guide():
    agent = InterviewPrepAgent(llm=_mock_llm())
    state = {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}

    out = agent.run(state)

    assert out is state  # 必须写回传入的 state dict
    guide = out["interview_guide"]
    assert guide["source"] == "llm"
    assert guide["position"] == "后端工程师"
    assert guide["strategy"]["summary"].startswith("候选人有订单系统实战经验")
    assert guide["strategy"]["tips"] == ["补齐一致性方案细节", "准备压测数据", "梳理一次故障复盘"]

    # 每维首题优先摊平，并标注所属考察维度
    assert [q["category"] for q in guide["questions"]] == ["项目经历深挖", "技术基础"]
    question = guide["questions"][0]
    assert question["question"] == "订单系统里你负责哪部分？"
    assert question["assess_point"] == "是否真正主导过模块"
    assert question["answer_points"] == ["讲背景", "讲方案", "讲结果"]
    assert question["follow_up"] == "流量翻倍怎么改？"


def test_agent_falls_back_to_rules_on_invalid_json():
    agent = InterviewPrepAgent(llm=_mock_llm(dim_text="不是 JSON", question_text="仍然不是"))
    state = {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}

    guide = agent.run(state)["interview_guide"]

    assert guide["source"] == "rule"
    assert len(guide["questions"]) == 6  # 3 个默认维度 × 2 道规则题
    assert all(q["question"] for q in guide["questions"])
    assert guide["strategy"]["summary"]


def test_agent_degrades_gracefully_when_llm_unavailable():
    """LLM 抛错属于"降级"而非"失败"：不写 state.error，直接落规则题库"""
    mock = MagicMock()
    mock.generate.side_effect = RuntimeError("模型挂了")
    agent = InterviewPrepAgent(llm=mock)
    state = {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}

    out = agent.run(state)

    assert not out.get("error")
    assert out["interview_guide"]["source"] == "rule"
    assert out["interview_guide"]["questions"]
    assert out["interview_guide"]["strategy"]["tips"]


def test_agent_rejects_empty_resume_via_executor():
    result = InterviewPrepExecutor(llm=_mock_llm()).run({"resume": ""})

    assert result.success is False
    assert "缺少必需输入" in result.error


DUP_QUESTION = {
    "question": "订单系统里如何保证下单与扣库存的数据一致性？",
    "assess_point": "分布式事务理解",
    "answer_points": ["最终一致性方案", "本地消息表", "幂等补偿"],
    "follow_up": "消息堆积怎么办？",
}


def test_agent_dedupes_same_question_across_dimensions():
    """同义题（题干不同但答题结构重合）跨维度只保留一次"""
    dims = json.dumps({
        "position": "后端工程师",
        "dimensions": [
            {"key": "project", "label": "项目深挖", "focus": ""},
            {"key": "tcc", "label": "分布式事务", "focus": ""},
        ],
    }, ensure_ascii=False)
    dup_qs = json.dumps({"questions": [DUP_QUESTION]}, ensure_ascii=False)
    rewrite_qs = json.dumps({"questions": [{
        "question": "你们的订单和库存服务之间怎么保证数据一致性？",
        "assess_point": "分布式事务理解",
        "answer_points": ["最终一致性方案", "本地消息表", "幂等补偿"],
        "follow_up": "消息堆积怎么办？",
    }]}, ensure_ascii=False)

    mock = MagicMock()
    mock.generate.side_effect = lambda prompt, **kwargs: (
        dims if "dimensions" in prompt else (
            dup_qs if "为「项目深挖」" in prompt else rewrite_qs
        )
    )

    guide = InterviewPrepAgent(llm=mock).run(
        {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}
    )["interview_guide"]

    assert len(guide["questions"]) == 1
    assert guide["questions"][0]["category"] == "项目深挖"


def test_agent_caps_llm_dimensions_at_three():
    dims = json.dumps({
        "position": "后端工程师",
        "dimensions": [
            {"key": f"d{i}", "label": f"维度{i}", "focus": ""} for i in range(1, 5)
        ],
    }, ensure_ascii=False)

    mock = MagicMock()
    topics = {
        1: ("讲一段你把接口 P99 从 800ms 压到 220ms 的过程", ["瓶颈定位", "索引改造", "回归验证"]),
        2: ("订单与库存跨服务时你选 TCC 还是本地消息表，为什么", ["一致性取舍", "补偿路径", "失败演练"]),
        3: ("Kafka 堆积到百万级消息时你的排查顺序", ["看 Lag", "扩消费者", "死信队列"]),
        4: ("链路追踪上线后你靠它定位过哪次线上故障", ["TraceID 串联", "慢调用分析", "告警联动"]),
    }

    def generate(prompt, **kwargs):
        if "dimensions" in prompt:
            return dims
        for i in range(1, 5):
            if f"为「维度{i}」" in prompt:
                question, points = topics[i]
                return json.dumps({"questions": [{
                    "question": question,
                    "assess_point": f"维度{i}考察点",
                    "answer_points": points,
                    "follow_up": f"维度{i}的追问",
                }]}, ensure_ascii=False)
        return json.dumps({"questions": []}, ensure_ascii=False)

    mock.generate.side_effect = generate

    guide = InterviewPrepAgent(llm=mock).run(
        {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}
    )["interview_guide"]

    # 4 个 LLM 维度收敛为 1 个（其余同主题丢弃）+ 2 个默认维度补齐
    assert [q["category"] for q in guide["questions"][:3]] == [
        "维度1", "项目经历深挖", "技术基础与深度",
    ]
    assert guide["questions"][0]["question"] == topics[1][0]
    assert all(q["question"] for q in guide["questions"])


def test_executor_returns_interview_guide():
    result = InterviewPrepExecutor(llm=_mock_llm()).run({
        "resume": SAMPLE_RESUME,
        "jd": SAMPLE_JD,
        "step_id": "step_1",
    })

    assert result.success is True
    assert result.step_id == "step_1"
    assert result.data["interview_guide"]["questions"]
    assert "高频面试问题" in result.summary


def test_executor_streams_one_dimension_at_a_time():
    """流式执行：每完成一个维度就推一份增量攻略，前端边生成边渲染"""
    events = list(InterviewPrepExecutor(llm=_mock_llm()).run_stream({
        "resume": SAMPLE_RESUME,
        "jd": SAMPLE_JD,
        "step_id": "step_1",
    }))

    guides = [e["data"]["interview_guide"] for e in events if e["type"] == "interview"]
    assert [len(g["questions"]) for g in guides] == [1, 2]
    assert all(e["data"]["partial"] is True for e in events if e["type"] == "interview")
    assert guides[-1]["questions"][-1]["category"] == "技术基础"
    assert guides[-1]["strategy"]["summary"]


def test_executor_stream_reports_missing_resume():
    events = list(InterviewPrepExecutor(llm=_mock_llm()).run_stream({"resume": ""}))

    assert events == [{"type": "error", "content": "缺少必需输入: resume"}]


def test_agent_dedupes_overlapping_dimensions_and_pads_to_three():
    """LLM 把同一主题拆成两个维度时只保留一个，并用默认维度补齐到 3 个"""
    dims = json.dumps({
        "position": "后端工程师",
        "dimensions": [
            {"key": "a", "label": "项目经历深挖", "focus": "追问个人贡献"},
            {"key": "b", "label": "项目经历深挖", "focus": "追问个人贡献"},
        ],
    }, ensure_ascii=False)

    topics = {
        "项目经历深挖": (
            "订单系统里你如何解决扣库存超卖？",
            ["锁定粒度", "热点行分散", "对账兜底"],
        ),
        "技术基础与深度": (
            "Redis 集群脑裂时你们怎么保证写入不丢？",
            ["min-replicas", "AOF 落盘", "故障演练"],
        ),
        "岗位匹配度": (
            "你没做过 Go 服务，怎么在三个月内接手网关维护？",
            ["相似经验迁移", "学习路径", "里程碑验收"],
        ),
    }

    def generate(prompt, **kwargs):
        if "dimensions" in prompt:
            return dims
        for label, (question, points) in topics.items():
            if f"为「{label}」" in prompt:
                return json.dumps({"questions": [{
                    "question": question,
                    "assess_point": f"{label}考察点",
                    "answer_points": points,
                    "follow_up": f"{label}的追问",
                }]}, ensure_ascii=False)
        return QUESTIONS_JSON

    mock = MagicMock()
    mock.generate.side_effect = generate

    guide = InterviewPrepAgent(llm=mock).run(
        {"resume": SAMPLE_RESUME, "jd": SAMPLE_JD}
    )["interview_guide"]

    categories = [q["category"] for q in guide["questions"]]
    assert len(categories) == 3
    assert categories.count("项目经历深挖") == 1
    assert categories[1:] == ["技术基础与深度", "岗位匹配度"]


def test_agent_skips_question_generation_when_cancelled():
    """客户端断开后不再发起出题调用（三维度并发出题，断开即整批跳过）"""
    mock = _mock_llm()
    agent = InterviewPrepAgent(llm=mock, should_stop=lambda: True)
    guide = agent.run({"resume": SAMPLE_RESUME, "jd": SAMPLE_JD})["interview_guide"]

    assert mock.generate.call_count == 1  # 仅维度规划一次
    assert guide["source"] == "rule"
    assert guide["questions"]


def test_agent_falls_back_to_rules_when_model_missing():
    """本地模型加载失败也不抛错：构造阶段降级，输出规则题库"""
    from agents.skills.resume_agents import interview_agent as module

    original = module.get_agent

    def _raise(*args, **kwargs):
        raise RuntimeError("模型文件缺失")

    module.get_agent = _raise
    try:
        agent = InterviewPrepAgent()
        guide = agent.run({"resume": SAMPLE_RESUME, "jd": SAMPLE_JD})["interview_guide"]
    finally:
        module.get_agent = original

    assert agent.llm is None
    assert guide["source"] == "rule"
    assert guide["questions"]
    assert guide["strategy"]["summary"]
