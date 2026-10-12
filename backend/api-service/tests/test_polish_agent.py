"""ResumePolishAgent 测试

覆盖：
- run_stream 正常透传 token
- run_stream LLM 异常时错误事件使用 content 键（master _iter_streaming 契约）
- _clean_result 清理 think/代码围栏/角色标记
- 清理后过短时回退原始简历
"""
from unittest.mock import MagicMock

import pytest

from agents.skills.resume_agents.polish_agent import ResumePolishAgent


def _agent_with_llm(llm):
    return ResumePolishAgent(llm=llm)


def test_run_stream_passes_tokens_through():
    mock_llm = MagicMock()
    mock_llm.generate_stream.return_value = iter([
        {"type": "token", "content": "工作经历："},
        {"type": "token", "content": "主导核心系统"},
    ])
    agent = _agent_with_llm(mock_llm)
    chunks = list(agent.run_stream({"resume": "原始简历", "jd": None}))
    assert chunks == [
        {"type": "token", "content": "工作经历："},
        {"type": "token", "content": "主导核心系统"},
    ]


def test_run_stream_error_chunk_uses_content_key():
    """回归：错误事件必须用 content 键，master._iter_streaming 据此捕获失败"""
    mock_llm = MagicMock()
    mock_llm.generate_stream.side_effect = RuntimeError("llm down")
    agent = _agent_with_llm(mock_llm)

    chunks = list(agent.run_stream({"resume": "原始简历", "jd": None}))
    assert len(chunks) == 1
    assert chunks[0]["type"] == "error"
    assert "content" in chunks[0]
    assert "llm down" in chunks[0]["content"]
    assert "message" not in chunks[0]


def test_clean_result_strips_think_fence_and_role_markers():
    agent = _agent_with_llm(MagicMock())
    raw = (
        "<think>思考过程</think>\n"
        "```\n工作经历：主导订单交易系统架构升级，核心链路性能提升三倍\n"
        "技能：Java、分布式架构、Kubernetes 云原生体系\n```\n"
        "assistant:\n**教育经历**：某大学计算机专业，主修操作系统与编译原理"
    )
    cleaned = agent._clean_result(raw, "原始简历内容" * 20)
    assert "<think>" not in cleaned
    assert "```" not in cleaned
    assert "工作经历：主导订单交易系统架构升级" in cleaned
    assert "教育经历" in cleaned
    assert "**" not in cleaned


def test_clean_result_falls_back_when_too_short():
    agent = _agent_with_llm(MagicMock())
    original = "详细的原始简历内容" * 20
    # 清理后只剩极短残片 → 回退原文，避免产出空简历
    cleaned = agent._clean_result("<think>x</think>", original)
    assert cleaned == original[:1000]


def test_non_stream_run_sets_optimized_resume():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "工作经历：主导核心交易系统，性能提升显著" * 3
    agent = _agent_with_llm(mock_llm)
    state = {"resume": "原始简历" * 100, "jd": None}
    result_state = agent.run(state)
    assert result_state["optimized_resume"]
    assert "核心交易系统" in result_state["optimized_resume"]


# ---------- 长文分段（消除 prompt 截断丢内容）----------

def test_short_resume_calls_llm_once():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "工作经历：主导核心系统，性能提升显著" * 4
    agent = _agent_with_llm(mock_llm)
    agent.run({"resume": "短简历" * 100, "jd": None})  # 300 字
    assert mock_llm.generate.call_count == 1
    prompt = mock_llm.generate.call_args.args[0]
    assert "第 1/1 部分" not in prompt


def test_long_resume_non_stream_splits_into_segments():
    mock_llm = MagicMock()
    # 每段返回不同的可识别内容
    mock_llm.generate.side_effect = lambda prompt, **kw: (
        "第一段优化结果：主导订单系统，性能提升三倍，内容足够长以通过清理阈值。"
        if "第 1/" in prompt else "第二段优化结果：主导支付清结算，资金准确率提升。"
    )
    agent = _agent_with_llm(mock_llm)
    resume = "工作经历内容。" * 1000  # 7000 字 → 3 段
    state = agent.run({"resume": resume, "jd": None})

    assert mock_llm.generate.call_count == 3
    out = state["optimized_resume"]
    assert "订单系统" in out and "支付清结算" in out
    # 每段 prompt 都应包含完整简历的对应部分（不再 resume[:3000] 截断）
    prompts = [c.args[0] for c in mock_llm.generate.call_args_list]
    assert all("第 3/3 部分" in p for p in prompts[2:])


def test_long_resume_stream_emits_separator_between_segments():
    mock_llm = MagicMock()

    def fake_stream(prompt, **kwargs):
        marker = "甲段" if "第 1/" in prompt else "乙段"
        yield {"type": "token", "content": marker}

    mock_llm.generate_stream.side_effect = fake_stream
    agent = _agent_with_llm(mock_llm)

    chunks = list(agent.run_stream({"resume": "段落内容。" * 600, "jd": None}))  # 3000 字 → 2 段
    assert mock_llm.generate_stream.call_count == 2
    # 段间补一个空行 token
    contents = [c["content"] for c in chunks if c["type"] == "token"]
    assert contents == ["甲段", "\n\n", "乙段"]


def test_extra_instruction_enters_prompt():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "工作经历：主导核心系统，性能提升显著，成果丰富" * 3
    agent = _agent_with_llm(mock_llm)
    agent.run({
        "resume": "原始简历" * 100,
        "jd": None,
        "extra_instruction": "突出带团队经验",
    })
    prompt = mock_llm.generate.call_args.args[0]
    assert "用户特别要求" in prompt
    assert "突出带团队经验" in prompt


def test_long_resume_tail_not_dropped():
    """回归：旧实现 resume[:3000] 会丢弃超长简历尾部，分段后末段必须进入 prompt"""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "优化后的工作经历描述，内容长度足以通过清理阈值检查。"
    agent = _agent_with_llm(mock_llm)
    tail_marker = "TAIL_MARKER_尾部项目"
    resume = "前段经历。" * 1000 + tail_marker  # 尾部标记
    agent.run({"resume": resume, "jd": None})
    prompts = [c.args[0] for c in mock_llm.generate.call_args_list]
    assert any(tail_marker in p for p in prompts)
