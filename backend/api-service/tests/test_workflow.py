import pytest
from unittest.mock import MagicMock, patch
from agents.langgraph.resume_agents.workflow import (
    ResumeOptimizationWorkflow,
    get_resume_workflow,
    optimize_stream,
    get_shared_llm,
)
from agents.langgraph.resume_agents.state import ResumeState


SAMPLE_RESUME = """
张三，高级后端工程师
工作经历：2020-2023 某科技公司 - 负责微服务架构设计
技能：Python、Go、MySQL、Redis、Kubernetes
"""

SAMPLE_JD = "招聘高级后端工程师，要求熟悉 Python/Go，有微服务经验"


class TestResumeOptimizationWorkflow:
    """简历优化工作流测试套件"""

    @pytest.fixture
    def mock_llm(self):
        mock = MagicMock()
        mock.generate.return_value = '{"completeness": 80, "professionalism": 75, "quantification": 70, "matching": 80}'

        def stream_generator(prompt, **kwargs):
            yield {"type": "token", "content": "优化后的"}
            yield {"type": "token", "content": "简历内容"}

        mock.generate_stream = stream_generator
        return mock

    @pytest.fixture(autouse=True)
    def _reset_globals(self):
        """每个测试前重置全局状态"""
        import agents.langgraph.resume_agents.workflow as wf_module
        wf_module._shared_llm_instance = None
        wf_module._resume_workflow = None
        yield
        wf_module._shared_llm_instance = None
        wf_module._resume_workflow = None

    # ---- 初始化测试 ----

    def test_workflow_initialization(self, mock_llm):
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            assert workflow is not None
            assert workflow.workflow is not None

    def test_get_shared_llm_singleton(self, mock_llm):
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm) as mock_get:
            llm1 = get_shared_llm()
            llm2 = get_shared_llm()
            assert llm1 is llm2
            mock_get.assert_called_once()

    # ---- optimize 测试 ----

    def test_optimize_returns_success_structure(self, mock_llm):
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            result = workflow.optimize(SAMPLE_RESUME)
            assert isinstance(result, dict)
            assert "success" in result
            assert "overall_score" in result
            assert "scores" in result
            assert "suggestions" in result
            assert "optimized_resume" in result
            assert "match_analysis" in result
            assert "error" in result

    def test_optimize_with_jd(self, mock_llm):
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            result = workflow.optimize(SAMPLE_RESUME, jd=SAMPLE_JD)
            assert result["match_analysis"] is not None

    def test_optimize_without_jd(self, mock_llm):
        """无 JD 时不应崩溃"""
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            result = workflow.optimize(SAMPLE_RESUME, jd=None)
            assert result["success"] is True

    # ---- optimize_stream 测试 ----

    def test_optimize_stream_yields_events(self, mock_llm):
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            events = list(workflow.optimize_stream(SAMPLE_RESUME))
            event_types = [e.get("type") for e in events]
            assert "score" in event_types
            assert "suggestions" in event_types
            assert "polished" in event_types
            assert "complete" in event_types

    def test_optimize_stream_order(self, mock_llm):
        """流式输出顺序应为 score → suggestions → polished → complete"""
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            events = list(workflow.optimize_stream(SAMPLE_RESUME))
            types = [e["type"] for e in events]

            score_idx = next((i for i, t in enumerate(types) if t == "score"), -1)
            suggestions_idx = next((i for i, t in enumerate(types) if t == "suggestions"), -1)
            polished_idx = next((i for i, t in enumerate(types) if t == "polished"), -1)
            complete_idx = next((i for i, t in enumerate(types) if t == "complete"), -1)

            assert score_idx < suggestions_idx < polished_idx < complete_idx

    def test_optimize_stream_error_handling(self, mock_llm):
        """Agent 错误时应产生 error 类型事件"""
        mock_llm.generate.side_effect = RuntimeError("模型错误")
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            workflow = ResumeOptimizationWorkflow()
            events = list(workflow.optimize_stream(SAMPLE_RESUME))
            error_events = [e for e in events if e.get("type") == "error"]
            assert len(error_events) > 0

    # ---- 全局实例测试 ----

    def test_get_resume_workflow_returns_same_instance(self, mock_llm):
        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            w1 = get_resume_workflow()
            w2 = get_resume_workflow()
            assert w1 is w2


class TestOptimizeStreamConvenience:
    """optimize_stream 便捷函数测试"""

    def test_convenience_function_yields(self):
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "{}"
        mock_llm.generate_stream = lambda prompt, **kwargs: iter([
            {"type": "token", "content": "结果"}
        ])

        with patch("agents.langgraph.resume_agents.workflow.get_agent", return_value=mock_llm):
            import agents.langgraph.resume_agents.workflow as wf_module
            wf_module._shared_llm_instance = None
            wf_module._resume_workflow = None

            events = list(optimize_stream(SAMPLE_RESUME))
            assert len(events) > 0
