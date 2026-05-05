import pytest
from unittest.mock import MagicMock, patch
from agents.langgraph.resume_agents.score_agent import ResumeScoreAgent
from agents.langgraph.resume_agents.state import ResumeState


SAMPLE_RESUME = """
张三，高级后端工程师

工作经历：
2020-2023 - 某科技公司 - 负责微服务架构设计和开发
2018-2020 - 某互联网公司 - 参与电商平台开发

技能：Python、Go、MySQL、Redis、Kubernetes
"""

SAMPLE_SCORE_JSON = '{"completeness": 85, "professionalism": 80, "quantification": 70, "matching": 75}'

SAMPLE_SCORE_WITH_NESTED = '{"scores": {"completeness": 90, "professionalism": 85, "quantification": 75, "matching": 80}}'


class TestResumeScoreAgent:
    """简历评分 Agent 测试套件"""

    def _make_mock_llm(self, return_value: str = SAMPLE_SCORE_JSON):
        mock = MagicMock()
        mock.generate.return_value = return_value
        return mock

    @pytest.fixture
    def agent(self):
        mock_llm = self._make_mock_llm()
        return ResumeScoreAgent(llm=mock_llm)

    @pytest.fixture
    def base_state(self) -> ResumeState:
        return {
            "resume": SAMPLE_RESUME,
            "jd": None,
            "position_type": None,
            "score_result": None,
            "match_result": None,
            "overall_score": None,
            "suggestions": None,
            "optimized_resume": None,
            "error": None,
            "current_step": "started"
        }

    # ---- 基础功能测试 ----

    def test_agent_initialization_with_llm(self):
        """Agent 应该正确初始化并接收 LLM 实例"""
        mock_llm = self._make_mock_llm()
        agent = ResumeScoreAgent(llm=mock_llm)
        assert agent.llm is mock_llm
        assert agent.name == "resume_score_agent"

    def test_agent_auto_init_llm(self):
        """未传入 LLM 时应该自动从 registry 获取"""
        with patch("agents.langgraph.resume_agents.score_agent.get_agent") as mock_get:
            mock_get.return_value = self._make_mock_llm()
            agent = ResumeScoreAgent()
            mock_get.assert_called_once_with(provider="local")

    def test_run_returns_valid_state(self, agent, base_state):
        """run 方法应返回包含评分结果的 ResumeState"""
        result = agent.run(base_state)
        assert isinstance(result, dict)
        assert "score_result" in result
        assert "overall_score" in result
        assert result["error"] is None

    def test_run_calls_llm_generate(self, agent, base_state):
        """run 方法应调用 LLM 的 generate 方法"""
        agent.run(base_state)
        agent.llm.generate.assert_called_once()

    def test_run_updates_current_step(self, agent, base_state):
        """run 方法完成后 current_step 应为 score_completed"""
        result = agent.run(base_state)
        assert result["current_step"] == "score_completed"

    # ---- 评分解析测试 ----

    def test_parse_standard_json_scores(self, agent):
        """标准 JSON 格式应正确解析"""
        scores = agent._parse_scores(SAMPLE_SCORE_JSON)
        assert scores["completeness"] == 85
        assert scores["professionalism"] == 80
        assert scores["quantification"] == 70
        assert scores["matching"] == 75

    def test_parse_nested_json_scores(self, agent):
        """嵌套 JSON 格式（含 scores key）应正确解析"""
        scores = agent._parse_scores(SAMPLE_SCORE_WITH_NESTED)
        assert scores["completeness"] == 90
        assert scores["professionalism"] == 85

    def test_parse_malformed_json_falls_back_to_regex(self, agent):
        """损坏的 JSON 应回退到正则提取"""
        malformed = "完整性: 88 专业度: 82 量化程度: 72 匹配度: 78"
        scores = agent._parse_scores(malformed)
        assert scores.get("completeness") == 88
        assert scores.get("professionalism") == 82

    def test_parse_numbers_only_fallback(self, agent):
        """纯数字文本应提取前4个数字作为评分"""
        text = "评分结果如下: 90 85 75 80"
        scores = agent._parse_scores(text)
        assert scores["completeness"] == 90
        assert scores["matching"] == 80

    def test_parse_garbage_returns_defaults(self, agent):
        """无法解析的内容应返回默认值"""
        scores = agent._parse_scores("无法解析的随机文本")
        default_keys = {"completeness", "professionalism", "quantification", "matching"}
        assert set(scores.keys()) == default_keys
        for v in scores.values():
            assert isinstance(v, int)
            assert 0 <= v <= 100

    # ---- 综合评分计算测试 ----

    def test_overall_score_is_average(self, agent, base_state):
        """综合评分应为各维度平均值"""
        result = agent.run(base_state)
        expected_avg = sum(result["score_result"].values()) / len(result["score_result"])
        assert abs(result["overall_score"]["score"] - expected_avg) < 0.01

    def test_overall_score_rating_mapping(self, agent, base_state):
        """分数段应映射到正确的评级"""
        result = agent.run(base_state)
        score = result["overall_score"]["score"]
        rating = result["overall_score"]["rating"]
        ratings_map = {
            (90, 100): "优秀",
            (80, 90): "良好",
            (70, 80): "中等",
            (60, 70): "及格",
            (0, 60): "需改进"
        }
        for (low, high), label in ratings_map.items():
            if low <= score < high:
                assert rating == label
                break

    # ---- 错误处理测试 ----

    def test_llm_error_sets_default_scores(self, base_state):
        """LLM 异常时应设置默认评分而非崩溃"""
        mock_llm = MagicMock()
        mock_llm.generate.side_effect = RuntimeError("模型推理失败")
        agent = ResumeScoreAgent(llm=mock_llm)
        result = agent.run(base_state)
        assert result["error"] is not None
        assert result["score_result"]["completeness"] == 70
        assert result["overall_score"]["score"] == 67.5

    def test_empty_resume_handling(self, agent):
        """空简历不应导致崩溃"""
        state: ResumeState = {
            "resume": "",
            "jd": None,
            "position_type": None,
            "score_result": None,
            "match_result": None,
            "overall_score": None,
            "suggestions": None,
            "optimized_resume": None,
            "error": None,
            "current_step": "started"
        }
        result = agent.run(state)
        assert "score_result" in result

    # ---- 边界情况测试 ----

    def test_extreme_high_scores(self, agent):
        """满分应返回优秀评级"""
        rating = agent._get_rating(100.0)
        assert rating == "优秀"

    def test_extreme_low_scores(self, agent):
        """低分应返回需改进评级"""
        rating = agent._get_rating(10.0)
        assert rating == "需改进"

    def test_boundary_ratings(self, agent):
        """边界分数评级测试"""
        assert agent._get_rating(90.0) == "优秀"
        assert agent._get_rating(89.9) == "良好"
        assert agent._get_rating(80.0) == "良好"
        assert agent._get_rating(79.9) == "中等"
        assert agent._get_rating(70.0) == "中等"
        assert agent._get_rating(69.9) == "及格"
        assert agent._get_rating(59.9) == "需改进"

    def test_description_generation(self, agent):
        """不同分数段应有不同描述"""
        descriptions = [agent._get_description(s) for s in [95, 85, 75, 65, 45]]
        assert len(set(descriptions)) > 1
