"""skillhub/vector_router.py 相似度方向与阈值测试

vector_store.query 返回余弦距离（0=完全相同，越大越不相似）；
路由层必须还原成"越大越相似"的相似度后再做阈值判断。
"""
from unittest.mock import MagicMock

from skillhub.vector_router import VectorSkillRouter


def _build_router(fake_results, threshold=0.5):
    router = VectorSkillRouter(similarity_threshold=threshold)
    fake_store = MagicMock()
    fake_store.query.return_value = fake_results
    router._vector_store = fake_store
    return router


def _fake_result(distance: float, skill: str = "resume_score") -> dict:
    return {
        "content": "sample",
        "metadata": {"skill": skill},
        "distance": distance,
        "id": "1",
    }


class TestVectorRouterScoring:
    def test_high_similarity_small_distance_kept(self):
        # 距离 0.05 → 相似度 0.95，应保留
        router = _build_router([_fake_result(0.05)])
        candidates = router.search("帮我给简历打分")
        assert len(candidates) == 1
        assert abs(candidates[0].score - 0.95) < 1e-6
        assert candidates[0].skill == "resume_score"

    def test_low_similarity_large_distance_filtered(self):
        # 距离 0.8 → 相似度 0.2，低于 0.5 阈值，必须过滤
        router = _build_router([_fake_result(0.8)])
        assert router.search("无关问题") == []

    def test_threshold_boundary_inclusive(self):
        # 距离 0.5 → 相似度恰好 0.5，按 >= 阈值保留
        router = _build_router([_fake_result(0.5)])
        candidates = router.search("边界用例")
        assert len(candidates) == 1
        assert candidates[0].score == 0.5

    def test_confident_skill_requires_high_similarity(self):
        # 旧实现的反向 bug：距离 0.9（极不相似）反而会被当成高置信
        router = _build_router([_fake_result(0.9, skill="resume_score")])
        bad = router.search("无关")
        assert router.get_confident_skill(bad, threshold=0.85) is None

        good_router = _build_router([_fake_result(0.05, skill="resume_score")])
        good = good_router.search("简历评分")
        assert good_router.get_confident_skill(good, threshold=0.85) == "resume_score"

    def test_search_failure_returns_empty(self):
        router = VectorSkillRouter(similarity_threshold=0.5)
        fake_store = MagicMock()
        fake_store.query.side_effect = RuntimeError("index missing")
        router._vector_store = fake_store
        assert router.search("anything") == []
