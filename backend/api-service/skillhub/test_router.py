import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillhub import SkillRouterAgent, RouteResult
from skillhub.keyword_matcher import KeywordMatcher
from skillhub.vector_router import VectorSkillRouter


def test_keyword_matcher():
    print("\n" + "=" * 60)
    print("[TEST 1] KeywordMatcher L1 测试")
    print("=" * 60)

    matcher = KeywordMatcher()
    test_cases = [
        "帮我润色一下简历",
        "这个JD匹配度怎么样",
        "给我简历打个分",
        "提取一下简历信息",
        "你好，你能做什么",
        "帮我改改这段项目描述",
        "看看这个岗位适合我吗",
        "简历质量如何",
    ]

    for query in test_cases:
        result = matcher.match(query)
        status = "[HIT]" if result.confidence >= 0.9 else "[LOW]"
        print(f"   {status} '{query[:20]}...' -> {result.skill} (confidence={result.confidence:.2f})")


def test_vector_router():
    print("\n" + "=" * 60)
    print("[TEST 2] VectorSkillRouter L2 测试")
    print("=" * 60)

    router = VectorSkillRouter()
    test_queries = [
        "帮我优化一下简历措辞",
        "这个岗位我能投吗",
        "评估一下我的简历水平",
        "解析这份PDF简历",
        "你好",
    ]

    for query in test_queries:
        candidates = router.search(query, top_k=3)
        if candidates:
            best = candidates[0]
            print(f"   [OK] '{query[:20]}...' -> {best.skill} (score={best.score:.3f})")
            for c in candidates:
                print(f"        - {c.skill}: {c.score:.3f} | {c.text[:30]}")
        else:
            print(f"   [WARN] '{query[:20]}...' -> 无候选")


def test_full_router():
    print("\n" + "=" * 60)
    print("[TEST 3] SkillRouterAgent 完整路由测试")
    print("=" * 60)

    agent = SkillRouterAgent()
    test_queries = [
        "帮我润色简历",
        "这个JD我能投吗",
        "给我简历打个分",
        "提取简历技能",
        "你好",
        "帮我改改这段描述",
        "看看这个岗位匹配度",
        "简历写得怎么样",
        "解析一下我的简历",
        "你有什么功能",
    ]

    for query in test_queries:
        result = agent.route(query)
        print(f"   [{result.source:8s}] '{query[:20]}...' -> {result.skill} (confidence={result.confidence:.2f}) {result.reason[:30]}")


def main():
    print("=" * 60)
    print("  Skill Router Agent 测试套件")
    print("=" * 60)

    test_keyword_matcher()
    test_vector_router()
    test_full_router()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
