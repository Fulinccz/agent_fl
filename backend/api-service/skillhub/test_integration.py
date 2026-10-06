"""
Skill Router 集成测试

测试 SkillExecutor 与 SkillRouterAgent 的集成效果。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from skillhub.registry import get_skill_executor


def test_skill_executor():
    print("\n" + "=" * 60)
    print("[TEST] SkillExecutor + SkillRouterAgent 集成测试")
    print("=" * 60)

    executor = get_skill_executor()

    test_cases = [
        "帮我润色一下简历",
        "这个JD匹配度怎么样",
        "给我简历打个分",
        "提取一下简历信息",
        "你好",
        "帮我改改这段项目描述",
        "看看这个岗位适合我吗",
        "简历质量如何",
    ]

    for query in test_cases:
        route = executor.auto_select_skill(query)
        source = route["source"]
        skill = route["skill"]
        conf = route["confidence"]
        reason = route["reason"][:30]
        print(f"   [{source:8s}] '{query[:20]}...' -> {skill} (confidence={conf:.2f}) {reason}")


def test_available_skills():
    print("\n" + "=" * 60)
    print("[TEST] 列出可用技能")
    print("=" * 60)

    executor = get_skill_executor()
    skills = executor.list_available_skills()
    for skill in skills:
        print(f"   - {skill}")


def main():
    print("=" * 60)
    print("  Skill Router 集成测试")
    print("=" * 60)

    test_skill_executor()
    test_available_skills()

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
