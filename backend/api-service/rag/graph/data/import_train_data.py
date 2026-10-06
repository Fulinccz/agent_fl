import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.graph.neo4j.neo4j_client import Neo4jClient
from rag.graph.data.train_data_builder import TrainDataBuilder
from logger import get_logger

logger = get_logger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
TRAIN_DATA_PATH = os.path.join(_PROJECT_ROOT, "ai", "train_data", "row_train.json")


def main():
    print("=" * 60)
    print("  从 row_train.json 导入知识图谱")
    print("=" * 60)

    client = Neo4jClient()
    if not client.health_check():
        print("[FAIL] Neo4j 连接失败，请检查:")
        print("   docker ps -a --filter name=neo4j-fulin")
        return

    print("[OK] Neo4j 连接成功")

    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"[FAIL] 训练数据文件不存在: {TRAIN_DATA_PATH}")
        return

    print(f"[OK] 找到训练数据: {TRAIN_DATA_PATH}")

    builder = TrainDataBuilder(client)

    print("\n[STEP 1] 初始化 Schema...")
    client.init_schema()

    print("\n[STEP 2] 清空旧训练数据...")
    client.run("""
        MATCH (t:TrainSample)
        OPTIONAL MATCH (t)-[r]->(n)
        WHERE NOT n:Skill AND NOT n:Ability AND NOT n:Scene
        DELETE r
        DETACH DELETE t
    """)
    print("[OK] 已清空旧训练样本")

    print("\n[STEP 3] 导入训练数据到图谱...")
    result = builder.build_from_json_file(TRAIN_DATA_PATH)

    print(f"\n[OK] 导入完成:")
    print(f"   总计: {result['total']} 条")
    print(f"   成功: {result['success']} 条")
    print(f"   失败: {result['failed']} 条")

    print("\n[STEP 4] 构建技能共现关系...")
    cooccurrence = builder.build_skill_relationships()
    print(f"   创建了 {len(cooccurrence)} 条技能共现关系")

    print("\n[STEP 5] 构建场景-技能映射...")
    scene_skills = builder.build_scene_skill_mapping()
    print(f"   创建了 {len(scene_skills)} 条场景-技能映射")

    print("\n[STEP 6] 统计信息...")
    stats = client.run("""
        MATCH (n)
        WITH labels(n)[0] AS label, count(n) AS count
        RETURN label, count
        ORDER BY count DESC
    """)
    print("\n[STATS] 节点统计:")
    for s in stats:
        print(f"   {s['label']}: {s['count']} 个")

    rel_stats = client.run("""
        MATCH ()-[r]->()
        WITH type(r) AS rel_type, count(r) AS count
        RETURN rel_type, count
        ORDER BY count DESC
    """)
    print("\n[STATS] 关系统计:")
    for s in rel_stats:
        print(f"   {s['rel_type']}: {s['count']} 条")

    print("\n[OK] 全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
