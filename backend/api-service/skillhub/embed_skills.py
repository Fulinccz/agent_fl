import json
import os
import sys
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embeddings import EmbeddingService
from rag.vector_store import VectorStore
from logger import get_logger

logger = get_logger(__name__)

SAMPLES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skill_query_samples.json")
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "skill_vector_db")
COLLECTION_NAME = "skill_router"


def load_samples(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        skill = item["skill"]
        for query in item["queries"]:
            samples.append({
                "text": query,
                "skill": skill,
                "type": "query_sample"
            })

    logger.info(f"Loaded {len(samples)} query samples from {path}")
    return samples


def build_skill_descriptions() -> List[Dict[str, Any]]:
    skill_descs = [
        {"text": "简历润色优化，提升措辞专业度，改进表达方式", "skill": "resume-polishing", "type": "description"},
        {"text": "简历与岗位描述JD的关键词匹配度分析", "skill": "jd-keyword-match", "type": "description"},
        {"text": "简历质量评分，综合评估竞争力与完整度", "skill": "resume-score", "type": "description"},
        {"text": "简历信息结构化提取，解析技能与经历", "skill": "resume-parse", "type": "description"},
        {"text": "通用对话与求职咨询，功能介绍与使用帮助", "skill": "chat-general", "type": "description"},
    ]
    return skill_descs


def embed_skills(force_rebuild: bool = False):
    print("=" * 60)
    print("  Skill Router 向量化")
    print("=" * 60)

    embedding_service = EmbeddingService()
    dim = embedding_service.dimension
    logger.info(f"Embedding dimension: {dim}")

    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=PERSIST_DIR,
        embedding_service=embedding_service,
        dimension=dim
    )

    if not force_rebuild:
        current_count = vector_store.count()
        if current_count > 0:
            print(f"[OK] 向量库已存在: {current_count} 条，跳过构建")
            print("   如需重建，添加 --rebuild 参数")
            return

    samples = load_samples(SAMPLES_PATH)
    descriptions = build_skill_descriptions()
    all_items = samples + descriptions

    print(f"[INFO] 总样本数: {len(all_items)} (查询样本 {len(samples)} + 描述 {len(descriptions)})")

    texts = [item["text"] for item in all_items]
    metadatas = [{"skill": item["skill"], "type": item["type"]} for item in all_items]
    ids = [f"skill_{i:04d}" for i in range(len(all_items))]

    print("[STEP] 正在编码...")
    embeddings = embedding_service.encode(texts, batch_size=32, show_progress_bar=True)

    print("[STEP] 写入向量库...")
    vector_store.delete_collection()
    vector_store.add_documents(texts, metadatas, ids)

    count = vector_store.count()
    print(f"[OK] 向量化完成: {count} 条技能样本已入库")
    print(f"   存储路径: {PERSIST_DIR}")
    print(f"   Collection: {COLLECTION_NAME}")
    print("=" * 60)


def test_search(query: str, top_k: int = 3):
    embedding_service = EmbeddingService()
    dim = embedding_service.dimension
    vector_store = VectorStore(
        collection_name=COLLECTION_NAME,
        persist_dir=PERSIST_DIR,
        embedding_service=embedding_service,
        dimension=dim
    )

    print(f"\n[TEST] 查询: '{query}'")
    results = vector_store.query(query_text=query, n_results=top_k)

    for i, r in enumerate(results, 1):
        skill = r["metadata"].get("skill", "unknown")
        score = r.get("distance", 0)
        text = r["content"]
        print(f"   [{i}] {skill} (score={score:.4f}): {text}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="强制重建向量库")
    parser.add_argument("--test", type=str, help="测试搜索查询")
    args = parser.parse_args()

    if args.test:
        test_search(args.test)
    else:
        embed_skills(force_rebuild=args.rebuild)
