import os
import sys
import io
import argparse

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from rag import RAGRetriever
from logger import get_logger

logger = get_logger(__name__)

def init_knowledge_base(force=False):
    retriever = RAGRetriever()
    result = retriever.initialize_knowledge_base(force_rebuild=force)

    status = result.get("status")
    if status == "exists":
        print(f"[OK] Knowledge base exists with {result.get('count', 0)} document chunks")
        print(f"   To rebuild, use: python manage_vector_db.py init --force")
    elif status == "success":
        print(f"[OK] Knowledge base initialized successfully!")
        print(f"   - Documents added: {result.get('documents_added', 0)}")
        print(f"   - Total chunks: {result.get('total_chunks', 0)}")
        print(f"   - Collection name: {result.get('collection_name')}")
    elif status == "empty":
        print(f"[WARN] No documents found")
        print(f"   Please add .txt or .md files to data/resume_knowledge/ directory")
    else:
        print(f"[INFO] {result.get('message', status)}")

def add_document(file_path):
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    retriever = RAGRetriever()
    result = retriever.add_file(file_path)

    if result.get("status") == "success":
        print(f"[OK] Document added: {result.get('file')}")
        print(f"   - Chunks added: {result.get('chunks_added', 0)}")
    else:
        print(f"[ERROR] Add failed: {result.get('message')}")

def show_stats():
    retriever = RAGRetriever()
    stats = retriever.get_stats()

    print("=" * 50)
    print("Vector Database Statistics")
    print("=" * 50)
    print(f"Collection name: {stats['collection_name']}")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Storage path: {stats['persist_dir']}")
    print(f"Retrieval Top-K: {stats['top_k']}")
    print(f"Similarity threshold: {stats['similarity_threshold']}")
    print("=" * 50)

def test_retrieval(query="如何优化简历"):
    retriever = RAGRetriever()

    print(f"\n[TEST] Retrieval query: '{query}'\n")
    results = retriever.retrieve(query, top_k=3)

    if not results:
        print("[WARN] No relevant content retrieved (knowledge base may be empty or similarity too low)")
        return

    for i, result in enumerate(results, 1):
        similarity = result.get('similarity', 0)
        content = result.get('content', '')[:200]
        source = result.get('metadata', {}).get('source', 'Unknown')

        print(f"\n[Result {i}] Similarity: {similarity:.1%} | Source: {source}")
        print("-" * 40)
        print(content + "..." if len(result.get('content', '')) > 200 else content)

def main():
    parser = argparse.ArgumentParser(
        description='Fulin AI 向量库管理工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python manage_vector_db.py init              # 初始化知识库
  python manage_vector_db.py init --force      # 强制重建
  python manage_vector_db.py stats             # 查看统计
  python manage_vector_db.py add file.txt      # 添加文档
  python manage_vector_db.py test              # 测试检索
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # init 命令
    init_parser = subparsers.add_parser('init', help='初始化/构建向量库')
    init_parser.add_argument('--force', '-f', action='store_true',
                            help='强制重建（即使已存在）')
    
    # add 命令
    add_parser = subparsers.add_parser('add', help='添加新文档到向量库')
    add_parser.add_argument('file', help='文档文件路径 (.txt/.md)')
    
    # stats 命令
    subparsers.add_parser('stats', help='显示向量库统计信息')
    
    # test 命令
    test_parser = subparsers.add_parser('test', help='测试检索功能')
    test_parser.add_argument('--query', '-q', default='如何优化简历',
                           help='测试查询内容')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'init':
        init_knowledge_base(force=args.force)
    elif args.command == 'add':
        add_document(args.file)
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'test':
        test_retrieval(query=args.query)

if __name__ == '__main__':
    main()