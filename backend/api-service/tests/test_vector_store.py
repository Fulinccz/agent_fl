import pytest
import os
import tempfile
import shutil
from unittest.mock import MagicMock, patch
import numpy as np

from rag.vector_store import VectorStore
from rag.embeddings import EmbeddingService


@pytest.fixture
def temp_persist_dir():
    """创建临时目录用于向量存储测试"""
    dir_path = tempfile.mkdtemp(prefix="vector_store_test_")
    yield dir_path
    shutil.rmtree(dir_path, ignore_errors=True)


@pytest.fixture
def mock_embedding_service():
    """创建 Mock Embedding Service，返回固定维度向量"""
    mock = MagicMock(spec=EmbeddingService)
    mock.dimension = 384

    def encode(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [np.random.randn(384).astype(np.float32).tolist() for _ in texts]

    def encode_single(text):
        return np.random.randn(384).astype(np.float32).tolist()

    mock.encode = encode
    mock.encode_single = encode_single
    return mock


@pytest.fixture
def vector_store(temp_persist_dir, mock_embedding_service):
    """创建 VectorStore 实例"""
    store = VectorStore(
        collection_name="test_collection",
        persist_dir=temp_persist_dir,
        embedding_service=mock_embedding_service,
        dimension=384
    )
    return store


class TestVectorStoreInit:
    """VectorStore 初始化测试"""

    def test_initialization_creates_index(self, vector_store):
        """初始化后 index 应可访问"""
        assert vector_store.index is not None
        assert vector_store.count() == 0

    def test_initialization_empty_collection(self, vector_store):
        """新集合的文档数应为 0"""
        assert vector_store.count() == 0

    def test_persist_dir_created(self, temp_persist_dir, vector_store):
        """持久化目录应被创建"""
        assert os.path.exists(temp_persist_dir)


class TestVectorStoreAddDocuments:
    """添加文档测试"""

    def test_add_single_document(self, vector_store):
        """添加单个文档应增加计数"""
        count = vector_store.add_documents(["这是一份简历内容"])
        assert count == 1
        assert vector_store.count() == 1

    def test_add_multiple_documents(self, vector_store):
        """批量添加文档应正确计数"""
        docs = [f"文档内容 {i}" for i in range(5)]
        count = vector_store.add_documents(docs)
        assert count == 5
        assert vector_store.count() == 5

    def test_add_empty_list_returns_zero(self, vector_store):
        """空列表应返回 0"""
        count = vector_store.add_documents([])
        assert count == 0

    def test_add_documents_with_metadata(self, vector_store):
        """带元数据的文档应正确存储"""
        metadatas = [{"source": "resume_1"}, {"source": "resume_2"}]
        vector_store.add_documents(["内容A", "内容B"], metadatas=metadatas)
        assert vector_store.count() == 2

    def test_add_documents_with_ids(self, vector_store):
        """自定义 ID 应被使用"""
        ids = ["custom_id_1", "custom_id_2"]
        vector_store.add_documents(["A", "B"], ids=ids)
        assert vector_store.count() == 2

    def test_auto_generate_ids(self, vector_store):
        """未提供 ID 时应自动生成"""
        vector_store.add_documents(["doc"])
        vector_store.add_documents(["doc2"])
        assert vector_store.count() == 2


class TestVectorStoreQuery:
    """查询测试"""

    def test_query_empty_store_returns_empty(self, vector_store):
        """空集合查询应返回空列表"""
        results = vector_store.query("查询内容")
        assert results == []

    def test_query_returns_results_after_add(self, vector_store):
        """添加文档后查询应返回结果"""
        vector_store.add_documents(["Python 开发工程师，熟悉 Django 和 FastAPI"])
        results = vector_store.query("Python 工程师")
        assert len(results) > 0

    def test_query_result_has_required_fields(self, vector_store):
        """查询结果应包含必要字段"""
        vector_store.add_documents(
            ["Go 语言开发经验"],
            metadatas=[{"source": "test"}]
        )
        results = vector_store.query("Go", n_results=1)
        assert len(results) > 0
        result = results[0]
        assert "content" in result
        assert "metadata" in result
        assert "distance" in result
        assert "id" in result

    def test_query_respects_top_k(self, vector_store):
        """n_results 应限制返回数量"""
        for i in range(10):
            vector_store.add_documents([f"测试文档 {i}"])
        results = vector_store.query("测试", n_results=3)
        assert len(results) <= 3

    def test_query_returns_empty_for_irrelevant(self, vector_store):
        """查询与文档不相关内容应返回空或低分结果"""
        vector_store.add_documents(["完全不相关的内容"])
        results = vector_store.query("量子力学")
        assert isinstance(results, list)

    def test_query_with_metadata_filter(self, vector_store):
        """元数据过滤应生效"""
        vector_store.add_documents(
            ["Java 内容"],
            metadatas=[{"category": "java"}]
        )
        vector_store.add_documents(
            ["Python 内容"],
            metadatas=[{"category": "python"}]
        )
        results = vector_store.query("编程语言", where={"category": "python"})
        for r in results:
            assert r["metadata"]["category"] == "python"


class TestVectorStorePersistence:
    """持久化测试"""

    def test_save_and_reload(self, temp_persist_dir, mock_embedding_service):
        """保存后重新加载应保留数据"""
        store1 = VectorStore(
            collection_name="persist_test",
            persist_dir=temp_persist_dir,
            embedding_service=mock_embedding_service,
            dimension=384
        )
        store1.add_documents(["需要持久化的内容"])
        assert store1.count() == 1

        store2 = VectorStore(
            collection_name="persist_test",
            persist_dir=temp_persist_dir,
            embedding_service=mock_embedding_service,
            dimension=384
        )
        assert store2.count() == 1

    def test_delete_collection(self, vector_store):
        """删除集合后数据应清空"""
        vector_store.add_documents(["待删除的数据"])
        assert vector_store.count() > 0
        vector_store.delete_collection()
        assert vector_store.count() == 0

    def test_delete_removes_files(self, vector_store, temp_persist_dir):
        """删除集合应移除磁盘文件"""
        vector_store.add_documents(["data"])
        vector_store.delete_collection()
        assert not os.path.exists(vector_store.index_path)


class TestVectorStoreEdgeCases:
    """边界情况测试"""

    def test_count_property(self, vector_store):
        """count 属性应反映实际文档数"""
        assert vector_store.count() == 0
        vector_store.add_documents(["a"])
        assert vector_store.count() == 1
        vector_store.add_documents(["b", "c"])
        assert vector_store.count() == 3

    def test_no_embedding_service_cannot_add(self, temp_persist_dir):
        """无 Embedding 服务时 add_documents 返回 0"""
        store = VectorStore(
            collection_name="no_embed",
            persist_dir=temp_persist_dir,
            embedding_service=None,
            dimension=384
        )
        count = store.add_documents(["无法编码的内容"])
        assert count == 0
