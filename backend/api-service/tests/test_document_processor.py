import numpy as np
import pytest
from unittest.mock import MagicMock

from rag.document_processor import DocumentProcessor
from rag.retriever import RAGRetriever


@pytest.fixture
def processor():
    return DocumentProcessor(chunk_size=800)


@pytest.fixture
def small_processor():
    """小 chunk_size，用于强制触发合并/二次切分逻辑"""
    return DocumentProcessor(chunk_size=50)


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return str(p)


class TestSplitByHeadings:
    """标题切分逻辑"""

    def test_sections_split_by_headings(self, processor):
        text = "# 标题A\n内容A\n## 标题B\n内容B\n### 标题C\n内容C"
        sections = processor._split_by_headings(text)
        titles = [t for t, _ in sections]
        assert titles == ["标题A", "标题B", "标题C"]
        # 标题行保留在正文中，保证 chunk 自带上下文
        assert sections[0][1].startswith("# 标题A")
        assert "内容B" in sections[1][1]

    def test_preamble_before_first_heading(self, processor):
        text = "前言内容\n## 第一节\n正文"
        sections = processor._split_by_headings(text)
        assert sections[0][0] == ""
        assert "前言内容" in sections[0][1]
        assert sections[1][0] == "第一节"

    def test_no_headings_single_section(self, processor):
        text = "纯文本没有标题\n第二行"
        sections = processor._split_by_headings(text)
        assert len(sections) == 1
        assert sections[0][0] == ""

    def test_empty_lines_dropped(self, processor):
        sections = processor._split_by_headings("\n\n## 标题\n\n内容\n\n")
        assert len(sections) == 1
        assert sections[0][1] == "## 标题\n\n内容"


class TestProcessFile:
    """单文件切片"""

    def test_small_sections_merged(self, processor, tmp_path):
        path = _write(tmp_path, "kb.md", "## 甲\n内容甲\n## 乙\n内容乙")
        chunks = processor.process_file(path)
        # 两小节合并为一个 chunk，标题取首个
        assert len(chunks) == 1
        assert chunks[0].metadata["section"] == "甲"
        assert "内容甲" in chunks[0].content and "内容乙" in chunks[0].content

    def test_merge_respects_chunk_size(self, small_processor, tmp_path):
        sec = "## 标题\n" + "字" * 40
        path = _write(tmp_path, "kb.md", f"{sec}\n{sec}")
        chunks = small_processor.process_file(path)
        assert len(chunks) == 2
        assert all(len(c.content) <= 50 for c in chunks)

    def test_oversized_section_split(self, small_processor, tmp_path):
        # 单章 200 字 > chunk_size=50，触发 split_text 二次切分
        body = "## 长章节\n" + "。".join(["句子" * 10] * 10)
        path = _write(tmp_path, "kb.md", body)
        chunks = small_processor.process_file(path)
        assert len(chunks) > 1
        assert all(len(c.content) <= 50 for c in chunks)
        # 切分后的每个 chunk 都继承原章节标题
        assert all(c.metadata["section"] == "长章节" for c in chunks)

    def test_plain_txt_without_headings(self, small_processor, tmp_path):
        path = _write(tmp_path, "kb.txt", "段一\n" * 30)
        chunks = small_processor.process_file(path)
        assert len(chunks) > 1
        assert all(c.metadata["section"] == "" for c in chunks)

    def test_metadata_fields(self, processor, tmp_path):
        path = _write(tmp_path, "resume_kb.md", "## 评分\n内容")
        chunks = processor.process_file(path)
        assert chunks[0].metadata["source"] == "resume_kb.md"
        assert chunks[0].metadata["chunk_index"] == 0

    def test_chunk_ids_deterministic_and_unique(self, processor, tmp_path):
        path = _write(tmp_path, "kb.md", "## A\n内容A\n" * 5)
        chunks1 = processor.process_file(path)
        chunks2 = processor.process_file(path)
        ids1 = [c.chunk_id for c in chunks1]
        assert ids1 == [c.chunk_id for c in chunks2]
        assert len(set(ids1)) == len(ids1)
        assert ids1[0].startswith("kb_")

    def test_missing_file_returns_empty(self, processor, tmp_path):
        assert processor.process_file(str(tmp_path / "nope.md")) == []

    def test_empty_file_returns_empty(self, processor, tmp_path):
        path = _write(tmp_path, "empty.md", "   \n\n")
        assert processor.process_file(path) == []


class TestProcessDirectory:
    """目录级切片"""

    def test_multiple_files_processed_sorted(self, processor, tmp_path):
        _write(tmp_path, "b.md", "## 乙\n内容乙")
        _write(tmp_path, "a.md", "## 甲\n内容甲")
        chunks = processor.process_directory(str(tmp_path))
        # 文件名排序保证重建结果稳定
        assert chunks[0].metadata["source"] == "a.md"
        assert chunks[1].metadata["source"] == "b.md"

    def test_unsupported_extension_skipped(self, processor, tmp_path):
        _write(tmp_path, "kb.md", "## 标题\n内容")
        _write(tmp_path, "notes.pdf", "假装是PDF")
        _write(tmp_path, "data.json", "{}")
        chunks = processor.process_directory(str(tmp_path))
        assert {c.metadata["source"] for c in chunks} == {"kb.md"}

    def test_missing_directory_returns_empty(self, processor, tmp_path):
        assert processor.process_directory(str(tmp_path / "nope")) == []


class TestInitializeKnowledgeBase:
    """retriever 集成：从目录重建知识库（embedding 走 mock，FAISS 真实）"""

    @pytest.fixture
    def mock_embedding(self):
        mock = MagicMock()
        mock.dimension = 384
        # 固定向量：query 与文档向量一致，保证相似度超过 0.3 阈值
        fixed = np.random.RandomState(42).randn(384).astype(np.float32).tolist()

        def encode(texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]
            return [list(fixed) for _ in texts]

        def encode_single(text):
            return list(fixed)

        mock.encode = encode
        mock.encode_single = encode_single
        return mock

    @pytest.fixture
    def retriever(self, tmp_path, mock_embedding):
        r = RAGRetriever(
            collection_name="test_kb",
            persist_dir=str(tmp_path / "vector_db"),
        )
        r._embedding_service = mock_embedding
        return r

    def test_rebuild_from_directory(self, retriever, tmp_path):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        _write(kb_dir, "kb.md", "## 评分标准\n优秀简历使用量化数据\n## 技巧\n使用动作动词")

        result = retriever.initialize_knowledge_base(
            knowledge_dir=str(kb_dir), force_rebuild=True
        )
        assert result["status"] == "success"
        assert result["documents_added"] == result["total_chunks"] > 0
        assert retriever.vector_store.count() == result["total_chunks"]

    def test_force_rebuild_replaces_old_data(self, retriever, tmp_path):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        _write(kb_dir, "kb.md", "## 第一版\n旧内容")
        retriever.initialize_knowledge_base(knowledge_dir=str(kb_dir), force_rebuild=True)
        first_count = retriever.vector_store.count()

        _write(kb_dir, "kb.md", "## 第二版\n新内容，只有这一节")
        result = retriever.initialize_knowledge_base(knowledge_dir=str(kb_dir), force_rebuild=True)
        # 重建后数量以新文件为准，而不是累加
        assert retriever.vector_store.count() == result["total_chunks"]
        assert result["total_chunks"] <= first_count

    def test_empty_directory_returns_empty_status(self, retriever, tmp_path):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        result = retriever.initialize_knowledge_base(knowledge_dir=str(kb_dir), force_rebuild=True)
        assert result["status"] == "empty"

    def test_retrieve_returns_chunk_metadata(self, retriever, tmp_path):
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        _write(kb_dir, "resume_kb.md", "## 量化成果\n将响应时间从500ms降到100ms")
        retriever.initialize_knowledge_base(knowledge_dir=str(kb_dir), force_rebuild=True)

        results = retriever.retrieve("量化成果", top_k=3)
        assert results
        assert results[0]["metadata"]["source"] == "resume_kb.md"
