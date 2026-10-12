"""多模态解析单元测试（mock OCR/STT/文档解析，不加载真实模型）"""
import pytest
from unittest import mock

from rag.multimodal import (
    detect_modality,
    parse_multimodal,
    MultimodalDependencyError,
)


class TestDetectModality:
    @pytest.mark.parametrize("name,expected", [
        ("a.pdf", "document"),
        ("a.DOCX", "document"),
        ("a.txt", "document"),
        ("a.png", "image"),
        ("a.JPG", "image"),
        ("a.webp", "image"),
        ("a.mp3", "audio"),
        ("a.wav", "audio"),
        ("a.m4a", "audio"),
    ])
    def test_detect(self, name, expected, tmp_path):
        f = tmp_path / name
        f.write_text("x", encoding="utf-8")
        assert detect_modality(str(f)) == expected

    def test_unsupported(self, tmp_path):
        f = tmp_path / "a.xyz"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            detect_modality(str(f))


class TestParseMultimodal:
    def test_document_uses_resume_parser(self, tmp_path):
        f = tmp_path / "r.pdf"
        f.write_text("x", encoding="utf-8")
        fake = {"full_content": "简历全文", "skills": "Java", "projects": "订单系统"}
        with mock.patch("rag.multimodal.parse_resume", return_value=fake):
            result = parse_multimodal(str(f))
        assert result["modality"] == "document"
        assert result["text"] == "简历全文"
        assert result["skills"] == "Java"
        assert result["projects"] == "订单系统"

    def test_image_uses_ocr(self, tmp_path):
        f = tmp_path / "r.png"
        f.write_bytes(b"\x89PNG")
        with mock.patch("rag.multimodal._ocr_image", return_value="图片中的文字"):
            result = parse_multimodal(str(f))
        assert result["modality"] == "image"
        assert result["text"] == "图片中的文字"
        assert result["skills"] is None

    def test_audio_uses_stt(self, tmp_path):
        f = tmp_path / "v.wav"
        f.write_bytes(b"RIFF")
        with mock.patch("rag.multimodal._transcribe_audio", return_value="语音转写内容"):
            result = parse_multimodal(str(f))
        assert result["modality"] == "audio"
        assert result["text"] == "语音转写内容"

    def test_missing_dependency_raises_friendly_error(self, tmp_path):
        f = tmp_path / "r.png"
        f.write_bytes(b"\x89PNG")
        with mock.patch("rag.multimodal._ocr_image",
                        side_effect=MultimodalDependencyError("need rapidocr")):
            with pytest.raises(MultimodalDependencyError):
                parse_multimodal(str(f))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_multimodal("nonexistent.pdf")

    def test_txt_read_directly_without_resume_parser(self, tmp_path):
        """.txt 直接按纯文本读取，不走只支持 pdf/doc/docx 的 parse_resume"""
        f = tmp_path / "r.txt"
        f.write_text("这是纯文本简历，技能：Java、Python", encoding="utf-8")
        result = parse_multimodal(str(f))
        assert result["modality"] == "document"
        assert result["skills"] is None
        assert result["projects"] is None
        assert "Java" in result["text"]
        assert result["metadata"]["engine"] == "plaintext"

    def test_txt_gbk_fallback(self, tmp_path):
        f = tmp_path / "gbk.txt"
        f.write_bytes("中文简历GBK编码".encode("gbk"))
        result = parse_multimodal(str(f))
        assert "中文简历" in result["text"]


class TestOcrEngineLazyLoad:
    def test_missing_package_raises_install_hint(self):
        """rapidocr 未安装时抛出带安装指引的错误"""
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.startswith("rapidocr"):
                raise ImportError("no module")
            return real_import(name, *a, **k)

        # 重置单例强制重新加载
        with mock.patch("builtins.__import__", side_effect=fake_import):
            with mock.patch("rag.multimodal._ocr_engine", None):
                from rag.multimodal import _get_ocr_engine
                with pytest.raises(MultimodalDependencyError) as exc:
                    _get_ocr_engine()
                assert "rapidocr" in str(exc.value)


class TestOcrConfidenceFilter:
    def test_low_confidence_lines_dropped(self):
        """置信度低于阈值的误识别行被丢弃，高置信度行保留"""
        from rag import multimodal

        fake_engine = mock.Mock()
        fake_engine.return_value = (
            [
                [[[0, 0], [1, 1]], "高质量文字", 0.95],
                [[[0, 0], [1, 1]], "噪点误识别", 0.20],
                [[[0, 0], [1, 1]], "及格线文字", 0.50],
            ],
            None,
        )
        with mock.patch.object(multimodal, "_get_ocr_engine", return_value=fake_engine):
            text = multimodal._ocr_image("fake.png")
        assert "高质量文字" in text
        assert "及格线文字" in text
        assert "噪点误识别" not in text

    def test_all_low_confidence_returns_empty(self):
        from rag import multimodal

        fake_engine = mock.Mock()
        fake_engine.return_value = (
            [[[[0, 0]], "模糊一", 0.1], [[[0, 0]], "模糊二", 0.2]],
            None,
        )
        with mock.patch.object(multimodal, "_get_ocr_engine", return_value=fake_engine):
            assert multimodal._ocr_image("fake.png") == ""

    def test_missing_score_treated_as_kept(self):
        """无 score 字段的结果（结构缺失）默认保留，不误删"""
        from rag import multimodal

        fake_engine = mock.Mock()
        fake_engine.return_value = ([[[[0, 0]], "无置信度字段"]], None)
        with mock.patch.object(multimodal, "_get_ocr_engine", return_value=fake_engine):
            assert "无置信度字段" in multimodal._ocr_image("fake.png")
