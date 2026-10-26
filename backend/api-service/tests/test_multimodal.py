"""多模态解析单元测试（mock OCR/STT/文档解析，不加载真实模型）"""
import pytest
from types import SimpleNamespace
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
        ("a.webm", "audio"),
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


class TestDecodeToArray:
    """_decode_to_array：自实现的 PyAV 解码（绕开 faster-whisper 与 av>=19 的兼容问题）

    用 stdlib wave 生成真实 PCM wav，走真实 PyAV 解码；不加载 whisper 模型。
    """

    @staticmethod
    def _make_wav(path, seconds=0.5, rate=8000, freq=440.0):
        import math
        import struct
        import wave

        frames = b"".join(
            struct.pack("<h", int(12000 * math.sin(2 * math.pi * freq * i / rate)))
            for i in range(int(seconds * rate))
        )
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(frames)

    def test_decodes_and_resamples_to_16k_float32(self, tmp_path):
        """8kHz int16 wav → 16kHz float32 单声道，长度≈时长×16000，幅值归一化到 [-1,1]"""
        import numpy as np
        from rag.multimodal import _decode_to_array

        wav = tmp_path / "tone.wav"
        self._make_wav(wav, seconds=0.5, rate=8000)
        audio = _decode_to_array(str(wav))

        assert audio.dtype == np.float32
        assert audio.ndim == 1
        # 0.5s @16k = 8000 采样点，重采样允许少量出入
        assert abs(audio.size - 8000) < 400
        # 生成幅值 12000/32768≈0.366：既验证非静音，也验证 int16→float 归一化生效
        assert 0.3 < float(np.abs(audio).max()) <= 1.0

    def test_corrupt_file_raises(self, tmp_path):
        """非音频内容应抛异常（PyAV 解码失败），而不是静默返回空"""
        from rag.multimodal import _decode_to_array

        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a real audio file")
        with pytest.raises(Exception):
            _decode_to_array(str(bad))


class TestTranscribeAudio:
    """_transcribe_audio：中文简体引导词、空音频短路（mock whisper，不加载模型）"""

    @staticmethod
    def _fake_model():
        model = mock.Mock()
        model.transcribe.return_value = (
            [SimpleNamespace(text="转写文本")],
            SimpleNamespace(duration=1.0, language="zh"),
        )
        return model

    @staticmethod
    def _stub(monkeypatch, multimodal, model, audio):
        """隔离 Langfuse 与真实解码，只留被测逻辑"""
        from contextlib import contextmanager
        import numpy as np

        @contextmanager
        def _cm(*a, **k):
            yield None

        monkeypatch.setattr(multimodal, "start_span", _cm)
        monkeypatch.setattr(multimodal, "update_observation", lambda *a, **k: None)
        monkeypatch.setattr(multimodal, "_get_stt_model", lambda: model)
        monkeypatch.setattr(
            multimodal, "_decode_to_array",
            lambda p: np.asarray(audio, dtype=np.float32),
        )

    def test_zh_uses_simplified_prompt(self, monkeypatch):
        """中文转写带简体引导词（否则 whisper 常输出繁体，实测回归）"""
        from rag import multimodal

        model = self._fake_model()
        self._stub(monkeypatch, multimodal, model, [0.0] * 160)

        text = multimodal._transcribe_audio("voice.webm", language="zh")

        assert text == "转写文本"
        kwargs = model.transcribe.call_args.kwargs
        assert kwargs["language"] == "zh"
        assert kwargs["vad_filter"] is True
        assert "普通话" in kwargs["initial_prompt"]

    def test_non_zh_prompt_disabled(self, monkeypatch):
        """非中文不加中文引导词（避免干扰其他语言识别）"""
        from rag import multimodal

        model = self._fake_model()
        self._stub(monkeypatch, multimodal, model, [0.0] * 160)

        multimodal._transcribe_audio("voice.webm", language="en")

        assert model.transcribe.call_args.kwargs["initial_prompt"] is None

    def test_empty_audio_short_circuits(self, monkeypatch):
        """解码为空（静音/无流）直接返回空串，不调用模型"""
        from rag import multimodal

        model = self._fake_model()
        self._stub(monkeypatch, multimodal, model, [])

        assert multimodal._transcribe_audio("silent.webm") == ""
        model.transcribe.assert_not_called()
