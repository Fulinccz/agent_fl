"""上传路由 MAS 化测试

上传文件解析后进入主从多智能体流水线（与粘贴链路同一套事件契约）。
依赖全部 mock：不读真实文件解析、不起模型、不起 HTTP 服务。
"""
import io
import json
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi import UploadFile

from api.routes import upload as upload_route
from api.routes import deps
from agents.mas.state import INTENT_RESUME_OPTIMIZE
from rag.multimodal import MultimodalDependencyError
from api.routes.upload import _cleanup_file


def _txt_upload(content: bytes = "张三\n三年Java开发经历".encode("utf-8"), filename="resume.txt"):
    return UploadFile(filename=filename, file=io.BytesIO(content))


def _fake_request():
    req = MagicMock()
    # dict 而非 MagicMock：endpoint 会 int(headers.get("content-length"))
    req.headers = {"content-length": None}
    # endpoint 在流式循环中 await request.is_disconnected()
    req.is_disconnected = AsyncMock(return_value=False)
    return req


def _allow_rate_limit(monkeypatch):
    monkeypatch.setattr(
        upload_route, "check_rate_limit",
        lambda request, path: (True, 0, 10)
    )


def _orch_recording(events=None):
    """记录收到的 context，并回放指定事件"""
    orch = MagicMock()
    captured = {}

    def _stream(user_input, context, preset_intent=None):
        captured.update(context)
        yield {"type": "score", "data": {"overall_score": {"score": 75}}}
        yield {"type": "polished", "data": {"optimized_resume": "优化后简历正文" * 5}}
        yield {"type": "complete", "data": {"elapsed_seconds": 1.2}}

    orch.run_stream = MagicMock(side_effect=_stream)
    orch._captured = captured
    return orch


# ---------- 临时文件清理 ----------

class TestCleanupFile:
    def test_removes_existing_file(self, tmp_path):
        f = tmp_path / "x.tmp"
        f.write_text("data", encoding="utf-8")
        _cleanup_file(str(f))
        assert not f.exists()

    def test_none_path_is_noop(self):
        _cleanup_file(None)

    def test_missing_path_is_noop(self, tmp_path):
        _cleanup_file(str(tmp_path / "already_gone.tmp"))


# ---------- MAS 流水线接入 ----------

class TestUploadMasStream:
    async def test_parsed_resume_enters_mas_with_full_context(
        self, monkeypatch, tmp_path
    ):
        _allow_rate_limit(monkeypatch)
        monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(
            upload_route, "parse_multimodal",
            lambda path: {"text": "张三\n三年Java开发经历，主导订单系统", "modality": "document"}
        )
        orch = _orch_recording()
        monkeypatch.setattr(deps, "get_master_orchestrator", lambda: orch)

        response = await upload_route.upload_file_stream(
            request=_fake_request(),
            file=_txt_upload(),
            query="突出量化成果",
            jd="高级Java工程师",
            provider="deepseek",
            model=None,
        )

        assert response.status_code == 200
        events = [json.loads(line) async for line in response.body_iterator]
        assert events[0]["type"] == "score"
        assert events[-1]["type"] == "complete"

        # MAS 以预置简历意图调用，user_input 为空、简历走 context
        args = orch.run_stream.call_args.args
        assert args[0] == ""
        ctx = args[1]
        assert ctx["resume"].startswith("张三")
        assert ctx["jd"] == "高级Java工程师"
        assert ctx["extra_instruction"] == "突出量化成果"
        assert ctx["provider"] == "deepseek"
        # 预置意图
        assert args[2].name == INTENT_RESUME_OPTIMIZE

        # 临时文件解析后即清理
        assert not list(tmp_path.glob("*.txt"))

    async def test_empty_query_and_jd_omitted_from_context(
        self, monkeypatch, tmp_path
    ):
        _allow_rate_limit(monkeypatch)
        monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(
            upload_route, "parse_multimodal",
            lambda path: {"text": "简历正文内容" * 10, "modality": "document"}
        )
        orch = _orch_recording()
        monkeypatch.setattr(deps, "get_master_orchestrator", lambda: orch)

        response = await upload_route.upload_file_stream(
            request=_fake_request(),
            file=_txt_upload(),
            query="  ",
            jd=None,
            provider="local",
            model=None,
        )
        assert response.status_code == 200
        # 消费完
        async for _ in response.body_iterator:
            pass

        ctx = orch.run_stream.call_args.args[1]
        assert "jd" not in ctx
        assert "extra_instruction" not in ctx

    async def test_unsupported_extension_rejected(self, monkeypatch):
        _allow_rate_limit(monkeypatch)
        response = await upload_route.upload_file_stream(
            request=_fake_request(),
            file=_txt_upload(filename="evil.exe"),
            query="",
            jd=None,
            provider="local",
            model=None,
        )
        assert response.status_code == 400

    async def test_no_text_recognized_returns_422(self, monkeypatch, tmp_path):
        _allow_rate_limit(monkeypatch)
        monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path))
        monkeypatch.setattr(
            upload_route, "parse_multimodal",
            lambda path: {"text": "  ", "modality": "image"}
        )
        response = await upload_route.upload_file_stream(
            request=_fake_request(),
            file=UploadFile(filename="a.png", file=io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"0" * 64)),
            query="",
            jd=None,
            provider="local",
            model=None,
        )
        assert response.status_code == 422
        # 失败路径也清理了临时文件
        assert not list(tmp_path.glob("*.png"))

    async def test_missing_optional_dependency_returns_400(self, monkeypatch, tmp_path):
        _allow_rate_limit(monkeypatch)
        monkeypatch.setattr(upload_route, "UPLOAD_DIR", str(tmp_path))

        def _raise(path):
            raise MultimodalDependencyError("缺少 tesseract")

        monkeypatch.setattr(upload_route, "parse_multimodal", _raise)
        response = await upload_route.upload_file_stream(
            request=_fake_request(),
            file=_txt_upload(),
            query="",
            jd=None,
            provider="local",
            model=None,
        )
        assert response.status_code == 400
        assert "tesseract" in response.body.decode("utf-8")
