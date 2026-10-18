"""services.json_utils 单测：覆盖平衡括号扫描的边界情况"""
import json

from services.json_utils import extract_json_object


def test_simple_object():
    assert extract_json_object('{"a": 1}') == '{"a": 1}'


def test_with_prefix_text():
    assert extract_json_object('前缀文本 {"a": 1} 后缀') == '{"a": 1}'


def test_nested_object():
    raw = '{"outer": {"inner": {"deep": 1}}}'
    assert extract_json_object(raw) == raw
    assert json.loads(extract_json_object(raw)) == {"outer": {"inner": {"deep": 1}}}


def test_string_contains_braces():
    """字符串值内含花括号不应破坏括号平衡"""
    raw = '{"thought": "请补充 {项目名} 和 {时间}", "action": "answer"}'
    extracted = extract_json_object(raw)
    assert extracted == raw
    assert json.loads(extracted)["thought"] == "请补充 {项目名} 和 {时间}"


def test_escaped_quotes_in_string():
    raw = '{"a": "say \\"hi\\" {x}"}'
    extracted = extract_json_object(raw)
    assert extracted is not None
    assert json.loads(extracted)["a"] == 'say "hi" {x}'


def test_multiple_objects_takes_first():
    raw = '{"a": 1} 中间 {"b": 2}'
    assert extract_json_object(raw) == '{"a": 1}'


def test_array_in_value():
    raw = '{"steps": [{"agent": "a"}, {"agent": "b"}]}'
    extracted = extract_json_object(raw)
    assert extracted == raw
    assert len(json.loads(extracted)["steps"]) == 2


def test_no_braces_returns_none():
    assert extract_json_object("纯文本无花括号") is None


def test_empty_and_none():
    assert extract_json_object("") is None
    assert extract_json_object(None) is None  # type: ignore[arg-type]


def test_unbalanced_returns_none():
    """只有左花括号无闭合 → 返回 None（不截断）"""
    assert extract_json_object('{"a": 1') is None
