import pytest
from agents.providers.local.token_processor import TokenProcessor
from agents.providers.local.utils import filter_think_content


class TestTokenProcessor:
    """Token 处理器测试套件"""

    @pytest.fixture
    def processor(self):
        return TokenProcessor(parse_think=True)

    @pytest.fixture
    def processor_no_think(self):
        return TokenProcessor(parse_think=False)

    # ---- 普通 Token 处理 ----

    def test_normal_token_output(self, processor):
        """普通 token 应输出为 type=token"""
        outputs = processor.process_token("你好")
        assert len(outputs) == 1
        assert outputs[0]["type"] == "token"
        assert outputs[0]["content"] == "你好"

    def test_normal_token_accumulates_full_text(self, processor):
        """普通 token 应累积到 full_text"""
        processor.process_token("Hello")
        processor.process_token(" World")
        assert processor.full_text == "Hello World"

    def test_multiple_tokens_sequentially(self, processor):
        """连续处理多个 token 应各自输出"""
        out1 = processor.process_token("第")
        out2 = processor.process_token("一")
        out3 = processor.process_token("个")
        assert out1[0]["content"] == "第"
        assert out2[0]["content"] == "一"
        assert out3[0]["content"] == "个"
        assert processor.full_text == "第一个"

    # ---- Think 标签解析（parse_think=True）----

    def test_think_start_tag_detected(self, processor):
        """检测到 <think> 开始标签应进入思考状态"""
        outputs = processor.process_token("开始")
        assert processor.in_think_tag is False
        assert outputs[0]["type"] == "token"

        outputs = processor.process_token("<think>思考内容")
        assert processor.in_think_tag is True
        assert processor.think_buffer == "思考内容"

    def test_think_end_tag_emits_thought(self, processor):
        """检测到 </think> 结束标签应输出思考内容"""
        processor.in_think_tag = True
        processor.think_buffer = "这是思考过程"

        outputs = processor.process_token("</think>结束正文")
        assert processor.in_think_tag is False
        thought_outputs = [o for o in outputs if o["type"] == "thought"]
        assert len(thought_outputs) == 1
        assert "思考过程" in thought_outputs[0]["content"]

    def test_think_end_tag_continues_normal(self, processor):
        """</think> 标签后的内容应作为正常 token 输出"""
        processor.in_think_tag = True
        processor.think_buffer = "思考"

        outputs = processor.process_token("</think>正常文本")
        token_outputs = [o for o in outputs if o["type"] == "token"]
        assert len(token_outputs) >= 1
        assert "正常文本" in token_outputs[-1]["content"]

    def test_full_think_cycle(self, processor):
        """完整的 <think> 标签周期：前缀 → 思考 → 后缀"""
        step1 = processor.process_token("前缀")
        assert step1[0]["type"] == "token"
        assert step1[0]["content"] == "前缀"

        step2 = processor.process_token("<think>内部思考")
        assert processor.in_think_tag is True

        step3 = processor.process_token("</think>后续内容")
        thought_outs = [o for o in step3 if o["type"] == "thought"]
        token_outs = [o for o in step3 if o["type"] == "token"]
        assert len(thought_outs) == 1
        assert len(token_outs) >= 1
        assert processor.full_text == "前缀后续内容"

    # ---- Think 过滤模式（parse_think=False）----

    def test_no_think_mode_filters_content(self, processor_no_think):
        """不解析模式下 <think> 内容应被过滤"""
        outs = processor_no_think.process_token("前")
        assert outs[0]["content"] == "前"

        outs = processor_no_think.process_token("<think>隐藏内容")
        assert processor_no_think.in_think_tag is True
        assert len(outs) == 0

        outs = processor_no_think.process_token("</think>可见内容")
        assert processor_no_think.in_think_tag is False
        assert any(o["content"] == "可见内容" for o in outs)

    def test_no_think_mode_never_emits_thought_type(self, processor_no_think):
        """不解析模式下不应输出 type=thought"""
        tokens = [
            "前",
            "<think>隐藏",
            "</think>后",
            "更多"
        ]
        all_outputs = []
        for t in tokens:
            all_outputs.extend(processor_no_think.process_token(t))

        thought_types = [o for o in all_outputs if o["type"] == "thought"]
        assert len(thought_types) == 0

    # ---- 边界情况 ----

    def test_empty_token(self, processor):
        """空 token 不应产生输出或崩溃"""
        outputs = processor.process_token("")
        assert len(outputs) == 1
        assert outputs[0]["content"] == ""

    def test_reset_clears_state(self, processor):
        """reset 方法应清除所有状态"""
        processor.process_token("<think>思考中")
        processor.process_token("</think>一些文本")
        processor.reset()
        assert processor.in_think_tag is False
        assert processor.think_buffer == ""
        assert processor.full_text == ""

    def test_consecutive_think_tags(self, processor):
        """连续的 <think> 标签对应正确处理"""
        processor.process_token("第一段")
        processor.process_token("<think>第一次思考")
        processor.process_token("</think>中间文本")
        processor.process_token("<think>第二次思考")
        processor.process_token("</think>最后文本")

        assert "第一段" in processor.full_text
        assert "中间文本" in processor.full_text
        assert "最后文本" in processor.full_text
        assert processor.in_think_tag is False

    def test_partial_think_start_in_token(self, processor):
        """token 中包含部分 <think> 标签（标签跨 token）"""
        outputs = processor.process_token("前半<think>")
        assert processor.in_think_tag is True
        assert processor.think_buffer == ""

        outputs = processor.process_token("后半")
        assert processor.think_buffer == "后半"


class TestFilterThinkContent:
    """思考内容过滤工具函数测试"""

    def test_filters_instruction_patterns(self):
        """应过滤 Prompt 指令重复"""
        result = filter_think_content("按以下3个部分输出\n这是思考\n每部分用【标题】开头")
        assert "按以下3个部分输出" not in result
        assert "每部分用【标题】开头" not in result

    def test_preserves_chinese_text(self):
        """应保留中文内容"""
        result = filter_think_content("这是中文思考过程")
        assert "中文思考过程" in result

    def test_handles_empty_string(self):
        """空字符串不应崩溃"""
        result = filter_think_content("")
        assert result == ""

    def test_handles_long_text(self):
        """长文本应正常处理"""
        long_text = "思考" * 1000
        result = filter_think_content(long_text)
        assert len(result) > 0

    def test_filters_resume_score_pattern(self):
        """应过滤简历评分相关指令"""
        result = filter_think_content("【简历评分】只给分数\n这是思考")
        assert "【简历评分】只给分数" not in result
