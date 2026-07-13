import json
import os
import unittest
from unittest.mock import patch

from sglang.srt.entrypoints.openai.encoding_dsv4 import encode_arguments_to_dsml
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, Function, Tool
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.usage_processor import UsageProcessor
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
from sglang.srt.parser.reasoning_parser import ReasoningParser


def make_request(**kwargs):
    return ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        **kwargs,
    )


class TestJDOpenAICompatibility(unittest.TestCase):
    def test_thinking_forms_are_normalized_into_template_kwargs(self):
        cases = [
            (True, True),
            ("enabled", True),
            ({"type": "disabled"}, False),
        ]
        for thinking, expected in cases:
            with self.subTest(thinking=thinking):
                request = make_request(thinking=thinking)
                self.assertEqual(
                    request.chat_template_kwargs["enable_thinking"], expected
                )

    def test_invalid_thinking_list_is_ignored(self):
        request = make_request(thinking=["unsupported"])

        self.assertIsNone(request.thinking)
        self.assertIsNone(request.chat_template_kwargs)

    def test_invalid_thinking_dict_without_type_is_ignored(self):
        request = make_request(thinking={"invalid": True})

        self.assertIsNone(request.thinking)
        self.assertIsNone(request.chat_template_kwargs)

    def test_invalid_thinking_unknown_string_is_ignored(self):
        request = make_request(thinking="not-a-supported-mode")

        self.assertIsNone(request.thinking)
        self.assertIsNone(request.chat_template_kwargs)

    def test_invalid_thinking_integer_is_ignored(self):
        request = make_request(thinking=1)

        self.assertIsNone(request.thinking)
        self.assertIsNone(request.chat_template_kwargs)

    def test_deepseek_v4_reasoning_switch(self):
        serving = object.__new__(OpenAIServingChat)
        serving.reasoning_parser = "deepseek-v4"

        self.assertTrue(
            serving._get_reasoning_from_request(make_request(thinking="enabled"))
        )
        self.assertFalse(
            serving._get_reasoning_from_request(make_request(thinking="disabled"))
        )
        self.assertFalse(serving._get_reasoning_from_request(make_request()))

    def test_zero_max_completion_tokens_is_not_replaced_by_max_tokens(self):
        request = make_request(max_completion_tokens=0, max_tokens=128)

        params = request.to_sampling_params(stop=[], model_generation_config={})

        self.assertEqual(params["max_new_tokens"], 0)

    def test_ignore_eos_requires_jd_gate(self):
        request = make_request(ignore_eos=True)
        with patch.dict(os.environ, {"JD_ENABLE_IGNORE_EOS": "false"}, clear=False):
            self.assertFalse(request._verify_ignore_eos())
        with patch.dict(os.environ, {"JD_ENABLE_IGNORE_EOS": "true"}, clear=False):
            self.assertTrue(request._verify_ignore_eos())

    def test_dsv4_scalar_or_malformed_arguments_are_encoded_not_rejected(self):
        for raw in ("not-json", "[1, 2]", 7):
            with self.subTest(raw=raw):
                encoded = encode_arguments_to_dsml({"arguments": raw})
                self.assertIn('name="arguments"', encoded)


class TestJDGLMReasoningCompatibility(unittest.TestCase):
    def test_glm45_non_stream_tool_interruption(self):
        parser = ReasoningParser("glm45")

        reasoning, content = parser.parse_non_stream(
            "<think>reasoning<tool_call>tool payload"
        )

        self.assertEqual(reasoning, "reasoning")
        self.assertEqual(content, "<tool_call>tool payload")

    def test_glm45_stream_tool_interruption(self):
        parser = ReasoningParser("glm45")
        reasoning_content = ""
        normal_content = ""

        for chunk in ["<think>", "reasoning", "<tool_call>", "tool args"]:
            reasoning, content = parser.parse_stream_chunk(chunk)
            reasoning_content += reasoning or ""
            normal_content += content or ""

        self.assertEqual(reasoning_content, "reasoning")
        self.assertEqual(normal_content, "<tool_call>tool args")

    def test_reasoning_token_usage(self):
        usage = UsageProcessor.calculate_response_usage(
            [
                {
                    "meta_info": {
                        "prompt_tokens": 3,
                        "completion_tokens": 5,
                        "reasoning_tokens": 2,
                    }
                }
            ]
        )

        self.assertEqual(usage.prompt_tokens, 3)
        self.assertEqual(usage.completion_tokens, 5)
        self.assertEqual(usage.reasoning_tokens, 2)


class TestJDKimiK2Compatibility(unittest.TestCase):
    def test_quoted_argument_object_is_parsed(self):
        detector = KimiK2Detector()
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="ReadFile",
                    description="read",
                    parameters={"type": "object"},
                ),
            )
        ]
        text = (
            "<|tool_calls_section_begin|>"
            "<|tool_call_begin|>functions.ReadFile:0"
            '<|tool_call_argument_begin|>"{\\"path\\": \\"/tmp/a\\"}"'
            "<|tool_call_end|><|tool_calls_section_end|>"
        )

        result = detector.detect_and_parse(text, tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "ReadFile")
        self.assertEqual(json.loads(result.calls[0].parameters)["path"], "/tmp/a")


if __name__ == "__main__":
    unittest.main(verbosity=2)
