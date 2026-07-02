import json
import logging
import re

from partial_json_parser.core.options import Allow

from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.utils import _partial_json_loads

logger = logging.getLogger(__name__)


class DeepSeekV4Detector(DeepSeekV32Detector):
    """
    Detector for DeepSeek V4 model function call format.

    The DeepSeek V4 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Examples:
    ```
    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>

    <｜DSML｜tool_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V4 format specification
    """

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜tool_calls>"
        self.eot_token = "</｜DSML｜tool_calls>"
        self.function_calls_regex = (
            r"<｜DSML｜tool_calls>(.*?)(?:</(?:｜DSML｜)?tool_calls>|$)"
        )
        self.invoke_regex = (
            r'<(?:｜DSML｜)?invoke\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)"
            r"|>(?P<body>.*?)(?P<end>(?:</(?:｜DSML｜)?invoke>|$)))"
        )
        self.parameter_regex = (
            r'<(?:｜DSML｜)?parameter\s+name="([^"]+)"\s+'
            r'string="(true|false)"\s*>(.*?)</(?:｜DSML｜)?parameter>'
        )
        self.partial_parameter_regex = (
            r'<(?:｜DSML｜)?parameter\s+name="([^"]+)"\s+'
            r'string="(true|false)"\s*>(.*)$'
        )

    def get_structural_tag_name(self) -> str:
        return "deepseek_v4"

    @staticmethod
    def _decode_parameter_value(param_type: str, param_value: str):
        param_value = param_value.strip()
        if param_type == "true":
            return param_value

        try:
            return json.loads(param_value)
        except (json.JSONDecodeError, ValueError):
            try:
                return _partial_json_loads(param_value, Allow.ALL)[0]
            except (json.JSONDecodeError, ValueError, TypeError):
                return param_value

    def _parse_parameters_from_xml(
        self, invoke_content: str, allow_partial: bool = False
    ) -> str:
        """
        Parse DeepSeek-V4 DSML parameters.

        The prompt asks the model to emit fully-qualified DSML tags, but in
        practice V4 can sometimes produce a mixed form such as
        `<｜DSML｜tool_calls><invoke ...><parameter ...>...`. Be lenient here
        so the OpenAI response does not leak raw DSML markers.
        """
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            if invoke_content_stripped.endswith("}"):
                return invoke_content_stripped

        parameters = {}
        param_matches = list(
            re.finditer(self.parameter_regex, invoke_content, re.DOTALL)
        )

        last_match_end = 0
        for match in param_matches:
            param_name = match.group(1)
            param_type = match.group(2)
            param_value = match.group(3)
            last_match_end = match.end()
            parameters[param_name] = self._decode_parameter_value(
                param_type, param_value
            )

        # Recovery for the observed malformed shape:
        # `<｜DSML｜parameter name="name" string="weekly-report">`
        # where the model placed the string value in the `string` attribute.
        attr_value_regex = (
            r'<(?:｜DSML｜)?parameter\s+name="([^"]+)"\s+'
            r'string="(?!true"|false")([^"]+)"\s*>\s*(?=\n|<|$)'
        )
        for match in re.finditer(attr_value_regex, invoke_content, re.DOTALL):
            param_name = match.group(1)
            if param_name not in parameters:
                parameters[param_name] = match.group(2).strip()

        if allow_partial:
            remaining_content = invoke_content[last_match_end:]
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)

            partial_match = re.search(
                self.partial_parameter_regex, remaining_content, re.DOTALL
            )
            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                parameters[param_name] = self._decode_parameter_value(
                    partial_match.group(2), param_value
                )

        return json.dumps(parameters, ensure_ascii=False)
