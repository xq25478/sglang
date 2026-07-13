#!/usr/bin/env python3
"""Run fixed JD HTTP regressions against one Qwen2.5-VL dummy-weight Server."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from case_progress import ProgressReporter


AGGREGATE_CASE_ID = "jd-server-api-regressions"
VALID_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
)
INVALID_IMAGE_URL = "http://127.0.0.1:1/unreachable.png"


@dataclass(frozen=True, slots=True)
class ServerCase:
    case_id: str
    assertion: str
    timeout_seconds: int
    handler_name: str


SERVER_CASES: tuple[ServerCase, ...] = (
    ServerCase(
        "jd-models-endpoint",
        "OpenAI models endpoint returns the single running model",
        30,
        "models_endpoint",
    ),
    ServerCase(
        "jd-text-non-streaming",
        "Text chat completion succeeds through the non-streaming HTTP path",
        60,
        "text_non_streaming",
    ),
    ServerCase(
        "jd-text-streaming",
        "Text chat completion emits valid streaming chunks and a terminal marker",
        60,
        "text_streaming",
    ),
    ServerCase(
        "jd-image-non-streaming",
        "Inline image chat completion succeeds through the non-streaming HTTP path",
        60,
        "image_non_streaming",
    ),
    ServerCase(
        "jd-image-streaming",
        "Inline image chat completion succeeds through the streaming HTTP path",
        60,
        "image_streaming",
    ),
    ServerCase(
        "jd-broken-base64-non-streaming",
        "Broken base64 image returns HTTP 400 instead of HTTP 500",
        60,
        "broken_base64_non_streaming",
    ),
    ServerCase(
        "jd-invalid-image-url-non-streaming",
        "Unreachable image URL returns HTTP 400 instead of HTTP 500",
        60,
        "invalid_image_url_non_streaming",
    ),
    ServerCase(
        "jd-broken-base64-streaming",
        "Broken base64 streaming request returns a client error instead of HTTP 500",
        60,
        "broken_base64_streaming",
    ),
    ServerCase(
        "jd-invalid-image-url-streaming",
        "Unreachable image URL streaming request returns a client error instead of HTTP 500",
        60,
        "invalid_image_url_streaming",
    ),
    ServerCase(
        "jd-ignore-eos-token-limit",
        "JD ignore_eos and default max tokens produce exactly eight completion tokens",
        60,
        "ignore_eos_token_limit",
    ),
    ServerCase(
        "jd-invalid-thinking-list",
        "List-valued thinking input is normalized without an HTTP 500",
        60,
        "invalid_thinking_list",
    ),
    ServerCase(
        "jd-invalid-thinking-dict",
        "Malformed dict-valued thinking input is normalized without an HTTP 500",
        60,
        "invalid_thinking_dict",
    ),
    ServerCase(
        "jd-invalid-thinking-string",
        "Unknown string-valued thinking input is normalized without an HTTP 500",
        60,
        "invalid_thinking_string",
    ),
    ServerCase(
        "jd-invalid-thinking-int",
        "Integer-valued thinking input is normalized without an HTTP 500",
        60,
        "invalid_thinking_int",
    ),
    ServerCase(
        "jd-tool-choice-none",
        "OpenAI tool_choice none remains accepted by the chat completion endpoint",
        60,
        "tool_choice_none",
    ),
)


@dataclass(frozen=True, slots=True)
class ServerAPIConfig:
    model_path: str
    timeout_seconds: int
    visible_gpu_count: int = 1

    @property
    def server_args(self) -> tuple[str, ...]:
        return (
            "--load-format",
            "dummy",
            "--disable-cuda-graph",
            "--tp-size",
            "1",
            "--mem-fraction-static",
            "0.20",
        )


@dataclass(slots=True)
class RequestContext:
    base_url: str
    model_path: str
    session: Any

    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw_value = env.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def build_config_from_env(env: Mapping[str, str]) -> ServerAPIConfig:
    return ServerAPIConfig(
        model_path=env.get(
            "JD_CI_SERVER_API_MODEL_PATH",
            "/mnt/nas/models/Qwen2.5-VL-7B-Instruct/",
        ),
        timeout_seconds=_positive_int(env, "JD_CI_SERVER_API_TIMEOUT_SEC", 600),
    )


def _response_payload(response: Any) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        payload = {"body": response.text[:1000]}
    return payload if isinstance(payload, dict) else {"payload": payload}


def _require_status(response: Any, expected: int, case_id: str) -> dict[str, Any]:
    payload = _response_payload(response)
    if response.status_code != expected:
        raise AssertionError(
            f"{case_id} returned HTTP {response.status_code}, expected {expected}: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )
    return payload


def _require_success(response: Any, case_id: str) -> dict[str, Any]:
    payload = _require_status(response, 200, case_id)
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AssertionError(f"{case_id} response has no choices: {payload}")
    return payload


def _require_client_error(response: Any, case_id: str, *, exact_400: bool) -> None:
    if exact_400:
        _require_status(response, 400, case_id)
        return
    if not 400 <= response.status_code < 500:
        raise AssertionError(
            f"{case_id} returned HTTP {response.status_code}, expected a 4xx client error: "
            f"{json.dumps(_response_payload(response), ensure_ascii=False)}"
        )


def _base_request(context: RequestContext) -> dict[str, Any]:
    return {
        "model": context.model_path,
        "messages": [{"role": "user", "content": "Reply with one token."}],
        "max_completion_tokens": 1,
        "temperature": 0,
    }


def _image_request(
    context: RequestContext, image_url: str, *, stream: bool
) -> dict[str, Any]:
    return {
        "model": context.model_path,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image briefly."},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "max_completion_tokens": 1,
        "temperature": 0,
        "stream": stream,
    }


def _consume_success_stream(response: Any, case_id: str) -> dict[str, Any]:
    if response.status_code != 200:
        raise AssertionError(
            f"{case_id} returned HTTP {response.status_code}: "
            f"{json.dumps(_response_payload(response), ensure_ascii=False)}"
        )
    chunk_count = 0
    saw_done = False
    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = str(raw_line)
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            saw_done = True
            continue
        payload = json.loads(data)
        if isinstance(payload.get("choices"), list):
            chunk_count += 1
    if chunk_count == 0 or not saw_done:
        raise AssertionError(
            f"{case_id} incomplete stream: chunks={chunk_count}, done={saw_done}"
        )
    return {"status_code": 200, "stream_chunks": chunk_count, "saw_done": True}


def _models_endpoint(context: RequestContext) -> dict[str, Any]:
    response = context.session.get(f"{context.base_url}/v1/models", timeout=30)
    payload = _require_status(response, 200, "jd-models-endpoint")
    models = payload.get("data")
    if not isinstance(models, list) or not models:
        raise AssertionError(f"models endpoint returned no models: {payload}")
    return {"status_code": 200, "model_count": len(models)}


def _text_non_streaming(context: RequestContext) -> dict[str, Any]:
    response = context.session.post(
        context.completions_url, json=_base_request(context), timeout=60
    )
    payload = _require_success(response, "jd-text-non-streaming")
    return {"status_code": 200, "choice_count": len(payload["choices"])}


def _text_streaming(context: RequestContext) -> dict[str, Any]:
    request = {**_base_request(context), "stream": True}
    response = context.session.post(
        context.completions_url, json=request, timeout=60, stream=True
    )
    return _consume_success_stream(response, "jd-text-streaming")


def _image_non_streaming(context: RequestContext) -> dict[str, Any]:
    response = context.session.post(
        context.completions_url,
        json=_image_request(context, VALID_PNG_DATA_URL, stream=False),
        timeout=60,
    )
    payload = _require_success(response, "jd-image-non-streaming")
    return {"status_code": 200, "choice_count": len(payload["choices"])}


def _image_streaming(context: RequestContext) -> dict[str, Any]:
    response = context.session.post(
        context.completions_url,
        json=_image_request(context, VALID_PNG_DATA_URL, stream=True),
        timeout=60,
        stream=True,
    )
    return _consume_success_stream(response, "jd-image-streaming")


def _invalid_image(
    context: RequestContext,
    *,
    case_id: str,
    image_url: str,
    stream: bool,
) -> dict[str, Any]:
    response = context.session.post(
        context.completions_url,
        json=_image_request(context, image_url, stream=stream),
        timeout=60,
        stream=stream,
    )
    _require_client_error(response, case_id, exact_400=not stream)
    return {"status_code": response.status_code}


def _broken_base64_non_streaming(context: RequestContext) -> dict[str, Any]:
    return _invalid_image(
        context,
        case_id="jd-broken-base64-non-streaming",
        image_url="data:image/png;base64,not-valid-base64!",
        stream=False,
    )


def _invalid_image_url_non_streaming(context: RequestContext) -> dict[str, Any]:
    return _invalid_image(
        context,
        case_id="jd-invalid-image-url-non-streaming",
        image_url=INVALID_IMAGE_URL,
        stream=False,
    )


def _broken_base64_streaming(context: RequestContext) -> dict[str, Any]:
    return _invalid_image(
        context,
        case_id="jd-broken-base64-streaming",
        image_url="data:image/png;base64,not-valid-base64!",
        stream=True,
    )


def _invalid_image_url_streaming(context: RequestContext) -> dict[str, Any]:
    return _invalid_image(
        context,
        case_id="jd-invalid-image-url-streaming",
        image_url=INVALID_IMAGE_URL,
        stream=True,
    )


def _ignore_eos_token_limit(context: RequestContext) -> dict[str, Any]:
    request = _base_request(context)
    request.pop("max_completion_tokens")
    request["ignore_eos"] = True
    response = context.session.post(context.completions_url, json=request, timeout=60)
    payload = _require_success(response, "jd-ignore-eos-token-limit")
    usage = payload.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    finish_reason = payload["choices"][0].get("finish_reason")
    if completion_tokens != 8 or finish_reason != "length":
        raise AssertionError(
            "ignore_eos/default-token contract failed: "
            f"completion_tokens={completion_tokens}, finish_reason={finish_reason}"
        )
    return {
        "status_code": 200,
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
    }


def _invalid_thinking(context: RequestContext, case_id: str, value: Any) -> dict[str, Any]:
    request = {**_base_request(context), "thinking": value}
    response = context.session.post(context.completions_url, json=request, timeout=60)
    if response.status_code == 500:
        raise AssertionError(f"{case_id} returned HTTP 500: {response.text[:1000]}")
    if response.status_code != 200:
        raise AssertionError(
            f"{case_id} returned HTTP {response.status_code}: {response.text[:1000]}"
        )
    return {"status_code": response.status_code}


def _invalid_thinking_list(context: RequestContext) -> dict[str, Any]:
    return _invalid_thinking(context, "jd-invalid-thinking-list", ["unsupported"])


def _invalid_thinking_dict(context: RequestContext) -> dict[str, Any]:
    return _invalid_thinking(context, "jd-invalid-thinking-dict", {"invalid": True})


def _invalid_thinking_string(context: RequestContext) -> dict[str, Any]:
    return _invalid_thinking(context, "jd-invalid-thinking-string", "unsupported")


def _invalid_thinking_int(context: RequestContext) -> dict[str, Any]:
    return _invalid_thinking(context, "jd-invalid-thinking-int", 1)


def _tool_choice_none(context: RequestContext) -> dict[str, Any]:
    request = {**_base_request(context), "tool_choice": "none"}
    response = context.session.post(context.completions_url, json=request, timeout=60)
    payload = _require_success(response, "jd-tool-choice-none")
    return {"status_code": 200, "choice_count": len(payload["choices"])}


HANDLERS: dict[str, Callable[[RequestContext], dict[str, Any]]] = {
    "models_endpoint": _models_endpoint,
    "text_non_streaming": _text_non_streaming,
    "text_streaming": _text_streaming,
    "image_non_streaming": _image_non_streaming,
    "image_streaming": _image_streaming,
    "broken_base64_non_streaming": _broken_base64_non_streaming,
    "invalid_image_url_non_streaming": _invalid_image_url_non_streaming,
    "broken_base64_streaming": _broken_base64_streaming,
    "invalid_image_url_streaming": _invalid_image_url_streaming,
    "ignore_eos_token_limit": _ignore_eos_token_limit,
    "invalid_thinking_list": _invalid_thinking_list,
    "invalid_thinking_dict": _invalid_thinking_dict,
    "invalid_thinking_string": _invalid_thinking_string,
    "invalid_thinking_int": _invalid_thinking_int,
    "tool_choice_none": _tool_choice_none,
}


def _blocked_result(case: ServerCase, detail: str) -> dict[str, Any]:
    return {
        "name": case.case_id,
        "case_id": case.case_id,
        "status": "blocked",
        "exit_code": 3,
        "detail": detail,
        "assertion": case.assertion,
        "duration_seconds": 0.0,
        "timeout_seconds": case.timeout_seconds,
        "log_file": "",
    }


def run_api_regressions(config: ServerAPIConfig) -> dict[str, Any]:
    # Heavy imports remain outside --list-cases and CPU configuration tests.
    import requests

    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        DEFAULT_URL_FOR_TEST,
        popen_launch_server,
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    process: subprocess.Popen[Any] | None = None
    server_return_code: int | None = None
    results: list[dict[str, Any]] = []
    startup_error = ""
    server_reporter = ProgressReporter(
        area="Server and API Regression",
        case_id="jd-qwen25-vl-dummy-server",
        index=None,
        total=None,
        assertion="one Qwen2.5-VL dummy-weight Server becomes ready",
        timeout_seconds=min(config.timeout_seconds, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH),
        action="SERVER",
    )
    server_reporter.start("startup")
    try:
        server_env = {
            "JD_ENABLE_IGNORE_EOS": "true",
            "JD_DEFAULT_MAX_TOKENS": "8",
        }
        try:
            process = popen_launch_server(
                config.model_path,
                DEFAULT_URL_FOR_TEST,
                timeout=min(
                    config.timeout_seconds, DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
                ),
                other_args=list(config.server_args),
                env=server_env,
                return_stdout_stderr=(stdout, stderr),
            )
        except Exception as error:  # startup must still produce a complete report
            startup_error = f"server startup failed: {type(error).__name__}: {error}"
            server_reporter.finish("FAIL", exit_code=1, detail=startup_error)
            results.extend(_blocked_result(case, startup_error) for case in SERVER_CASES)
        else:
            server_reporter.finish("PASS", detail="health endpoint is ready")
            context = RequestContext(
                base_url=DEFAULT_URL_FOR_TEST,
                model_path=config.model_path,
                session=requests.Session(),
            )
            server_dead_detail = ""
            for index, case in enumerate(SERVER_CASES, start=1):
                if process.poll() is not None:
                    server_dead_detail = (
                        f"dummy-weight server exited with code {process.returncode}"
                    )
                if server_dead_detail:
                    print(
                        f"[JD CI][Server and API Regression][CASE {index}/{len(SERVER_CASES)}]"
                        f"[BLOCKED] id={case.case_id} duration=0.0s exit_code=3 "
                        f"detail={server_dead_detail}",
                        flush=True,
                    )
                    results.append(_blocked_result(case, server_dead_detail))
                    continue

                reporter = ProgressReporter(
                    area="Server and API Regression",
                    case_id=case.case_id,
                    index=index,
                    total=len(SERVER_CASES),
                    assertion=case.assertion,
                    timeout_seconds=case.timeout_seconds,
                )
                reporter.start("http-request")
                started_at = time.monotonic()
                try:
                    evidence = HANDLERS[case.handler_name](context)
                    if process.poll() is not None:
                        raise AssertionError(
                            f"dummy-weight server exited with code {process.returncode}"
                        )
                except Exception as error:
                    detail = f"{type(error).__name__}: {error}"
                    reporter.finish("FAIL", exit_code=1, detail=detail)
                    results.append(
                        {
                            "name": case.case_id,
                            "case_id": case.case_id,
                            "status": "failed",
                            "exit_code": 1,
                            "detail": detail,
                            "assertion": case.assertion,
                            "duration_seconds": round(
                                time.monotonic() - started_at, 3
                            ),
                            "timeout_seconds": case.timeout_seconds,
                            "log_file": "",
                        }
                    )
                    if process.poll() is not None:
                        server_dead_detail = detail
                else:
                    reporter.finish("PASS")
                    results.append(
                        {
                            "name": case.case_id,
                            "case_id": case.case_id,
                            "status": "passed",
                            "exit_code": 0,
                            "detail": "",
                            "assertion": case.assertion,
                            "duration_seconds": round(
                                time.monotonic() - started_at, 3
                            ),
                            "timeout_seconds": case.timeout_seconds,
                            "log_file": "",
                            "evidence": evidence,
                        }
                    )
            server_return_code = process.poll()
    finally:
        if process is not None:
            kill_process_tree(process.pid)

    log_text = stdout.getvalue() + stderr.getvalue()
    status = (
        "passed"
        if len(results) == len(SERVER_CASES)
        and all(result["status"] == "passed" for result in results)
        else "failed"
    )
    return {
        "status": status,
        "case_id": AGGREGATE_CASE_ID,
        "cases": results,
        "server": {
            "model_path": config.model_path,
            "server_args": list(config.server_args),
            "startup_error": startup_error,
        },
        "config": asdict(config),
        "server_return_code": server_return_code,
        "log_tail": log_text[-4000:],
    }


def run_case(case_id: str, config: ServerAPIConfig) -> dict[str, Any]:
    if case_id == AGGREGATE_CASE_ID:
        return run_api_regressions(config)
    raise ValueError(f"unknown JD Server and API Regression case: {case_id}")


def write_result(path: str | Path, result: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--case", choices=(AGGREGATE_CASE_ID,))
    operation.add_argument("--list-cases", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_cases:
        write_result(args.output, [asdict(case) for case in SERVER_CASES])
        return 0

    config = build_config_from_env(os.environ)
    result = run_case(args.case, config)
    write_result(args.output, result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
