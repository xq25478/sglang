#!/usr/bin/env python3
"""Observable command runner used by the fixed JD regression inventories."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import IO, Sequence


class ProgressReporter:
    """Print stable case lifecycle messages and a periodic running heartbeat."""

    def __init__(
        self,
        area: str,
        case_id: str,
        index: int | None,
        total: int | None,
        assertion: str,
        timeout_seconds: float,
        action: str = "CASE",
        heartbeat_seconds: float = 5.0,
        stream: IO[str] | None = None,
    ) -> None:
        if (index is None) != (total is None):
            raise ValueError("index and total must both be set or both be None")
        if index is not None and (index < 1 or total is None or total < index):
            raise ValueError("case counter must satisfy 1 <= index <= total")
        if heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be positive")

        self.area = area
        self.case_id = case_id
        self.index = index
        self.total = total
        self.assertion = assertion
        self.timeout_seconds = timeout_seconds
        self.action = action
        self.heartbeat_seconds = heartbeat_seconds
        self.stream = stream or sys.stdout

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._phase = "pending"
        self._finished = False

    @property
    def prefix(self) -> str:
        counter = "" if self.index is None else f" {self.index}/{self.total}"
        return f"[JD CI][{self.area}][{self.action}{counter}]"

    @property
    def elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def _emit(self, message: str) -> None:
        with self._lock:
            self.stream.write(f"{message}\n")
            self.stream.flush()

    def write_child_output(self, text: str) -> None:
        """Copy child output without interleaving it with heartbeat lines."""
        with self._lock:
            self.stream.write(text)
            self.stream.flush()

    def start(self, phase: str) -> None:
        if self._started_at is not None:
            raise RuntimeError("reporter already started")
        self._phase = phase
        self._started_at = time.monotonic()
        self._emit(
            f"{self.prefix}[START] id={self.case_id} "
            f"assertion={self.assertion} phase={phase} "
            f"timeout={_format_seconds(self.timeout_seconds)}"
        )
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"jd-ci-progress-{self.case_id}",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                phase = self._phase
                elapsed = self.elapsed_seconds
                self.stream.write(
                    f"{self.prefix}[RUNNING] id={self.case_id} phase={phase} "
                    f"elapsed={elapsed:.1f}s "
                    f"timeout={_format_seconds(self.timeout_seconds)}\n"
                )
                self.stream.flush()

    def finish(
        self,
        status: str,
        exit_code: int = 0,
        detail: str = "",
        log_file: str = "",
    ) -> None:
        if self._finished:
            return
        if self._started_at is None:
            raise RuntimeError("reporter has not started")
        self._finished = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

        fields = [
            f"id={self.case_id}",
            f"duration={self.elapsed_seconds:.1f}s",
            f"exit_code={exit_code}",
        ]
        if detail:
            fields.append(f"detail={_single_line(detail)}")
        if log_file:
            fields.append(f"log_file={log_file}")
        self._emit(f"{self.prefix}[{status.upper()}] {' '.join(fields)}")


def _single_line(value: str) -> str:
    return " ".join(value.replace("\t", " ").splitlines())


def _format_seconds(seconds: float) -> str:
    value = float(seconds)
    if value.is_integer():
        return f"{int(value)}s"
    return f"{value:g}s"


def _terminate_process_group(process: subprocess.Popen[str], signal_number: int) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal_number)
    except ProcessLookupError:
        pass


def run_command(
    command: Sequence[str],
    *,
    reporter: ProgressReporter,
    log_file: str | Path,
    timeout_seconds: float,
    kill_after_seconds: float,
) -> int:
    """Run one command, streaming combined output and enforcing a deadline."""
    if not command:
        raise ValueError("command must not be empty")

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    forwarded_signal: int | None = None

    with log_path.open("w", encoding="utf-8") as log_stream:
        try:
            process = subprocess.Popen(
                list(command),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            message = f"failed to start command: {error}\n"
            reporter.write_child_output(message)
            log_stream.write(message)
            log_stream.flush()
            return 127

        def copy_output() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                reporter.write_child_output(line)
                log_stream.write(line)
                log_stream.flush()

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()

        previous_handlers: dict[int, signal.Handlers] = {}

        def forward_signal(signal_number: int, _frame: object) -> None:
            nonlocal forwarded_signal
            forwarded_signal = signal_number
            _terminate_process_group(process, signal_number)

        if threading.current_thread() is threading.main_thread():
            for signal_number in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signal_number] = signal.getsignal(signal_number)
                signal.signal(signal_number, forward_signal)

        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        try:
            while process.poll() is None:
                if forwarded_signal is not None:
                    break
                if time.monotonic() >= deadline:
                    timed_out = True
                    reporter.set_phase("terminating-timeout")
                    _terminate_process_group(process, signal.SIGTERM)
                    break
                time.sleep(min(0.05, max(0.001, deadline - time.monotonic())))

            if timed_out or forwarded_signal is not None:
                try:
                    process.wait(timeout=kill_after_seconds)
                except subprocess.TimeoutExpired:
                    reporter.set_phase("killing-process-group")
                    _terminate_process_group(process, signal.SIGKILL)
                    process.wait()
            else:
                process.wait()
        finally:
            for signal_number, previous_handler in previous_handlers.items():
                signal.signal(signal_number, previous_handler)
            reader.join(timeout=max(1.0, kill_after_seconds))
            if process.stdout is not None:
                process.stdout.close()

    if timed_out:
        return 124
    if forwarded_signal == signal.SIGINT:
        return 130
    if forwarded_signal == signal.SIGTERM:
        return 143
    return int(process.returncode or 0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--area", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--assertion", required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--kill-after-seconds", type=float, default=15.0)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--command-json", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    command = json.loads(args.command_json)
    if not isinstance(command, list) or not all(
        isinstance(argument, str) for argument in command
    ):
        raise SystemExit("--command-json must encode a list of strings")

    reporter = ProgressReporter(
        area=args.area,
        case_id=args.case_id,
        index=args.index,
        total=args.total,
        assertion=args.assertion,
        timeout_seconds=args.timeout_seconds,
    )
    reporter.start("command")
    exit_code = run_command(
        command,
        reporter=reporter,
        log_file=args.log_file,
        timeout_seconds=args.timeout_seconds,
        kill_after_seconds=args.kill_after_seconds,
    )
    reporter.finish(
        "PASS" if exit_code == 0 else "FAIL",
        exit_code=exit_code,
        detail="" if exit_code == 0 else f"command exited with code {exit_code}",
        log_file=args.log_file,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
