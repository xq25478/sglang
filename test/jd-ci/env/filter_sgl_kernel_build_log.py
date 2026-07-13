#!/usr/bin/env python3
"""Filter known noisy sgl-kernel build diagnostics from live CI logs."""

from __future__ import annotations

import sys
from collections import Counter


NOISY_PTXAS_INFO_PREFIX = "ptxas info    : (C75"
NOISY_NVCC_STD_REDEFINITION = (
    "nvcc warning : incompatible redefinition for option 'std', "
    "the last value of this option was used"
)


def main() -> int:
    suppressed: Counter[str] = Counter()
    for line in sys.stdin:
        if NOISY_PTXAS_INFO_PREFIX in line:
            code = line.split("(", 1)[1].split(")", 1)[0]
            suppressed[f"ptxas_{code}"] += 1
            continue
        if NOISY_NVCC_STD_REDEFINITION in line:
            suppressed["nvcc_std_redefinition"] += 1
            continue
        sys.stdout.write(line)
        sys.stdout.flush()

    if suppressed:
        total = sum(suppressed.values())
        details = ", ".join(f"{key}={count}" for key, count in sorted(suppressed.items()))
        print(
            f"[SGLang CI] Suppressed {total} noisy sgl-kernel build diagnostic lines: {details}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
