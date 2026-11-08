"""Apply Linux process limits before loading the Python TIR worker.

These limits bound runaway resource use. They do not isolate files, network,
credentials, or other processes and are not a security sandbox.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import runpy
import sys


MAX_ADDRESS_SPACE_BYTES = 4 * 1024**3
MAX_CPU_SECONDS = 60


def limit_and_run(worker_path: str | Path, *, platform: str | None = None,
                  resource_module=None, run_path=None) -> None:
    if (platform or sys.platform) != "linux":
        raise RuntimeError("Python TIR worker requires Linux resource limits")
    limits = resource_module or importlib.import_module("resource")
    limits.setrlimit(limits.RLIMIT_AS, (MAX_ADDRESS_SPACE_BYTES, MAX_ADDRESS_SPACE_BYTES))
    limits.setrlimit(limits.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))
    limits.setrlimit(limits.RLIMIT_CORE, (0, 0))
    (run_path or runpy.run_path)(str(worker_path), run_name="__main__")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("resource_limited_worker_v44r1.py requires exactly one worker path")
    limit_and_run(sys.argv[1])
