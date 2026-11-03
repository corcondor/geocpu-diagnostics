"""Run the packaged source's pinned static preflight without loading checkpoints."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zipfile

from package_sources import OUTPUT, REPOSITORY


def main() -> None:
    reports = (REPOSITORY / "reports" / "v44fix").resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="geocpu_head_preflight_", dir=reports) as name:
        target = Path(name).resolve(strict=True)
        if not target.is_relative_to(reports):
            raise RuntimeError("Temporary extraction escaped the intended reports directory")
        with zipfile.ZipFile(OUTPUT) as archive:
            for member_name in archive.namelist():
                member = PurePosixPath(member_name)
                if member.is_absolute() or ".." in member.parts:
                    raise RuntimeError("Unsafe source member")
                path = target.joinpath(*member.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.read(member_name))
        environment = dict(os.environ, GEOCPU_DRIVE_ROOT="G:\\マイドライブ")
        subprocess.run([sys.executable, str(target / "runner.py"), "--preflight",
                        "--assets-root", "G:\\マイドライブ",
                        "--output-dir", str(target / "unused")],
                       env=environment, check=True)


if __name__ == "__main__":
    main()
