"""Read-only, source-pinned v43 parent diagnostic on a private free Kaggle GPU."""
from __future__ import annotations

import base64
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import zlib

EMBEDDED = __EMBEDDED__
V43_ZIP_SHA = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
PARENT_SHA = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"
ROOT = Path("/kaggle/working/geocpu_parent_fixed44_fp16_20261104")


def digest(path: Path) -> str:
    h = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not Path("/kaggle").is_dir() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Only a private disposable Kaggle runtime is admitted")
    if ROOT.exists():
        raise FileExistsError("This notebook output is one-use")
    ROOT.mkdir(parents=True)
    code = ROOT / "source"
    code.mkdir()
    for name, item in EMBEDDED.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid bundled relative path")
        raw = zlib.decompress(base64.b64decode(item["zlib_base64"], validate=True))
        if len(raw) != item["bytes"] or sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError("Bundled source changed: " + name)
        target = code / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    candidates = (Path("/kaggle/input/geocpu-v43-v44fix-paired-eval-20261103"),
                  Path("/kaggle/input/datasets/geocpuresearch/geocpu-v43-v44fix-paired-eval-20261103"))
    mounted = [path for path in candidates if path.is_dir()]
    if len(mounted) != 1:
        raise FileNotFoundError("Expected one pinned v43 Kaggle dataset mount")
    data = mounted[0]
    source_zip = data / "v43_source.zip.bin"
    parent = data / "v43_step15000.pt"
    if not source_zip.is_file() or not parent.is_file():
        raise FileNotFoundError("Pinned original v43 Kaggle input is absent")
    if digest(source_zip) != V43_ZIP_SHA or digest(parent) != PARENT_SHA:
        raise ValueError("Pinned original v43 Kaggle input changed")
    command = [sys.executable, "-u", str(code / "launch_fixed_diagnostic_bounded.py"),
               "--v43-zip", str(source_zip),
               "--overlay-source", str(code / "overlay"),
               "--checkpoint", str(parent), "--checkpoint-kind", "parent",
               "--output-dir", str(ROOT / "actual"),
               "--prelaunch-contract", str(code / "prelaunch_contract_once.json"),
               "--hard-stop-seconds", "1800", "--save-reserve-seconds", "180"]
    with (ROOT / "evaluation.log").open("x", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=code, stdout=log, stderr=subprocess.STDOUT,
                                text=True, check=False)
    if result.returncode:
        raise RuntimeError("Parent diagnostic failed; inspect the saved evaluation.log")
    complete = ROOT / "actual" / "COMPLETE.json"
    if not complete.is_file():
        raise RuntimeError("Parent diagnostic did not produce a COMPLETE record")
    record = json.loads(complete.read_text(encoding="utf-8"))
    raw = ROOT / "actual" / "raw_answers_44_profiles.jsonl"
    if (record.get("status") != "COMPLETE" or record.get("arm") != "parent"
            or record.get("training_updates") != 0
            or record.get("precision_profile") != "eval_fp16_native_v1"
            or record.get("max_generated_tokens") != 2048
            or record.get("profile_counts") != {"native": 44, "python_cued": 20,
                                                 "python_uncued": 20}
            or record.get("raw_answers_sha256") != digest(raw)):
        raise RuntimeError("Parent raw-answer closure failed")
    print(json.dumps({"status": "COMPLETE", "raw_answers": 84,
                      "raw_answers_sha256": digest(raw),
                      "complete_sha256": digest(complete)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
