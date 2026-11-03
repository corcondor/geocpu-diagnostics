"""Re-execute every pinned v43 CPU -> Python row in an isolated CPU process.

No model is loaded and no optimizer update occurs.  A COMPLETE manifest is
written only after each of the 24,836 original CPU traces and converted Python
observations has been executed and matched again.  The conversion code is
source-pinned and its CPU/Python operations are rerun, not trusted as a count.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

from source_pools import (CONVERTED_ROWS, ORIGINAL_TRAIN_MEMBER, PINNED_CONVERTED_SHA,
                          PINNED_V43_ZIP_SHA, file_sha)

PINNED_CONVERTER_SHA = "82119783a01cfbac8a39f27e49458ce478434ecebce0043787b17091fcc44a41"
PINNED_PURE_SOURCE_MANIFEST_SHA = "7c072451921710d7a43ee3172957def3c279e40a21ae03ec6c416d07a84baebf"
IMPORTABLE_SUFFIXES = {".py", ".pyc", ".pyd", ".so", ".dll", ".pth"}


def _verify_importable_tree(root: Path, pinned: dict[str, str]) -> None:
    """Reject unpinned modules before adding the source directory to sys.path."""
    if root.is_symlink():
        raise ValueError("Replay source root is a symbolic link")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValueError("Replay source contains a symbolic link")
        if not candidate.is_file() or candidate.suffix.lower() not in IMPORTABLE_SUFFIXES:
            continue
        relative = candidate.relative_to(root).as_posix()
        if relative not in pinned or file_sha(candidate) != pinned[relative]:
            raise ValueError("Replay source contains an unpinned importable module: " + relative)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load pinned conversion source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def replay(*, v43_zip: Path, converted: Path, v43_code_dir: Path,
           python_code_dir: Path, converter_path: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError("Use a new output directory for this one-time replay")
    if (file_sha(v43_zip) != PINNED_V43_ZIP_SHA
            or file_sha(converted) != PINNED_CONVERTED_SHA
            or file_sha(converter_path) != PINNED_CONVERTER_SHA):
        raise ValueError("Pinned v43, conversion, or converter source differs")
    manifest_path = python_code_dir / "hotfix_manifest.json"
    if file_sha(manifest_path) != PINNED_PURE_SOURCE_MANIFEST_SHA:
        raise ValueError("Pinned public Python source manifest differs")
    pure_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(pure_manifest.get("files"), dict):
        raise ValueError("Pinned public Python source file list is missing")
    for name, item in pure_manifest["files"].items():
        relative = Path(name)
        if (relative.is_absolute() or ".." in relative.parts
                or not isinstance(item, dict) or file_sha(python_code_dir / relative) != item.get("sha256")):
            raise ValueError("Pinned public Python source file changed: " + name)
    pure_pins = {name: item["sha256"] for name, item in pure_manifest["files"].items()}
    _verify_importable_tree(python_code_dir, pure_pins)
    for name in ("tir_engine_v44r1.py", "training_data_v44.py",
                 "executor_worker_v44r1.py", "resource_limited_worker_v44r1.py"):
        if file_sha(python_code_dir / name) != pure_manifest["files"][name]["sha256"]:
            raise ValueError("Python execution source differs from pinned pure-v44 bundle: " + name)
    with zipfile.ZipFile(v43_zip) as archive:
        v43_pins = {member.filename: sha256(archive.read(member)).hexdigest()
                    for member in archive.infolist() if not member.is_dir()
                    and Path(member.filename).suffix.lower() in IMPORTABLE_SUFFIXES}
        _verify_importable_tree(v43_code_dir, v43_pins)
        for name in ("qwenputer_controller_v30.py", "cpu_circuit_v30.py"):
            if archive.read(name) != (v43_code_dir / name).read_bytes():
                raise ValueError("Runtime CPU source differs from original v43 ZIP: " + name)
        meta = json.loads(archive.read("base_metadata.json"))
        vocab = meta["vocab"]
        converted_rows = [json.loads(line) for line in converted.read_text(encoding="utf-8").splitlines()]
        if len(converted_rows) != CONVERTED_ROWS:
            raise ValueError("Converted source has the wrong row count")
        required_lines = {row["source_ref"]["line"] for row in converted_rows}
        originals = {}
        with archive.open(ORIGINAL_TRAIN_MEMBER) as stream:
            for line_no, raw in enumerate(stream, 1):
                if line_no in required_lines:
                    originals[line_no] = (raw, json.loads(raw))
        if len(originals) != len(required_lines) or len(required_lines) != CONVERTED_ROWS:
            raise ValueError("Converted source lines are missing or duplicated")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(v43_code_dir))
    sys.path.insert(0, str(python_code_dir))
    import torch
    cpu_module = _load_module(v43_code_dir / "cpu_circuit_v30.py", "cpu_circuit_v30")
    sys.modules["cpu_circuit_v30"] = cpu_module
    controller_module = _load_module(v43_code_dir / "qwenputer_controller_v30.py", "qwenputer_controller_v30")
    sys.modules["qwenputer_controller_v30"] = controller_module
    tir_module = _load_module(python_code_dir / "tir_engine_v44r1.py", "tir_engine_v44r1")
    sys.modules["tir_engine_v44r1"] = tir_module
    training_module = _load_module(python_code_dir / "training_data_v44.py", "training_data_v44")
    sys.modules["training_data_v44"] = training_module
    for module, expected in ((tir_module, "tir_engine_v44r1.py"),
                             (training_module, "training_data_v44.py")):
        if Path(module.__file__).resolve() != (python_code_dir / expected).resolve():
            raise ValueError("Python execution imported an unverified module")
    CompiledOperators32 = cpu_module.CompiledOperators32
    ExactRAM8x32 = cpu_module.ExactRAM8x32
    QwenputerExecutionController = controller_module.QwenputerExecutionController
    PythonExecutor = tir_module.PythonExecutor
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)
    converter = _load_module(converter_path, "pinned_legacy_to_python_converter")
    cpu = QwenputerExecutionController(ExactRAM8x32(), CompiledOperators32())
    output_dir.mkdir(parents=True)
    row_file = output_dir / "all_row_replay_receipts.jsonl"
    total_blocks = 0
    with PythonExecutor() as executor, row_file.open("x", encoding="utf-8", newline="\n") as sink:
        for index, row in enumerate(converted_rows, 1):
            ref = row["source_ref"]
            raw, original = originals[ref["line"]]
            if (ref["kind"] != "v43_zip_member" or ref["file"] != ORIGINAL_TRAIN_MEMBER
                    or sha256(raw).hexdigest() != ref["raw_sha256"]
                    or row["prompt"] != original["prompt"]):
                raise ValueError(f"Original source closure failed at converted row {index}")
            regenerated = converter.convert_row(original, cpu, executor, vocab)
            for key in ("prompt", "target", "full_text", "target_segments",
                        "original_source", "original_prompt_sha256"):
                if regenerated.get(key) != row.get(key):
                    raise ValueError(f"CPU/Python execution differs at converted row {index}: {key}")
            blocks = len(regenerated["target_segments"])
            if not 1 <= blocks <= 8:
                raise ValueError("Python execution block count is invalid")
            total_blocks += blocks
            receipt = {"line": index, "source_line": ref["line"],
                       "source_raw_sha256": ref["raw_sha256"],
                       "converted_row_sha256": sha256(json.dumps(row, ensure_ascii=False,
                           sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                       "python_blocks_executed": blocks,
                       "tool_observation_sha256": sha256("".join(row["target"][s["start"]:s["end"]]
                           for s in row["target_segments"]).encode()).hexdigest()}
            sink.write(json.dumps(receipt, ensure_ascii=False, sort_keys=True) + "\n")
            if index % 1000 == 0:
                sink.flush()
                print(json.dumps({"replayed": index, "python_blocks_executed": total_blocks}), flush=True)
    manifest = {"schema": "geocpu_hybrid_independent_python_replay_v1",
                "source_zip_sha256": PINNED_V43_ZIP_SHA,
                "converted_sha256": PINNED_CONVERTED_SHA,
                "converter_source_sha256": PINNED_CONVERTER_SHA,
                "verifier_source_sha256": file_sha(__file__),
                "accepted_rows": CONVERTED_ROWS,
                "python_blocks_executed": total_blocks,
                "all_python_blocks_executed": True,
                "row_receipts_sha256": file_sha(row_file)}
    with (output_dir / "COMPLETE.json").open("x", encoding="utf-8") as sink:
        json.dump(manifest, sink, ensure_ascii=False, sort_keys=True, indent=2)
        sink.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    for name in ("v43_zip", "converted", "v43_code_dir", "python_code_dir", "converter_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
