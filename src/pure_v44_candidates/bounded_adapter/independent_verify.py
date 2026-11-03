"""CPU-only full replay of a proposed Python-TIR curriculum.

Run this separately from dataset construction. Every train row must name an
immutable original JSONL source row. Public Python observations are executed
again; legacy conversions are rebuilt through the original v43 CPU circuit
and the pure-Python worker. The output is evidence, not a model score.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from adapter import file_sha256
from admission import source_ref_fields, verify_dataset, verify_source_manifest


SCHEMA = "geocpu_independent_full_row_replay_v1"


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_digest(value: str) -> str:
    return digest(value.encode("utf-8"))


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def safe_relative(value: str) -> str:
    p = Path(value)
    if (not value or p.is_absolute() or ":" in value or value.startswith(("/", "\\"))
            or any(part in ("..", "") for part in p.parts)):
        raise ValueError("Source reference escapes its pinned root")
    return value.replace("\\", "/")


def source_rows(train_rows: list[dict], source_root: Path, legacy_zip: Path,
                manifest: dict, pure_source_dir: Path) -> dict[str, tuple[dict, str]]:
    """Return exact source JSON and line SHA for every referenced train row."""
    requests = defaultdict(dict)
    for row in train_rows:
        ref = row.get("source_ref")
        kind, name, line, source_sha = source_ref_fields(ref)
        name = safe_relative(name)
        key = (kind, name)
        if line in requests[key] and requests[key][line] != source_sha:
            raise ValueError("Conflicting source SHA for the same original row")
        requests[key][line] = source_sha
    found = {}
    public_lock = json.loads((pure_source_dir / "public_source_lock_v44.json").read_text(encoding="utf-8"))
    allowed = {"numina/" + public_lock["repos"]["numina"]["files"]["train"]["path"]:
               ("public_parquet", public_lock["repos"]["numina"]["files"]["train"]["sha256"]),
               "japanese/" + public_lock["repos"]["japanese"]["files"]["train"]["path"]:
               ("public_json", public_lock["repos"]["japanese"]["files"]["train"]["sha256"]),
               "code/" + public_lock["repos"]["code"]["files"]["train"]["path"]:
               ("public_jsonl", public_lock["repos"]["code"]["files"]["train"]["sha256"])}
    with zipfile.ZipFile(legacy_zip) as archive:
        if file_sha256(legacy_zip) != manifest["source_files"]["v43_zip_sha256"]:
            raise ValueError("Pinned v43 archive SHA differs")
        for (kind, name), lines in requests.items():
            if kind != "v43_zip_member":
                path = source_root / name
                expected = manifest["source_files"].get(name)
                if allowed.get(name) != (kind, expected) or file_sha256(path) != expected:
                    raise ValueError("Public source file SHA differs: " + name)
                if kind == "public_parquet":
                    import pyarrow.parquet as pq
                    records = (item for batch in pq.ParquetFile(path).iter_batches(batch_size=256)
                               for item in batch.to_pylist())
                elif kind == "public_json":
                    records = iter(json.loads(path.read_text(encoding="utf-8-sig")))
                else:
                    records = (json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip())
                remaining = set(lines)
                for index, item in enumerate(records):
                    if index not in remaining:
                        continue
                    sha = digest(canonical_bytes(item))
                    if sha != lines[index]:
                        raise ValueError(f"Public source row SHA differs: {name}:{index}")
                    found[(kind, name, index)] = (item, sha)
                    remaining.remove(index)
                    if not remaining:
                        break
                if remaining:
                    raise ValueError(f"Public source rows absent from {name}: {sorted(remaining)[:3]}")
                continue
            else:
                if name != "data/train_v43_knowledge_unified.jsonl":
                    raise ValueError("Unsupported v43 source member")
                stream = archive.open(name)
            with stream:
                remaining = set(lines)
                for index, raw in enumerate(stream, 1):
                    if index not in remaining:
                        continue
                    sha = digest(raw)
                    if sha != lines[index]:
                        raise ValueError(f"Source row SHA differs: {name}:{index}")
                    found[(kind, name, index)] = (json.loads(raw), sha)
                    remaining.remove(index)
                    if not remaining:
                        break
                if remaining:
                    raise ValueError(f"Source rows absent from {name}: {sorted(remaining)[:3]}")
    return found


def replay_public(row: dict, source: dict, executor, public_prepare) -> tuple[int, str]:
    """Rebuild a Numina row from its original problem/solution/messages fields."""
    candidate = public_prepare.public_candidate("tir", source)
    target, spans, receipts = public_prepare.normalize_tir(candidate["target"], executor=executor)
    if (row.get("prompt") != candidate["prompt"] or row.get("target") != target
            or row.get("full_text") != candidate["prompt"] + target
            or row.get("target_segments") != spans):
        raise ValueError("Public Numina source to executed Python curriculum differs")
    if not 1 <= len(receipts) <= 8 or any(r["verification"] != "executed_all_blocks_in_this_problem_stdout_exact" for r in receipts):
        raise ValueError("Public Python observations were not all executed and matched")
    return len(receipts), text_digest("".join(target[s["start"]:s["end"]]
                                               for s in spans if s["role"] == "tool"))


def replay_legacy(row: dict, source: dict, cpu, executor, converter, vocab: dict) -> tuple[int, str]:
    produced = converter.convert_row(source, cpu, executor, vocab)
    for key in ("prompt", "target", "full_text", "target_segments"):
        if row.get(key) != produced.get(key):
            raise ValueError(f"Legacy CPU/Python conversion differs: {row['id']}:{key}")
    return len(produced["target_segments"]), text_digest("".join(
        row["target"][span["start"]:span["end"]] for span in produced["target_segments"]))


def replay_anchor(row: dict, source: dict, public_prepare) -> tuple[int, str]:
    if row["bucket"] == "japanese_prose":
        candidate = public_prepare.public_candidate("japanese", source)
    elif row["bucket"] == "public_multilingual_code":
        candidate = public_prepare.public_candidate("code", source)
    else:
        candidate = public_prepare.anchor_candidate(source)
    if (row.get("prompt") != candidate["prompt"] or row.get("target") != candidate["target"]
            or row.get("full_text") != candidate["prompt"] + candidate["target"]):
        raise ValueError("Original Dolly/v43 tool-free source was changed")
    if row.get("target_segments"):
        raise ValueError("Tool-free anchor cannot carry tool spans")
    return 0, text_digest("")


def verify(*, data_dir: Path, data_manifest_sha256: str, source_root: Path,
           legacy_zip: Path, legacy_cpu_dir: Path, pure_source_dir: Path,
           pure_source_manifest_sha256: str,
           converter_path: Path, converter_sha256: str, meta_path: Path,
           meta_sha256: str, output_dir: Path) -> dict:
    if not Path("/kaggle").exists() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Independent public-code replay requires disposable Kaggle without mounted Drive")
    structural = verify_dataset(data_dir, data_manifest_sha256)
    verify_source_manifest(pure_source_dir, pure_source_manifest_sha256)
    if file_sha256(converter_path) != converter_sha256:
        raise ValueError("Converter source SHA differs")
    manifest = json.loads((data_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("verification", {}).get("converter_source_sha256") != converter_sha256:
        raise ValueError("Converter SHA is not pinned in dataset manifest")
    v43_files = ("qwenputer_controller_v30.py", "cpu_circuit_v30.py")
    with zipfile.ZipFile(legacy_zip) as archive:
        for name in v43_files:
            if archive.read(name) != (legacy_cpu_dir / name).read_bytes():
                raise ValueError("Original CPU implementation differs from v43 archive")
    train_raw = [line for line in (data_dir / "train.jsonl").read_bytes().splitlines(keepends=True) if line.strip()]
    train = [json.loads(line) for line in train_raw]
    sources = source_rows(train, source_root, legacy_zip, manifest, pure_source_dir)
    sys.path[:0] = [str(pure_source_dir), str(legacy_cpu_dir)]
    from qwenputer_controller_v30 import QwenputerExecutionController
    from cpu_circuit_v30 import CompiledOperators32, ExactRAM8x32
    from tir_engine_v44r1 import PythonExecutor
    from training_data_v44 import encode_supervision
    import prepare_public_v44 as public_prepare
    module_spec = importlib.util.spec_from_file_location("pinned_legacy_converter", converter_path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError("Converter cannot be loaded")
    converter = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(converter)
    if file_sha256(meta_path) != meta_sha256:
        raise ValueError("Tokenizer metadata SHA differs")
    vocab = json.loads(meta_path.read_text(encoding="utf-8"))["vocab"]
    cpu = QwenputerExecutionController(ExactRAM8x32(), CompiledOperators32())
    receipts = []
    with PythonExecutor() as executor:
        for line, row in zip(train_raw, train):
            ref = row["source_ref"]
            kind, name, line_number, ref_sha = source_ref_fields(ref)
            source, source_sha = sources[(kind, safe_relative(name), line_number)]
            if source_sha != ref_sha:
                raise ValueError("Training row source reference SHA differs")
            bucket = row["bucket"]
            if bucket == "converted_legacy_python":
                if kind != "v43_zip_member":
                    raise ValueError("Converted legacy row is not bound to v43 archive")
                calls, observation_sha = replay_legacy(row, source, cpu, executor, converter, vocab)
            elif bucket == "public_python_tir":
                if kind != "public_parquet":
                    raise ValueError("Public Python row lacks public source")
                calls, observation_sha = replay_public(row, source, executor, public_prepare)
            elif bucket in ("japanese_prose", "v43_tool_free_anchor", "public_multilingual_code"):
                if bucket == "japanese_prose" and kind != "public_json":
                    raise ValueError("Japanese prose row lacks original Dolly source")
                if bucket == "public_multilingual_code" and kind != "public_jsonl":
                    raise ValueError("Multilingual code row lacks original Magicoder source")
                if bucket == "v43_tool_free_anchor" and kind != "v43_zip_member":
                    raise ValueError("Original anchor is not bound to v43 archive")
                calls, observation_sha = replay_anchor(row, source, public_prepare)
            else:
                raise ValueError("Unrecognized training bucket: " + str(bucket))
            ids, tags = encode_supervision(row, vocab, max_positions=8192)
            if len(ids) < 2 or tags[-1] != 1:
                raise ValueError("Full supervision or true EOS is missing")
            receipt = {"id": row["id"], "train_row_sha256": digest(line),
                       "source_row_sha256": source_sha, "bucket": bucket,
                       "tool_calls_replayed": calls, "observations_sha256": observation_sha}
            receipts.append(receipt)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    receipts_path = output_dir / "receipts.jsonl"
    with receipts_path.open("x", encoding="utf-8", newline="\n") as stream:
        for receipt in receipts:
            stream.write(canonical_bytes(receipt).decode("utf-8") + "\n")
    proof = {"schema": SCHEMA, "status": "ALL_ROWS_REPLAYED",
             "verifier_source_sha256": file_sha256(__file__),
             "converter_source_sha256": converter_sha256,
             "pure_source_manifest_sha256": pure_source_manifest_sha256,
             "meta_sha256": meta_sha256,
             "data_manifest_sha256": data_manifest_sha256,
             "train_sha256": structural["files_sha256"]["train.jsonl"],
             "v43_zip_sha256": manifest["source_files"]["v43_zip_sha256"],
             "train_rows": len(train), "receipts_sha256": file_sha256(receipts_path),
             "replayed_tool_calls": sum(r["tool_calls_replayed"] for r in receipts)}
    (output_dir / "proof.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("data_dir", "data_manifest_sha256", "source_root", "legacy_zip",
                 "legacy_cpu_dir", "pure_source_dir", "pure_source_manifest_sha256", "converter_path",
                 "converter_sha256", "meta_path", "meta_sha256", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    args = parser.parse_args(argv)
    result = verify(data_dir=Path(args.data_dir), data_manifest_sha256=args.data_manifest_sha256,
                    source_root=Path(args.source_root), legacy_zip=Path(args.legacy_zip),
                    legacy_cpu_dir=Path(args.legacy_cpu_dir), pure_source_dir=Path(args.pure_source_dir),
                    pure_source_manifest_sha256=args.pure_source_manifest_sha256,
                    converter_path=Path(args.converter_path), converter_sha256=args.converter_sha256,
                    meta_path=Path(args.meta_path), meta_sha256=args.meta_sha256,
                    output_dir=Path(args.output_dir))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
