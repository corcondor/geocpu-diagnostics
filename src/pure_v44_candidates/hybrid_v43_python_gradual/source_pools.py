"""Pinned v43 corpus plus verified-conversion pool for gradual hybrid stages.

All 268,032 original rows are inspected.  Evaluation overlaps are quarantined
before sampling; the accepted old pool keeps the original row text, tokenizer,
and target mask.  No v44fix prepared mixed dataset is accepted here.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
import unicodedata
import zipfile

from hybrid_data import EncodedExample, encode_example


PINNED_V43_ZIP_SHA = "0102dc9b01b4b795c3123b18cefc7a8734e47a06dd85292af464d3232704d198"
PINNED_META_SHA = "7efa8dea9a0100e2250f72805c89cca7a281efec30b4479795743814756cdcd5"
PINNED_CONVERTED_SHA = "53387b2987df02d91b51b4afc8ab2369cc0e53b9e4bdd0cfd480c3586de7f3db"
PINNED_REPLAY_SOURCE_SHA = "9b1b4116dd9bb60e85d257dd5e39dd53581bc35a96ee1670894e2724cc6cf4a8"
ORIGINAL_TRAIN_MEMBER = "data/train_v43_knowledge_unified.jsonl"
ORIGINAL_ROWS = 268_032
CONVERTED_ROWS = 24_836
PYTHON_MODE_CUE = "Pythonで計算し、実行結果を根拠に日本語で説明してください。\n"
PURE_V44_PYTHON_BLOCK = re.compile(r'(?ms)^```python[ \t]*\r?\n(.*?)^```[ \t]*(?:\r?\n|$)')
KNOWN_BAD_GSM_LINE = 13
KNOWN_BAD_GSM_RAW_SHA = "8eb226f7ded3ea96ac91a175defb708e6f119667729b116e4be1fab09b10f18d"
REQUIRED_EVAL_SHA = {
    "data/eval_v42_search_retrieval_aligned.jsonl": "c07d962d4fa59196d74b8223353a5f52d60cabe3d58bfe604c9d83b708e45ab6",
    "data/eval_v42_search_retrieval_heldout.jsonl": "805d4b91ef9128095191cdeb1180e95b0759196f3d471204d2e47c6f77e6ef53",
    "data/eval_swe_step_control_heldout.jsonl": "0bf62099ea2a41c8eb35c1ebdc2584f2c164ae3c82f741ab3eca44113f70f8cd",
    "data/eval_v38_decomposed_state_bench200.jsonl": "e96db14ba2b6ec620d42bcaf598199da814d93cef971c334889bd88147567cd1",
    "data/eval_v33_state_tracking_heldout.jsonl": "6406c92c106512d2046368d6acba2dfb2ff24181301a14e975d76700ab7ab7b5",
    "data/jamcqa_manifest_test.jsonl": "f7590adbce3d8f444b089d102a7f79be091c942c7c132d7090fd4e3d0710543e",
    "data/benchmark_8100_manifest.jsonl": "33723d78aaa1595b1ed7dbc3382ecb165ce788246f6f70a63470f5fa107de78e",
    "data/eval_v41_logical_instruction_operators_heldout.jsonl": "b887aade00323a7241088a4e84dfcb166430f554209a09668ad204ea564e9989",
}


def file_sha(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_question(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).strip()
    text = re.sub(r"^(?:ユーザー|user)\s*[:：]\s*", "", text, flags=re.I)
    text = re.sub(r"\n\s*(?:アシスタント|assistant)\s*[:：]\s*$", "", text, flags=re.I)
    text = re.sub(r"^\s*【分野:[^\n]*】\s*", "", text)
    text = re.sub(r"(?m)^\s*分野\s*[:：][^\n]*\n?", "", text)
    text = re.sub(r"\s*(?:正解|回答|答え)\s*[:：]\s*$", "", text)
    return re.sub(r"\s+", "", text)


def _original_normalizer(source: str):
    tree = ast.parse(source)
    base = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_BASE_NORM_DICT"
                                                for target in node.targets):
            base = ast.literal_eval(node.value)
            break
    if not isinstance(base, dict):
        raise ValueError("Pinned original normalization map is missing")
    for code in range(0xFF01, 0xFF5F):
        char = chr(code)
        if char not in ("（", "）", "！", "？", "：", "～"):
            base[char] = chr(code - 0xFEE0)
    table = str.maketrans(base)
    return lambda text: text.translate(table)


def _eval_keys(archive: zipfile.ZipFile) -> set[str]:
    keys: set[str] = set()
    for member, expected in REQUIRED_EVAL_SHA.items():
        raw = archive.read(member)
        if sha256(raw).hexdigest() != expected:
            raise ValueError("Original evaluation manifest changed: " + member)
        for line in raw.splitlines():
            row = json.loads(line)
            for key in ("prompt", "question", "input", "instruction", "problem"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    keys.add(canonical_question(value))
    if not keys:
        raise ValueError("Original evaluation family index is empty")
    return keys


@dataclass
class SourcePools:
    old: list[EncodedExample]
    python: list[EncodedExample]
    old_source_lines: list[int]
    python_source_lines: list[int]
    counts: dict[str, int]
    metadata: dict


def load_verified_pools(*, v43_zip: str | Path, converted_jsonl: str | Path,
                        require_independent_replay: bool = True,
                        replay_receipt: str | Path | None = None) -> SourcePools:
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA or file_sha(converted_jsonl) != PINNED_CONVERTED_SHA:
        raise ValueError("Pinned original v43 or conversion source SHA changed")
    replay_rows = None
    if require_independent_replay:
        if replay_receipt is None:
            raise ValueError("Independent full Python execution replay is required before training")
        receipt_path = Path(replay_receipt)
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        replay_file = receipt_path.parent / "all_row_replay_receipts.jsonl"
        source_file = Path(__file__).with_name("replay_python_full.py")
        if (receipt.get("schema") != "geocpu_hybrid_independent_python_replay_v1"
                or receipt.get("source_zip_sha256") != PINNED_V43_ZIP_SHA
                or receipt.get("converted_sha256") != PINNED_CONVERTED_SHA
                or receipt.get("accepted_rows") != CONVERTED_ROWS
                or receipt.get("all_python_blocks_executed") is not True
                or receipt.get("verifier_source_sha256") != PINNED_REPLAY_SOURCE_SHA
                or file_sha(source_file) != PINNED_REPLAY_SOURCE_SHA
                or not replay_file.is_file()
                or receipt.get("row_receipts_sha256") != file_sha(replay_file)):
            raise ValueError("Independent Python replay receipt is incomplete")
        replay_rows = [json.loads(line) for line in replay_file.read_text(encoding="utf-8").splitlines()]
        if len(replay_rows) != CONVERTED_ROWS:
            raise ValueError("Independent Python replay is missing rows")
    counts: Counter[str] = Counter()
    old: list[EncodedExample] = []
    python: list[EncodedExample] = []
    old_lines: list[int] = []
    py_lines: list[int] = []
    originals: dict[int, tuple[str, dict]] = {}
    original_training_prompts: set[str] = set()
    converted_rows = []
    python_heldout_keys = set()
    with Path(converted_jsonl).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            fences = row["target"].count("```python")
            if not fences or len(PURE_V44_PYTHON_BLOCK.findall(row["target"])) != fences:
                raise ValueError("A converted Python tool call is not executable by the actual v44 parser")
            converted_rows.append(row)
            key = canonical_question(row.get("prompt", ""))
            if int(sha256(key.encode()).hexdigest()[:12], 16) % 100 >= 90:
                python_heldout_keys.add(key)
    if len(converted_rows) != CONVERTED_ROWS:
        raise ValueError("Converted full source is incomplete")
    required_original_lines = {row["source_ref"]["line"] for row in converted_rows}
    if len(required_original_lines) != CONVERTED_ROWS:
        raise ValueError("Converted source references are not unique")
    with zipfile.ZipFile(v43_zip) as archive:
        metadata_raw = archive.read("base_metadata.json")
        if sha256(metadata_raw).hexdigest() != PINNED_META_SHA:
            raise ValueError("Original v43 tokenizer or config changed")
        metadata = json.loads(metadata_raw)
        if metadata["config"]["max_positions"] != 2048:
            raise ValueError("Original v43 context is not 2048")
        vocab = metadata["vocab"]
        normalize = _original_normalizer(archive.read("qwenputer_controller_v30.py").decode("utf-8"))
        eval_keys = _eval_keys(archive)
        with archive.open(ORIGINAL_TRAIN_MEMBER) as stream:
            for line_number, raw in enumerate(stream, 1):
                counts["original_rows_scanned"] += 1
                row = json.loads(raw)
                if line_number in required_original_lines:
                    originals[line_number] = (sha256(raw).hexdigest(), row)
                original_training_prompts.add(row.get("prompt", ""))
                if line_number == KNOWN_BAD_GSM_LINE:
                    if (sha256(raw).hexdigest() != KNOWN_BAD_GSM_RAW_SHA
                            or row.get("source") != "GSM8K_Japanese_Full"):
                        raise ValueError("Pinned incorrect-label row identity changed")
                    counts["known_incorrect_gsm_label_quarantined"] += 1
                    continue
                if canonical_question(row.get("prompt", "")) in eval_keys:
                    counts["original_eval_overlap_quarantined"] += 1
                    continue
                if canonical_question(row.get("prompt", "")) in python_heldout_keys:
                    counts["original_python_heldout_overlap_quarantined"] += 1
                    continue
                try:
                    example = encode_example(row, vocab, kind="v43", normalize=normalize)
                except ValueError:
                    counts["original_no_active_labels_quarantined"] += 1
                    continue
                old.append(example)
                old_lines.append(line_number)
                counts["original_family_rows:" + example.source_family] += 1
                if re.search(r"[ぁ-んァ-ン一-龯]", row.get("target", "")):
                    counts["original_japanese_character_rows_retained"] += 1
                counts["original_supervised_positions_retained"] += example.active_positions
    if counts["original_rows_scanned"] != ORIGINAL_ROWS:
        raise ValueError("Original v43 268,032-row corpus is incomplete")
    seen_refs: set[int] = set()
    for converted_line, row in enumerate(converted_rows, 1):
        counts["converted_rows_scanned"] += 1
        if replay_rows is not None:
            replay = replay_rows[converted_line-1]
            expected_digest = sha256(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                      separators=(",", ":")).encode()).hexdigest()
            expected_observation = sha256("".join(row["target"][s["start"]:s["end"]]
                for s in row["target_segments"] if s.get("role") == "tool").encode()).hexdigest()
            if (replay.get("line") != converted_line
                    or replay.get("source_line") != row["source_ref"]["line"]
                    or replay.get("source_raw_sha256") != row["source_ref"]["raw_sha256"]
                    or replay.get("converted_row_sha256") != expected_digest
                    or replay.get("tool_observation_sha256") != expected_observation
                    or replay.get("python_blocks_executed") != len(row["target_segments"])):
                raise ValueError("Independent executed row differs from converted row")
        ref = row.get("source_ref", {})
        if (ref.get("kind") != "v43_zip_member" or ref.get("file") != ORIGINAL_TRAIN_MEMBER
                or type(ref.get("line")) is not int or ref["line"] not in originals
                or ref["line"] in seen_refs):
            raise ValueError("Converted row lacks a unique original ZIP line")
        raw_sha, original = originals[ref["line"]]
        if (ref.get("raw_sha256") != raw_sha or row.get("prompt") != original.get("prompt")
                or row.get("original_prompt_sha256") != sha256(row["prompt"].encode()).hexdigest()):
            raise ValueError("Converted row does not close over the original v43 row")
        seen_refs.add(ref["line"])
        key = canonical_question(row["prompt"])
        if key in eval_keys:
            counts["converted_eval_overlap_quarantined"] += 1
            continue
        split = int(sha256(key.encode()).hexdigest()[:12], 16) % 100
        if split >= 90:
            counts["converted_dev_test_quarantined"] += 1
            continue
        training_prompt = PYTHON_MODE_CUE + row["prompt"]
        if training_prompt in original_training_prompts:
            raise ValueError("Explicit Python cue collides with an original v43 prompt")
        training_row = {"prompt": training_prompt, "target": row["target"],
                        "full_text": training_prompt + row["target"],
                        "target_segments": row["target_segments"],
                        "original_source": row["original_source"]}
        example = encode_example(training_row, vocab, kind="python", normalize=normalize)
        python.append(example)
        py_lines.append(converted_line)
        counts["python_family_rows:" + example.source_family] += 1
        counts["python_supervised_positions_retained"] += example.active_positions
        counts["python_tool_positions_masked"] += sum(end-start for start, end in example.tool_target_ranges)
    if counts["converted_rows_scanned"] != CONVERTED_ROWS:
        raise ValueError("Converted full source is incomplete")
    if len(old) < 200_000 or len(python) < 10_000:
        raise ValueError("Quarantine leaves too few original or Python examples")
    counts["original_eligible_rows"] = len(old)
    counts["converted_eligible_rows"] = len(python)
    counts["pinned_original_zip_rows"] = ORIGINAL_ROWS
    return SourcePools(old, python, old_lines, py_lines, dict(counts), metadata)
