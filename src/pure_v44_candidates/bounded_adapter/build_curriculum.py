"""Build from pinned original public rows and original v43 rows only.

No previous v44 prepared/train file is accepted. Numina Python blocks are
executed during selection; the separate independent verifier must replay all
selected rows again before any GPU training is admitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

from adapter import file_sha256
from admission import (SCHEMA, PINNED_V43_DATA_ZIP_SHA256, PINNED_V43_META_SHA256,
                       REQUIRED_V43_EVAL_SHA256,
                       VERIFIER_SOURCE_SHA256, canonical_prompt,
                       source_ref_fields, verify_source_manifest)
from independent_verify import canonical_bytes


V43_MEMBER = "data/train_v43_knowledge_unified.jsonl"
REQUIRED_V43_EVAL_MEMBERS = (
    "data/eval_v42_search_retrieval_aligned.jsonl",
    "data/eval_v42_search_retrieval_heldout.jsonl",
    "data/eval_swe_step_control_heldout.jsonl",
    "data/eval_v38_decomposed_state_bench200.jsonl",
    "data/eval_v33_state_tracking_heldout.jsonl",
    "data/jamcqa_manifest_test.jsonl",
    "data/benchmark_8100_manifest.jsonl",
    "data/eval_v41_logical_instruction_operators_heldout.jsonl",
)
PARENT_SHA256 = "a4b95d7d0f486146196271b3b1fbc6a1b6b3342d8fadd590320fc3842c1e80e5"


def digest_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def new_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(*, source_root, pure_source_dir, source_manifest_sha256, v43_zip,
          converted_path, converted_sha256, converter_path, converter_sha256,
          meta_path, meta_sha256, evaluation_dir, old_skill_eval,
          old_skill_eval_sha256, unseen_eval, unseen_eval_sha256,
          v43_original_answers, v43_original_answers_sha256, output_dir,
          max_tir=25_000, max_code=30_000, max_anchor=80_000):
    source_root, pure_source_dir, output_dir = map(Path, (source_root, pure_source_dir, output_dir))
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not Path("/kaggle").exists() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Public Python execution requires a disposable Kaggle runtime without mounted Drive")
    verify_source_manifest(pure_source_dir, source_manifest_sha256)
    if meta_sha256 != PINNED_V43_META_SHA256:
        raise ValueError("Tokenizer metadata differs from original v43 checkpoint")
    for path, expected in ((converted_path, converted_sha256), (converter_path, converter_sha256),
                           (meta_path, meta_sha256), (old_skill_eval, old_skill_eval_sha256),
                           (unseen_eval, unseen_eval_sha256),
                           (v43_original_answers, v43_original_answers_sha256)):
        if file_sha256(path) != expected:
            raise ValueError("Pinned input SHA differs: " + str(path))
    lock = json.loads((pure_source_dir / "public_source_lock_v44.json").read_text(encoding="utf-8"))
    source_files = {}
    paths = {}
    for name in ("numina", "japanese", "code"):
        spec = lock["repos"][name]["files"]["train"]
        relative = name + "/" + spec["path"]
        path = source_root / relative
        if path.stat().st_size != spec["bytes"] or file_sha256(path) != spec["sha256"]:
            raise ValueError("Original public source differs: " + relative)
        paths[name] = path
        source_files[relative] = spec["sha256"]
    test_spec = lock["repos"]["numina"]["files"]["test"]
    numina_test = source_root / ("numina/" + test_spec["path"])
    if numina_test.stat().st_size != test_spec["bytes"] or file_sha256(numina_test) != test_spec["sha256"]:
        raise ValueError("Original Numina test split differs")
    source_files["numina/"+test_spec["path"]] = test_spec["sha256"]
    if file_sha256(v43_zip) != PINNED_V43_DATA_ZIP_SHA256:
        raise ValueError("Original v43 dataset ZIP differs from fixed SHA")
    source_files["v43_zip_sha256"] = PINNED_V43_DATA_ZIP_SHA256
    sys.path.insert(0, str(pure_source_dir))
    from prepare_public_v44 import (RowRejected, anchor_candidate, canonical_question,
                                    evaluation_index, json_records, normalize_tir,
                                    public_candidate, split_question)
    from tir_engine_v44r1 import PythonExecutor
    from training_data_v44 import encode_supervision
    metadata = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    max_positions = metadata["config"]["max_positions"]
    if max_positions != 2048:
        raise ValueError("Pinned original v43 model context length differs from 2048")
    vocab = metadata["vocab"]
    with zipfile.ZipFile(v43_zip) as archive:
        for member in REQUIRED_V43_EVAL_MEMBERS:
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            if expected != REQUIRED_V43_EVAL_SHA256[Path(member).name]:
                raise ValueError("Pinned original v43 evaluation source differs: " + member)
            local = Path(evaluation_dir) / Path(member).name
            if not local.is_file() or file_sha256(local) != expected:
                raise ValueError("Required original v43 evaluation split missing or changed: " + member)
    eval_keys, eval_ids, eval_pins = evaluation_index(evaluation_dir, numina_test)
    old_eval, unseen_eval_rows = list(read_jsonl(old_skill_eval)), list(read_jsonl(unseen_eval))
    for row in old_eval + unseen_eval_rows:
        eval_keys.add(canonical_question(row["prompt"]))
        eval_ids.add(str(row["id"]))
    rows, prompt_keys, family_keys, stats = [], set(), set(), Counter()

    def accept(bucket, raw, ref, candidate, *, executor=None):
        try:
            question = candidate["question"]
            key = canonical_question(question)
            if not key or key in eval_keys or split_question(key) != "train":
                raise RowRejected("evaluation_or_nontrain_group")
            prompt = candidate["prompt"]
            if canonical_prompt(prompt) in prompt_keys or key in family_keys:
                raise RowRejected("duplicate_training_question")
            if bucket == "public_python_tir":
                target, spans, receipts = normalize_tir(candidate["target"], executor=executor)
                if not 1 <= len(receipts) <= 8:
                    raise RowRejected("python_tool_count_outside_generation_cap")
            else:
                target = candidate["target"]
                spans = []
            item = {"id": digest_text(bucket + ":" + ref["raw_sha256"] + ":" + prompt),
                    "family_id": digest_text("family:"+key), "bucket": bucket,
                    "prompt": prompt, "target": target, "full_text": prompt+target,
                    "target_segments": spans, "source_ref": ref}
            ids, tags = encode_supervision(item, vocab, max_positions=max_positions)
            if tags[-1] != 1 or len(ids) <= len(prompt):
                raise RowRejected("whole_answer_or_eos_not_supervised")
            rows.append(item)
            prompt_keys.add(canonical_prompt(prompt))
            family_keys.add(key)
            stats["accepted_"+bucket] += 1
        except (RowRejected, ValueError) as exc:
            stats["rejected_"+bucket+":"+str(exc).split(":", 1)[0]] += 1

    with PythonExecutor() as executor:
        for index, raw in enumerate(json_records(paths["numina"])):
            if stats["accepted_public_python_tir"] >= max_tir:
                break
            try:
                candidate = public_candidate("tir", raw)
            except RowRejected:
                continue
            ref = {"kind": "public_parquet", "file": "numina/"+lock["repos"]["numina"]["files"]["train"]["path"],
                   "line": index, "raw_sha256": hashlib.sha256(canonical_bytes(raw)).hexdigest()}
            accept("public_python_tir", raw, ref, candidate, executor=executor)
    for index, raw in enumerate(json_records(paths["japanese"])):
        try:
            candidate = public_candidate("japanese", raw)
        except RowRejected:
            continue
        ref = {"kind": "public_json", "file": "japanese/"+lock["repos"]["japanese"]["files"]["train"]["path"],
               "line": index, "raw_sha256": hashlib.sha256(canonical_bytes(raw)).hexdigest()}
        accept("japanese_prose", raw, ref, candidate)
    for index, raw in enumerate(json_records(paths["code"])):
        if stats["accepted_public_multilingual_code"] >= max_code:
            break
        try:
            candidate = public_candidate("code", raw)
        except RowRejected:
            continue
        ref = {"kind": "public_jsonl", "file": "code/"+lock["repos"]["code"]["files"]["train"]["path"],
               "line": index, "raw_sha256": hashlib.sha256(canonical_bytes(raw)).hexdigest()}
        accept("public_multilingual_code", raw, ref, candidate)
    with zipfile.ZipFile(v43_zip) as archive, archive.open(V43_MEMBER) as stream:
        for line_number, raw_line in enumerate(stream, 1):
            if stats["accepted_v43_tool_free_anchor"] >= max_anchor:
                break
            raw = json.loads(raw_line)
            try:
                candidate = anchor_candidate(raw)
            except RowRejected:
                continue
            ref = {"kind": "v43_zip_member", "file": V43_MEMBER, "line": line_number,
                   "raw_sha256": hashlib.sha256(raw_line).hexdigest()}
            accept("v43_tool_free_anchor", raw, ref, candidate)
    for converted in read_jsonl(converted_path):
        ref = converted.get("source_ref")
        kind, name, line, source_sha = source_ref_fields(ref)
        if kind != "v43_zip_member" or name != V43_MEMBER:
            raise ValueError("Converted CPU row lacks original v43 source reference")
        prompt = converted["prompt"]
        key = canonical_question(prompt)
        if key in eval_keys or split_question(key) != "train" or key in family_keys or canonical_prompt(prompt) in prompt_keys:
            stats["rejected_converted_legacy_python:overlap_or_nontrain"] += 1
            continue
        item = {"id": digest_text("converted:"+source_sha+":"+prompt),
                "family_id": digest_text("family:"+key), "bucket": "converted_legacy_python",
                "prompt": prompt, "target": converted["target"],
                "full_text": converted["full_text"],
                "target_segments": converted["target_segments"],
                "source_ref": {"kind": kind, "file": name, "line": line, "raw_sha256": source_sha}}
        try:
            encode_supervision(item, vocab, max_positions=max_positions)
        except ValueError as exc:
            stats["rejected_converted_legacy_python:"+str(exc).split(":", 1)[0]] += 1
            continue
        rows.append(item)
        prompt_keys.add(canonical_prompt(prompt))
        family_keys.add(key)
        stats["accepted_converted_legacy_python"] += 1
    if not rows:
        raise ValueError("No admitted original public/v43 rows")
    output_dir.mkdir(parents=True)
    originals = list(read_jsonl(v43_original_answers))
    files = {"train.jsonl": rows, "old_skill_eval.jsonl": old_eval,
             "unseen_eval.jsonl": unseen_eval_rows,
             "v43_original_answers.jsonl": originals}
    for name, content in files.items():
        new_jsonl(output_dir/name, content)
    counts = Counter()
    for row in rows:
        counts["train_rows"] += 1
        counts["train_"+row["bucket"]] += 1
        if row["bucket"] in ("public_python_tir", "converted_legacy_python"):
            counts["python_candidate_rows"] += 1
    for split, content in (("old_skill_eval", old_eval), ("unseen_eval", unseen_eval_rows)):
        counts[split+"_rows"] = len(content)
        for row in content:
            counts[split+"_"+row["bucket"]] += 1
            if split == "old_skill_eval" and row.get("v43_cpu_task") is True:
                counts["old_skill_eval_original_cpu_tasks"] += 1
    counts["v43_original_answers_rows"] = len(originals)
    manifest = {"schema": SCHEMA,
                "files": {name: {"sha256": file_sha256(output_dir/name)} for name in files},
                "source_files": source_files,
                "verification": {"converter_source_sha256": converter_sha256,
                                 "verifier_source_sha256": VERIFIER_SOURCE_SHA256},
                "quality_audit": {"source_family_separation_checked": True},
                "counts": dict(counts), "selected_source_rows": dict(stats),
                "evaluation_input_pins": eval_pins,
                "public_source_lock_sha256": file_sha256(pure_source_dir/"public_source_lock_v44.json"),
                "model_updates": 0, "independent_replay_status": "NOT_STARTED"}
    (output_dir/"dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return {"data_manifest_sha256": file_sha256(output_dir/"dataset_manifest.json"),
            "counts": dict(counts), "independent_replay_status": "NOT_STARTED"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("source_root", "pure_source_dir", "source_manifest_sha256", "v43_zip",
                 "converted_path", "converted_sha256", "converter_path", "converter_sha256",
                 "meta_path", "meta_sha256", "evaluation_dir", "old_skill_eval",
                 "old_skill_eval_sha256", "unseen_eval", "unseen_eval_sha256",
                 "v43_original_answers", "v43_original_answers_sha256", "output_dir"):
        parser.add_argument("--"+name.replace("_", "-"), required=True)
    parser.add_argument("--max-tir", type=int, default=25_000)
    parser.add_argument("--max-code", type=int, default=30_000)
    parser.add_argument("--max-anchor", type=int, default=80_000)
    args = parser.parse_args(argv)
    print(json.dumps(build(**vars(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
