"""Add the independently checked v43 SWE-to-Python bridge to the frozen v1 pool.

The original v43 rows remain unchanged.  The new pool is a separate admission
version so the earlier 24,836-row experiment can still be reproduced exactly.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from hybrid_data import encode_example
from source_pools import (CONVERTED_ROWS, ORIGINAL_TRAIN_MEMBER, PINNED_V43_ZIP_SHA,
                          PURE_V44_PYTHON_BLOCK, PYTHON_MODE_CUE, SourcePools,
                          _eval_keys, _original_normalizer, canonical_question,
                          file_sha, load_verified_pools)

SWE_SHA = "89b129cebec20eda5622c4c169601f8c6a67f844719f986b170f172421f5abc5"
SWE_MANIFEST_SHA = "b4b054164bdb8c8b50883ccdc22ebb936fe54a3c60ba190bccbd707e7917174a"
SWE_AUDIT_SHA = "78e3558c92bfd90cf5553734d09336c6e5a696331b7f450b68d2aef698705b64"
SWE_ROWS = 12_859


def load_verified_pools_v2(*, v43_zip: Path, converted_jsonl: Path,
                           replay_receipt: Path, swe_jsonl: Path,
                           swe_manifest: Path, swe_audit: Path) -> SourcePools:
    """Require both original provenance closures and isolate new held-out keys."""
    if (file_sha(v43_zip) != PINNED_V43_ZIP_SHA or file_sha(swe_jsonl) != SWE_SHA
            or file_sha(swe_manifest) != SWE_MANIFEST_SHA
            or file_sha(swe_audit) != SWE_AUDIT_SHA):
        raise ValueError("Pinned SWE or original source changed")
    manifest = json.loads(swe_manifest.read_text(encoding="utf-8"))
    audit = json.loads(swe_audit.read_text(encoding="utf-8"))
    if (manifest.get("status") != "complete" or manifest.get("counts", {}).get("accepted") != SWE_ROWS
            or manifest.get("output_sha256") != SWE_SHA
            or manifest.get("source_zip_sha256") != PINNED_V43_ZIP_SHA
            or audit.get("status") != "PASS" or audit.get("accepted") != SWE_ROWS
            or audit.get("converted_sha256") != SWE_SHA
            or audit.get("source_zip_sha256") != PINNED_V43_ZIP_SHA):
        raise ValueError("Executed SWE conversion or independent closure is incomplete")
    base = load_verified_pools(v43_zip=v43_zip, converted_jsonl=converted_jsonl,
                               replay_receipt=replay_receipt)
    with zipfile.ZipFile(v43_zip) as archive:
        normalize = _original_normalizer(archive.read("qwenputer_controller_v30.py").decode("utf-8"))
        eval_keys = _eval_keys(archive)
        vocab = base.metadata["vocab"]
        swe_rows = [json.loads(line) for line in swe_jsonl.open(encoding="utf-8")]
        if len(swe_rows) != SWE_ROWS:
            raise ValueError("The verified SWE conversion has missing rows")
        heldout_keys: set[str] = set()
        for row in swe_rows:
            key = canonical_question(row["prompt"])
            if int(sha256(key.encode()).hexdigest()[:12], 16) % 100 >= 90:
                heldout_keys.add(key)
        # A new held-out SWE question must not remain in either legacy source
        # pool merely because it was introduced after the v1 split was frozen.
        old_lines = set(base.old_source_lines)
        blocked_old_lines: set[int] = set()
        with archive.open(ORIGINAL_TRAIN_MEMBER) as stream:
            for line_number, raw in enumerate(stream, 1):
                if line_number in old_lines and canonical_question(json.loads(raw)["prompt"]) in heldout_keys:
                    blocked_old_lines.add(line_number)
        bad_dialogue = "JMultiWOZ_Dialogue_State_Tracking"
        removed_dialogue = sum(row.source_family == bad_dialogue for row in base.old)
        base.old, base.old_source_lines = (list(values) for values in zip(*(
            (row, line) for row, line in zip(base.old, base.old_source_lines)
            if line not in blocked_old_lines and row.source_family != bad_dialogue)))
        keep_v1_python: set[int] = set()
        v1_keys: set[str] = set()
        for line_number, raw in enumerate(converted_jsonl.open(encoding="utf-8"), 1):
            key = canonical_question(json.loads(raw)["prompt"])
            v1_keys.add(key)
            if key not in heldout_keys:
                keep_v1_python.add(line_number)
        removed_v1_python = sum(line not in keep_v1_python for line in base.python_source_lines)
        base.python, base.python_source_lines = (list(values) for values in zip(*(
            (row, line) for row, line in zip(base.python, base.python_source_lines)
            if line in keep_v1_python)))
        counts = Counter(base.counts)
        counts["new_swe_heldout_keys"] = len(heldout_keys)
        counts["new_swe_old_rows_quarantined"] = len(blocked_old_lines)
        counts["known_truncated_dialogue_rows_quarantined"] = removed_dialogue
        counts["new_swe_v1_python_rows_quarantined"] = removed_v1_python
        seen_refs: set[int] = set()
        for line_number, row in enumerate(swe_rows, 1):
            ref = row.get("source_ref", {})
            key = canonical_question(row["prompt"])
            if (key in eval_keys or not isinstance(ref.get("line"), int)
                    or ref.get("line") in seen_refs
                    or ref.get("file") != ORIGINAL_TRAIN_MEMBER
                    or row.get("full_text") != row["prompt"] + row["target"]):
                raise ValueError("SWE row changed provenance or overlaps an evaluation question")
            seen_refs.add(ref["line"])
            fences = row["target"].count("```python")
            if fences != 1 or len(PURE_V44_PYTHON_BLOCK.findall(row["target"])) != 1:
                raise ValueError("SWE row has no native executable Python fence")
            if key in heldout_keys:
                counts["swe_heldout_rows"] += 1
                continue
            if key in v1_keys:
                counts["swe_v1_duplicate_rows"] += 1
                continue
            prompt = PYTHON_MODE_CUE + row["prompt"]
            train_row = {"prompt": prompt, "target": row["target"],
                         "full_text": prompt + row["target"],
                         "target_segments": row["target_segments"],
                         "original_source": row["original_source"]}
            example = encode_example(train_row, vocab, kind="python", normalize=normalize)
            base.python.append(example)
            base.python_source_lines.append(CONVERTED_ROWS + line_number)
            counts["swe_train_rows"] += 1
            counts["swe_train_active_positions"] += example.active_positions
            counts["swe_train_family:" + example.source_family] += 1
    counts["converted_eligible_rows"] = len(base.python)
    counts["original_eligible_rows"] = len(base.old)
    base.counts = dict(counts)
    return base
