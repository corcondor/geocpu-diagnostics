"""Streaming read-only census of all pinned v43 training rows by source."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
import zipfile

from source_pools import (CONVERTED_ROWS, ORIGINAL_ROWS, ORIGINAL_TRAIN_MEMBER,
                          PINNED_CONVERTED_SHA, PINNED_V43_ZIP_SHA,
                          KNOWN_BAD_GSM_LINE, KNOWN_BAD_GSM_RAW_SHA,
                          _eval_keys, canonical_question, file_sha)


def scan(v43_zip: Path, converted: Path) -> dict:
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA or file_sha(converted) != PINNED_CONVERTED_SHA:
        raise ValueError("Pinned original or converted data differs")
    withheld = set()
    with converted.open(encoding="utf-8") as stream:
        count = 0
        for line in stream:
            row = json.loads(line)
            key = canonical_question(row["prompt"])
            if int(sha256(key.encode()).hexdigest()[:12], 16) % 100 >= 90:
                withheld.add(key)
            count += 1
    if count != CONVERTED_ROWS:
        raise ValueError("Converted corpus is incomplete")
    counts = Counter()
    with zipfile.ZipFile(v43_zip) as archive:
        eval_keys = _eval_keys(archive)
        with archive.open(ORIGINAL_TRAIN_MEMBER) as stream:
            for line_no, raw in enumerate(stream, 1):
                row = json.loads(raw)
                counts["all_original_rows"] += 1
                family = str(row.get("source", "unclassified"))
                counts["all_source:"+family] += 1
                if line_no == KNOWN_BAD_GSM_LINE:
                    if (sha256(raw).hexdigest() != KNOWN_BAD_GSM_RAW_SHA
                            or family != "GSM8K_Japanese_Full"):
                        raise ValueError("Pinned incorrect-label row identity changed")
                    counts["known_incorrect_gsm_label_quarantined"] += 1
                    continue
                key = canonical_question(row.get("prompt", ""))
                if key in eval_keys:
                    counts["evaluation_family_quarantined"] += 1
                    continue
                if key in withheld:
                    counts["python_dev_test_family_quarantined"] += 1
                    continue
                counts["retained_original_rows"] += 1
                counts["retained_source:"+family] += 1
                if re.search(r"[ぁ-んァ-ン一-龯]", row.get("target", "")):
                    counts["retained_japanese_character_target_rows"] += 1
                if re.search(r"<(?:(?:IMP|IFF|AND|OR|NOT|LAND|LOR|LNOT)|(?:EQ|NE|LT|GT))\b", row.get("target", "")):
                    counts["retained_formal_logic_opcode_target_rows"] += 1
    if counts["all_original_rows"] != ORIGINAL_ROWS:
        raise ValueError("Original corpus is incomplete")
    return {"schema": "geocpu_v43_original_source_census_readonly_v2",
            "v43_zip_sha256": PINNED_V43_ZIP_SHA,
            "converted_sha256": PINNED_CONVERTED_SHA,
            "counts": dict(counts), "training_updates": 0}


def main():
    root = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1")
    v43 = root / "v43_v44fix_paired_eval_assets" / "v43_source.zip"
    converted = Path(__file__).parents[1] / "v43_alu_to_python" / "provenance_full_final" / "converted_with_source_refs.jsonl"
    result = scan(v43, converted)
    output = Path(__file__).with_name("original_source_census_result_v2.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"all": result["counts"]["all_original_rows"],
                      "retained": result["counts"]["retained_original_rows"],
                      "japanese": result["counts"]["retained_japanese_character_target_rows"],
                      "logic": result["counts"]["retained_formal_logic_opcode_target_rows"]}))


if __name__ == "__main__":
    main()
