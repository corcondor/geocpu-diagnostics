"""Read-only, low-memory full scan of the 24,836 converted mask profiles."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import zipfile

from hybrid_data import encode_example
from source_pools import (PINNED_CONVERTED_SHA, PINNED_META_SHA,
                          PINNED_V43_ZIP_SHA, PYTHON_MODE_CUE,
                          _original_normalizer, file_sha)


def scan(v43_zip: Path, converted_path: Path) -> dict:
    if file_sha(v43_zip) != PINNED_V43_ZIP_SHA or file_sha(converted_path) != PINNED_CONVERTED_SHA:
        raise ValueError("Pinned source differs")
    with zipfile.ZipFile(v43_zip) as archive:
        metadata = archive.read("base_metadata.json")
        if sha256(metadata).hexdigest() != PINNED_META_SHA:
            raise ValueError("Pinned metadata differs")
        vocab = json.loads(metadata)["vocab"]
        normalize = _original_normalizer(archive.read("qwenputer_controller_v30.py").decode())
    counts = Counter()
    maximum = 0
    with converted_path.open(encoding="utf-8") as stream:
        for line in stream:
            original = json.loads(line)
            prompt = PYTHON_MODE_CUE + original["prompt"]
            row = {"prompt": prompt, "target": original["target"],
                   "full_text": prompt + original["target"],
                   "target_segments": original["target_segments"],
                   "original_source": original["original_source"]}
            encoded = encode_example(row, vocab, kind="python", normalize=normalize)
            counts["rows"] += 1
            counts["assistant_plus_eos_active_positions"] += encoded.active_positions
            counts["tool_observation_masked_positions"] += sum(
                end-start for start, end in encoded.tool_target_ranges)
            counts["family:"+encoded.source_family] += 1
            counts["true_eos_active_rows"] += int(encoded.target_mask()[-1] and encoded.tokens[-1] == 3)
            counts["first_answer_token_active_rows"] += int(encoded.target_mask()[encoded.prompt_length-1])
            maximum = max(maximum, encoded.sequence_length)
    if counts["rows"] != 24_836 or counts["true_eos_active_rows"] != counts["rows"] or counts["first_answer_token_active_rows"] != counts["rows"]:
        raise ValueError("Incomplete converted full-scan target closure")
    return {"schema": "geocpu_hybrid_converted_mask_readonly_v1",
            "v43_zip_sha256": PINNED_V43_ZIP_SHA,
            "converted_sha256": PINNED_CONVERTED_SHA,
            "mode_cue": PYTHON_MODE_CUE,
            "maximum_sequence_length": maximum,
            "counts": dict(counts),
            "training_updates": 0}


def main():
    root = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1")
    v43 = root / "v43_v44fix_paired_eval_assets" / "v43_source.zip"
    converted = (Path(__file__).parents[1] / "v43_alu_to_python" / "provenance_full_final"
                 / "converted_with_source_refs.jsonl")
    result = scan(v43, converted)
    output = Path(__file__).with_name("converted_mask_scan_result.json")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
