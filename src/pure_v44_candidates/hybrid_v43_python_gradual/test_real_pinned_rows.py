"""Small read-only real-data checks, without importing the 247M model or torch."""
import json
from pathlib import Path
import unittest
import zipfile

from hybrid_data import encode_example
from source_pools import PYTHON_MODE_CUE, _original_normalizer


ZIP = Path(r"C:\Users\81808\.openclaw\reports\jmmlu_v1\ag_fused_fullqa_training_v1\v43_v44fix_paired_eval_assets\v43_source.zip")
CONVERTED = (Path(__file__).parents[1] / "v43_alu_to_python" /
             "provenance_full_final" / "converted_with_source_refs.jsonl")


@unittest.skipUnless(ZIP.is_file() and CONVERTED.is_file(), "Pinned local v43 assets unavailable")
class RealPinnedRowsTests(unittest.TestCase):
    def test_original_v43_rows_have_identical_old_loader_mask(self):
        with zipfile.ZipFile(ZIP) as archive:
            meta = json.loads(archive.read("base_metadata.json"))
            vocab = meta["vocab"]
            normalize = _original_normalizer(archive.read("qwenputer_controller_v30.py").decode())
            with archive.open("data/train_v43_knowledge_unified.jsonl") as stream:
                rows = [json.loads(next(stream)) for _ in range(12)]
        for row in rows:
            full = normalize(row.get("full_text") or row.get("prompt", "") + row.get("target", ""))
            prompt = normalize(row.get("prompt", ""))
            expected_tokens = [vocab[c] for c in full if c in vocab]
            expected_prompt = sum(c in vocab for c in prompt)
            if len(expected_tokens) < 2048:
                expected_tokens.append(3)
            else:
                expected_tokens = expected_tokens[:2048]
            expected_prompt = min(expected_prompt, len(expected_tokens)-1)
            expected_mask = [False] * (len(expected_tokens)-1)
            expected_mask[max(0, expected_prompt-1):] = [True] * (
                len(expected_mask) - max(0, expected_prompt-1))
            encoded = encode_example(row, vocab, kind="v43", normalize=normalize)
            self.assertEqual(encoded.tokens, tuple(expected_tokens))
            self.assertEqual(encoded.prompt_length, expected_prompt)
            self.assertEqual(encoded.target_mask(), tuple(expected_mask))
            self.assertTrue(encoded.target_mask()[expected_prompt-1])
            if len(expected_tokens) < 2048:
                self.assertEqual(encoded.tokens[-1], 3)

    def test_converted_original_source_and_mask_on_real_row(self):
        with zipfile.ZipFile(ZIP) as archive:
            meta = json.loads(archive.read("base_metadata.json"))
            normalize = _original_normalizer(archive.read("qwenputer_controller_v30.py").decode())
            with archive.open("data/train_v43_knowledge_unified.jsonl") as stream:
                first = json.loads(next(stream))
                second_raw = next(stream)
                second = json.loads(second_raw)
        with CONVERTED.open(encoding="utf-8") as stream:
            converted = json.loads(next(stream))
        import hashlib
        self.assertEqual(converted["source_ref"]["line"], 2)
        self.assertEqual(converted["source_ref"]["raw_sha256"], hashlib.sha256(second_raw).hexdigest())
        self.assertEqual(converted["prompt"], second["prompt"])
        encoded = encode_example(converted, meta["vocab"], kind="python", normalize=normalize)
        self.assertEqual(encoded.tokens[-1], 3)
        self.assertTrue(encoded.target_mask()[encoded.prompt_length-1])
        self.assertTrue(encoded.target_mask()[-1])
        self.assertGreater(sum(end-start for start,end in encoded.tool_target_ranges), 0)
        for start, end in encoded.tool_target_ranges:
            self.assertFalse(any(encoded.target_mask()[start:end]))
        training_prompt = PYTHON_MODE_CUE + converted["prompt"]
        self.assertFalse([char for char in PYTHON_MODE_CUE if char not in meta["vocab"]])
        cued = dict(converted, prompt=training_prompt,
                    full_text=training_prompt+converted["target"])
        cued_encoded = encode_example(cued, meta["vocab"], kind="python", normalize=normalize)
        self.assertEqual(cued_encoded.active_positions, encoded.active_positions)
        self.assertTrue(cued_encoded.target_mask()[cued_encoded.prompt_length-1])
        self.assertTrue(cued_encoded.target_mask()[-1])


if __name__ == "__main__":
    unittest.main()
