"""Knowledge-format snapshot checks, not JMMLU/JamC-QA model results."""
import copy
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_knowledge as knowledge
import frozen_eval


class KnowledgeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / "data"
        cls.support = cls.data / "knowledge_support"
        cls.manifest = json.loads((cls.data / "knowledge_manifest.json").read_text(encoding="utf-8"))
        cls.rows = {split: [json.loads(line) for line in (cls.data / meta["file"]).read_text(encoding="utf-8").splitlines()]
                    for split, meta in cls.manifest["splits"].items()}
        cls.all_rows = [row for rows in cls.rows.values() for row in rows]
        cls.vocab = set(json.loads((ROOT / "frozen/r6/base_metadata.json").read_text(encoding="utf-8"))["vocab"])
        cls.normalizer = knowledge.normalizer((cls.support / "qwenputer_controller_v30.py").read_text(encoding="utf-8"))
        cls.sources = {(record["source_split"], record["row"]["id"]): record["row"] for record in
                       (json.loads(line) for line in (cls.support / "used_source_rows.jsonl").read_text(encoding="utf-8").splitlines())}

    def test_file_hashes_and_measured_counts(self):
        self.assertEqual({split: len(rows) for split, rows in self.rows.items()},
                         {"train": 336, "dev": 4, "free_dev": 64})
        for split, meta in self.manifest["splits"].items():
            blob = (self.data / meta["file"]).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), meta["sha256"])
            self.assertEqual(len(blob), meta["size_bytes"])
            self.assertEqual(len({row["source_chain_id"] for row in self.rows[split]}), meta["underlying_chains"])
        for name, metadata in self.manifest["support_files"].items():
            blob = (self.data / name).read_bytes()
            self.assertEqual(hashlib.sha256(blob).hexdigest(), metadata["sha256"])

    def test_vocab_context_boundaries_and_native_observation_masks(self):
        for row in self.all_rows:
            knowledge.validate_row(row, self.vocab, self.normalizer)
            for (start, end), observation in zip(row["loss_mask_spans"], row["reference_observations"]):
                self.assertEqual(row["full_text"][start:end], " => " + observation + " ")
            self.assertTrue(row["full_text"].endswith(row["final_answer_line"]))
        expected = self.manifest["runtime_observation_protocol"]
        self.assertEqual(knowledge.protocol_source_audit(ROOT / "frozen"), expected)

    def test_mask_cannot_consume_search_action(self):
        row = copy.deepcopy(self.rows["train"][0])
        row["loss_mask_spans"][0][0] -= 1
        with self.assertRaisesRegex(ValueError, "runtime-injected span"):
            knowledge.validate_row(row, self.vocab, self.normalizer)

    def test_multihop_entities_are_observed_before_use(self):
        for row in self.all_rows:
            triples = knowledge.validate_chain({"search_queries": row["search_queries"], "answer": row["reference_answer_text"]},
                                               row["reference_observations"])
            for previous, following in zip(triples, triples[1:]):
                self.assertEqual(previous["value"], following["subject"])
                self.assertIn(following["subject"], previous["body"])
            self.assertEqual(triples[-1]["value"], row["reference_answer_text"])
        # A plausible-looking second result must not make a broken bridge pass.
        with self.assertRaisesRegex(ValueError, "Next query entity"):
            knowledge.validate_chain({"search_queries": ["A relation", "C relation"], "answer": "D"}, [
                knowledge.SEARCH_PREFIX + "A relation | AのrelationはB。]",
                knowledge.SEARCH_PREFIX + "C relation | CのrelationはD。]",
            ])

    def test_spaces_inside_relation_are_parsed_correctly(self):
        triple = knowledge.parse_observation("人物 名前 関係", knowledge.SEARCH_PREFIX + "人物 名前 関係 | 人物の名前 関係は値。]")
        self.assertEqual(triple, {"subject": "人物", "relation": "名前 関係", "value": "値", "body": "人物の名前 関係は値。"})

    def test_choice_mapping_and_same_relation_provenance(self):
        pairs = defaultdict(list)
        for split in ("train", "dev"):
            for row in self.rows[split]:
                pairs[(split, row["source_chain_id"])].append(row)
                correct = row["answer_index"]
                self.assertEqual(row["options"][correct], row["reference_answer_text"])
                self.assertEqual({item["final_relation"] for item in row["option_provenance"]}, {row["final_relation"]})
                self.assertEqual({item["source_split"] for item in row["option_provenance"]}, {split})
            for pair in [value for key, value in pairs.items() if key[0] == split]:
                self.assertEqual({row["answer_format"] for row in pair}, {"letter", "text"})
                self.assertEqual(pair[0]["options"], pair[1]["options"])
        self.assertEqual(Counter(row["answer_index"] for row in self.rows["train"]), {0: 84, 1: 84, 2: 84, 3: 84})
        corrupt = copy.deepcopy(self.rows["train"][0])
        corrupt["answer_index"] = (corrupt["answer_index"] + 1) % 4
        with self.assertRaisesRegex(ValueError, "Answer does not map"):
            knowledge.validate_row(corrupt, self.vocab, self.normalizer)
        corrupt = copy.deepcopy(self.rows["train"][0])
        corrupt["option_provenance"][0]["final_relation"] = "wrong relation"
        with self.assertRaisesRegex(ValueError, "same relation and split"):
            knowledge.validate_row(corrupt, self.vocab, self.normalizer)

    def test_no_cross_relation_fallback(self):
        class ForbiddenSearch:
            def search(self, query):
                raise AssertionError("An unrelated relation must never be searched as fallback")
        got = knowledge.select_distractors({"id": "wikidata_test_2", "answer": "A", "split": "train"},
                                          "country", {"city": {"B": [], "C": [], "D": []}}, ForbiddenSearch())
        self.assertEqual(got, [])

    def test_original_source_rows_and_split_ancestry(self):
        used_train = set()
        for row in self.all_rows:
            split = row["source_split"]
            source = self.sources[(split, row["source_problem_id"])]
            self.assertEqual(knowledge.canonical_row_sha(source), row["source_row_sha256"])
            self.assertEqual(source["split"], row["split"])
            self.assertEqual(knowledge.chain_id(source), row["source_chain_id"])
            if split == "train":
                used_train.add(knowledge.ancestor_id(source))
            for option in row["option_provenance"]:
                original = self.sources[(split, option["source_problem_id"])]
                self.assertEqual(knowledge.canonical_row_sha(original), option["source_row_sha256"])
                self.assertEqual(knowledge.source_license(original), option["source_license"])
                self.assertIn(option["source_license"], row["component_licenses"])
                if split == "train":
                    used_train.add(knowledge.ancestor_id(original))
        original_dev = [json.loads(line) for line in frozen_eval.source_path("public/data/dev_overlay.jsonl").read_text(encoding="utf-8").splitlines()]
        old_dev_ancestors = {knowledge.ancestor_id(row) for row in original_dev if row.get("search_queries")}
        self.assertFalse(used_train & old_dev_ancestors)

    def test_free_answer_dev_has_no_choice_hints_and_is_not_training(self):
        train_chains = {row["source_chain_id"] for row in self.rows["train"]}
        for row in self.rows["free_dev"]:
            self.assertEqual(row["answer_format"], "free_text")
            self.assertEqual(row["split"], "dev")
            self.assertEqual(row["options"], [])
            self.assertEqual(row["option_provenance"], [])
            self.assertIsNone(row["answer_index"])
            self.assertNotIn("選択肢:", row["prompt"])
            self.assertEqual(row["reference_final_answer"], row["reference_answer_text"])
            self.assertNotIn(row["source_chain_id"], train_chains)

    def test_original_evaluation_overlap_is_reported_honestly(self):
        observed = knowledge.frozen_overlap_audit(self.rows, ROOT / "frozen")
        self.assertEqual(observed, self.manifest["frozen_evaluation_overlap"])
        self.assertEqual(observed["new_full_prompt_normalized_overlap"], {"train": 0, "dev": 0, "free_dev": 0})
        self.assertEqual(observed["underlying_source_prompt_overlap"], {"train": 0, "dev": 4, "free_dev": 64})

    def test_actual_retriever_matches_every_used_query_and_option(self):
        path = self.support / "local_search_engine_v42.py"
        spec = importlib.util.spec_from_file_location("knowledge_test_native_search", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        requests = {}
        for row in self.all_rows:
            pairs = list(zip(row["search_queries"], row["reference_observations"]))
            pairs += [(option["query"], option["native_observation"]) for option in row["option_provenance"]]
            for query, observation in pairs:
                key = (row["source_split"], query)
                self.assertEqual(requests.get(key, observation), observation)
                requests[key] = observation
        engines = {}
        for split in ("train", "dev"):
            corpus = self.support / f"corpus_{split}.jsonl"
            self.assertTrue(corpus.is_file())
            engines[split] = module.LocalSearchEngineV42(corpus, geosearch_url="")
        for (split, query), expected in requests.items():
            self.assertEqual(engines[split].search(query), expected)


if __name__ == "__main__":
    unittest.main()
