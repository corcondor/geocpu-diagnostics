"""Build action/answer-format curricula from the existing TRAIN snapshot.

Knowledge claims here are snapshot-consistency checks, not independently
verified current-world facts. Original T1-T8 inputs and scoring are untouched.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import re
import zipfile

from build_curriculum import fingerprint, read_r6

ROOT = Path(__file__).resolve().parent
GROUP = "practical_knowledge_multistep_v1"
SEARCH_PREFIX = "[RESULT (src: local, date: 常駐): "


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_row_sha(row: dict) -> str:
    return sha(json.dumps(row, ensure_ascii=False, sort_keys=True).encode())


def dump(path: Path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def normalizer(source: str):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "_BASE_NORM_DICT" for target in node.targets):
            mapping = ast.literal_eval(node.value)
            for number in range(0xFF01, 0xFF5F):
                char = chr(number)
                if char not in ("（", "）", "！", "？", "：", "～"):
                    mapping[char] = chr(number - 0xFEE0)
            return str.maketrans(mapping)
    raise ValueError("Pinned normalizer not found")


def chain_id(row):
    if row.get("parent_chain_id"):
        return str(row["parent_chain_id"])
    return re.sub(r"_(?:chain|direct|context)$", "", row["id"])


def ancestor_id(row):
    value = chain_id(row)
    if value.startswith("wikidata_"):
        value = re.sub(r"_[0-3]$", "", value)
    return value


def source_family(row):
    if row["id"].startswith("jemhop"):
        return "jemhop"
    if row["id"].startswith("wikidata"):
        return "wikidata"
    raise ValueError("Unexpected source family: " + row["id"])


def source_license(row):
    if source_family(row) == "jemhop":
        return "CC BY-SA 4.0"
    return "CC0"


def select_chains(rows, limits):
    seen, chosen, answer_counts = set(), [], Counter()
    ordered = sorted((row for row in rows if row.get("id") and row.get("search_queries")),
                     key=lambda row: sha(row["id"].encode()))
    counts = Counter()
    for row in ordered:
        family, cid = source_family(row), chain_id(row)
        queries = row.get("search_queries", [])
        if not 2 <= len(queries) <= 3 or not row.get("answer") or cid in seen:
            continue
        if family == "jemhop" and row.get("family") != "chain":
            continue
        if counts[family] >= limits[family] or answer_counts[(family, row["answer"])] >= 8:
            continue
        seen.add(cid)
        chosen.append(row)
        counts[family] += 1
        answer_counts[(family, row["answer"])] += 1
    return chosen


def parse_observation(query: str, observation: str) -> dict:
    """Parse the retained snapshot's literal subject/predicate/object rendering.

    A predicate may contain spaces, so splitting only on the final space is
    insufficient. Every query whitespace boundary is checked against the body.
    """
    if not observation.startswith(SEARCH_PREFIX) or not observation.endswith("]"):
        raise ValueError("Unexpected native search observation")
    title, separator, body = observation[len(SEARCH_PREFIX):-1].partition(" | ")
    if not separator or title != query or not body.endswith("。"):
        raise ValueError("Query did not return its exact complete snapshot document")
    candidates = []
    for match in re.finditer(r"\s+", query):
        subject, predicate = query[:match.start()], query[match.end():]
        prefix = subject + "の" + predicate + "は"
        if body.startswith(prefix):
            value = body[len(prefix):-1]
            if value:
                candidates.append({"subject": subject, "relation": predicate,
                                   "value": value, "body": body})
    if len(candidates) != 1:
        raise ValueError("Cannot unambiguously parse the snapshot triple")
    return candidates[0]


def validate_chain(source: dict, observations: list[str]) -> list[dict]:
    queries = source["search_queries"]
    if len(queries) != len(observations) or not 2 <= len(queries) <= 3:
        raise ValueError("Unexpected multi-hop search count")
    triples = [parse_observation(query, observation) for query, observation in zip(queries, observations)]
    for previous, following in zip(triples, triples[1:]):
        if following["subject"] != previous["value"] or following["subject"] not in previous["body"]:
            raise ValueError("Next query entity does not equal the preceding observed value")
    if triples[-1]["value"] != str(source["answer"]):
        raise ValueError("Final observed value differs from the source label")
    return triples


class CachedRetriever:
    def __init__(self, engine):
        self.engine = engine
        self.cache = {}
        self.native_calls = 0

    def search(self, query):
        if query not in self.cache:
            self.cache[query] = self.engine.search(query)
            self.native_calls += 1
        return self.cache[query]


def relation_pools(source_rows, documents):
    """Provisional same-split relation pools; chosen options are natively replayed."""
    titles = defaultdict(list)
    for doc in documents:
        titles[doc["title"]].append(doc)
    result = defaultdict(lambda: defaultdict(list))
    for row in sorted((value for value in source_rows if value.get("id")), key=lambda value: sha(value["id"].encode())):
        if not row.get("answer") or not row.get("search_queries"):
            continue
        query = row["search_queries"][-1]
        docs = titles.get(query, [])
        if len({doc["text"] for doc in docs}) != 1:
            continue
        document = docs[0]
        observation = SEARCH_PREFIX + query + " | " + document["text"] + "]"
        try:
            triple = parse_observation(query, observation)
        except ValueError:
            continue
        if triple["value"] == str(row["answer"]):
            result[triple["relation"]][triple["value"]].append(row)
    return result


def option_record(source: dict, observation: str, relation: str):
    return {"value": str(source["answer"]), "source_problem_id": source["id"],
            "source_chain_id": chain_id(source), "source_ancestor_id": ancestor_id(source),
            "source_row_sha256": canonical_row_sha(source), "source_split": source["split"],
            "source_family": source_family(source), "source_license": source_license(source),
            "final_relation": relation, "query": source["search_queries"][-1],
            "native_observation": observation}


def select_distractors(source, relation, pools, engine):
    answer = str(source["answer"])
    candidates = [value for value in pools.get(relation, {})
                  if fingerprint(value) != fingerprint(answer) and answer not in value and value not in answer]
    rng = random.Random(int(sha(source["id"].encode())[:16], 16))
    rng.shuffle(candidates)
    chosen, seen = [], {fingerprint(answer)}
    for value in candidates:
        if fingerprint(value) in seen:
            continue
        for candidate in pools[relation][value]:
            observation = engine.search(candidate["search_queries"][-1])
            try:
                triple = parse_observation(candidate["search_queries"][-1], observation)
            except ValueError:
                continue
            if triple["relation"] != relation or triple["value"] != value or candidate["split"] != source["split"]:
                continue
            chosen.append(option_record(candidate, observation, relation))
            seen.add(fingerprint(value))
            break
        if len(chosen) == 3:
            return chosen
    return []  # Never fall back to a different relation or split.


def render_row(source, observations, triples, split, answer_format, option_provenance, correct_index):
    answer = str(source["answer"])
    question = source["prompt"].split("\n思考:", 1)[0]
    options = [item["value"] for item in option_provenance]
    if answer_format == "free_text":
        prompt = question + "\n必要な資料を検索して判断してください。最後の行を答え: に続けて値だけにしてください。\n思考:"
        final = answer
    else:
        instruction = ("最後の行を答え: に続けて選択肢の記号一文字にしてください。" if answer_format == "letter"
                       else "最後の行を答え: に続けて選択肢の本文だけにしてください。")
        prompt = (question + "\n選択肢:\n" + "\n".join(f"{letter}. {value}" for letter, value in zip("ABCD", options))
                  + "\n必要な資料を検索して判断してください。" + instruction + "\n思考:")
        final = "ABCD"[correct_index] if answer_format == "letter" else answer
    full, spans = prompt, []
    for hop, (query, observation) in enumerate(zip(source["search_queries"], observations)):
        full += " まず必要な資料を調べます。" if hop == 0 else " 得られた情報を使って次の資料を調べます。"
        full += f"<SEARCH {query}>"
        start = len(full)
        full += f" => {observation} "
        spans.append([start, len(full)])
    final_line = "答え: " + final
    full += " 資料を確認できました。\n" + final_line
    licenses = sorted({source_license(source)} | {item["source_license"] for item in option_provenance})
    return {
        "id": f"knowledge_{split}_{sha(source['id'].encode())[:16]}_{answer_format}",
        "source": source.get("source", "existing_source_snapshot"),
        "source_problem_id": source["id"], "source_chain_id": chain_id(source),
        "source_ancestor_id": ancestor_id(source), "source_split": source["split"],
        "source_snapshot_id": "GeoCPU_public_search_science_20261105_v4",
        "source_row_sha256": canonical_row_sha(source),
        "source_question": question, "source_prompt": source["prompt"],
        "source_provenance": source.get("provenance", "Deterministic Wikidata snapshot triple rendering"),
        "license": "; ".join(licenses), "component_licenses": licenses,
        "source_family": source_family(source), "family": "retrieval_then_option_selection" if options else "retrieval_then_free_answer",
        "group_id": "source_chain:" + chain_id(source), "split": split, "_v46_group": GROUP,
        "prompt": prompt, "target": full[len(prompt):], "full_text": full, "loss_mask_spans": spans,
        "answer_format": answer_format, "options": options, "answer_index": correct_index,
        "option_provenance": option_provenance, "final_relation": triples[-1]["relation"],
        "reference_final_answer": final, "final_answer_line": final_line, "reference_answer_text": answer,
        "search_queries": source["search_queries"], "reference_observations": observations,
        "observed_triples": triples, "tool_search_count": len(observations),
        "verification": "native_snapshot_replay; exact_entity_chaining; exact_final_value; same_relation_options",
        "novel_fact": False,
    }


def validate_row(row, vocabulary, table):
    prompt, target, full = row["prompt"], row["target"], row["full_text"]
    if full != prompt + target or not target:
        raise ValueError("Knowledge prompt/target boundary differs")
    if len(full) + 1 > 2048 or full.translate(table) != full or set(full) - vocabulary:
        raise ValueError("Knowledge tokenizer or context check failed")
    queries, observations = row["search_queries"], row["reference_observations"]
    triples = validate_chain({"search_queries": queries, "answer": row["reference_answer_text"]}, observations)
    if triples != row["observed_triples"]:
        raise ValueError("Recorded intermediate triples differ")
    spans = row["loss_mask_spans"]
    matches = list(re.finditer(r"<SEARCH ([^<>]+)>", target))
    if len(matches) != len(queries) or len(spans) != len(queries):
        raise ValueError("Search count or masks differ")
    previous = len(prompt)
    for match, query, observation, span in zip(matches, queries, observations, spans):
        start, end = span
        expected = " => " + observation + " "
        if (match.group(1) != query or start != len(prompt) + match.end() or
                not previous <= start < end <= len(full) or full[start:end] != expected):
            raise ValueError("Search observation is not exactly the runtime-injected span")
        previous = end
    if not full.endswith(row["final_answer_line"]) or row["final_answer_line"] != "答え: " + row["reference_final_answer"]:
        raise ValueError("Knowledge final answer line differs")
    if row["answer_format"] == "free_text":
        if row["options"] or row["option_provenance"] or row["answer_index"] is not None:
            raise ValueError("Free answer diagnostic contains options")
        if row["reference_final_answer"] != row["reference_answer_text"]:
            raise ValueError("Free answer label differs")
    else:
        options, correct = row["options"], row["answer_index"]
        if len(options) != 4 or len({fingerprint(value) for value in options}) != 4 or not 0 <= correct < 4:
            raise ValueError("Invalid four-option mapping")
        if options[correct] != row["reference_answer_text"]:
            raise ValueError("Answer does not map to the selected option")
        expected = "ABCD"[correct] if row["answer_format"] == "letter" else options[correct]
        if row["reference_final_answer"] != expected:
            raise ValueError("Requested answer format differs")
        if len(row["option_provenance"]) != 4:
            raise ValueError("Missing option provenance")
        for text, option in zip(options, row["option_provenance"]):
            triple = parse_observation(option["query"], option["native_observation"])
            if (triple["relation"] != row["final_relation"] or option["final_relation"] != row["final_relation"] or
                    triple["value"] != text or option["value"] != text or option["source_split"] != row["split"]):
                raise ValueError("An option is not grounded in the same relation and split")


def create_rows(selected, engine, split, pools, vocabulary, table):
    result, rejected = [], Counter()
    for source in selected:
        observations = [engine.search(query) for query in source["search_queries"]]
        try:
            triples = validate_chain(source, observations)
        except ValueError:
            rejected["retrieval_or_chain_inconsistent"] += 1
            continue
        relation = triples[-1]["relation"]
        choices = select_distractors(source, relation, pools, engine)
        if len(choices) != 3:
            rejected["fewer_than_three_same_relation_distractors"] += 1
            continue
        correct = (len(result) // 2) % 4
        choices.insert(correct, option_record(source, observations[-1], relation))
        pair = [render_row(source, observations, triples, split, format_, choices, correct) for format_ in ("letter", "text")]
        try:
            for row in pair:
                validate_row(row, vocabulary, table)
        except ValueError:
            rejected["tokenizer_context_or_schema"] += 1
            continue
        result.extend(pair)
    return result, dict(rejected)


def create_free_rows(selected, engine, vocabulary, table):
    rows = []
    for source in selected:
        observations = [engine.search(query) for query in source["search_queries"]]
        triples = validate_chain(source, observations)
        row = render_row(source, observations, triples, "dev", "free_text", [], None)
        validate_row(row, vocabulary, table)
        rows.append(row)
    return rows


def frozen_overlap_audit(rows_by_split, frozen_root):
    import frozen_eval
    records = list(frozen_eval.iter_heldout_prompts(frozen_root))
    old_prompts = {fingerprint(record["prompt"]) for record in records}
    result = {
        "frozen_prompt_views": len(records),
        "registry_sha256": sha((frozen_root / "registry.json").read_bytes()),
        "source_inventory_sha256": sha((frozen_root / "source_files.json").read_bytes()),
        "new_full_prompt_normalized_overlap": {}, "underlying_source_prompt_overlap": {},
        "normalization": "NFKC + collapsed whitespace + casefold",
    }
    for split, rows in rows_by_split.items():
        overlap = [row["id"] for row in rows if fingerprint(row["prompt"]) in old_prompts]
        origin_overlap = [row["id"] for row in rows if fingerprint(row["source_prompt"]) in old_prompts
                          or fingerprint(row["source_question"]) in old_prompts]
        result["new_full_prompt_normalized_overlap"][split] = len(overlap)
        result["underlying_source_prompt_overlap"][split] = len(origin_overlap)
        if overlap or split == "train" and origin_overlap:
            raise ValueError("New training or wrapped prompt overlaps frozen evaluation: " + str((overlap + origin_overlap)[:3]))
    result["note"] = "Development views intentionally reuse old development chains; they are not independent new facts. Exact prompt matching does not exclude all semantic overlap."
    return result


def protocol_source_audit(frozen_root):
    import frozen_eval
    path = frozen_eval.source_path("base/code/integrated_model_v44fix.py", frozen_root)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    matched = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "inject" or len(node.args) < 2:
            continue
        joined = node.args[1]
        if not isinstance(joined, ast.JoinedStr) or len(joined.values) != 3:
            continue
        a, value, b = joined.values
        if (isinstance(a, ast.Constant) and a.value == " => " and isinstance(b, ast.Constant) and b.value == " "
                and isinstance(value, ast.FormattedValue) and isinstance(value.value, ast.Name) and value.value.id == "res_snippet"):
            if not any(kw.arg == "normalize" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords):
                raise ValueError("Runtime search normalization changed")
            matched += 1
    if matched != 1:
        raise ValueError("Native search injection formula changed")
    return {"source": "base/code/integrated_model_v44fix.py", "sha256": sha(path.read_bytes()),
            "injection_template": " => {native_search_observation} ", "matching_ast_sites": matched}


def build(r6_bundle: Path, output: Path, frozen_root: Path | None = None):
    if output.exists() and any(output.glob("knowledge_*.json*")):
        raise FileExistsError("Use a new output directory; existing generated data is not overwritten")
    frozen_root = frozen_root or ROOT / "frozen"
    r6_blob = read_r6(r6_bundle)
    output.mkdir(parents=True, exist_ok=True)
    support = output / "knowledge_support"
    support.mkdir(exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(r6_blob)) as r6:
        meta = json.loads(r6.read("base_metadata.json"))
        normalizer_blob = r6.read("qwenputer_controller_v30.py")
        table = normalizer(normalizer_blob.decode())
        (support / "qwenputer_controller_v30.py").write_bytes(normalizer_blob)
        engine_path = support / "local_search_engine_v42.py"
        engine_path.write_bytes(r6.read("local_search_engine_v42.py"))
        public_blob = r6.read("GeoCPU_public_search_science_20261105_v4.zip")
    with zipfile.ZipFile(io.BytesIO(public_blob)) as public:
        groups = {split: [json.loads(line) for line in public.read(f"data/{split}_overlay.jsonl").splitlines()]
                  for split in ("train", "dev")}
        corpora = {}
        for split in ("train", "dev"):
            corpus = support / f"corpus_{split}.jsonl"
            corpus.write_bytes(public.read(f"retrieval/combined_{split}_wikidata_first.jsonl"))
            corpora[split] = corpus
        for filename in ("sources/jemhop_manifest.json", "sources/wikidata_manifest.json"):
            (support / Path(filename).name).write_bytes(public.read(filename))
    module_spec = importlib.util.spec_from_file_location("knowledge_exact_retriever", engine_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    native_engines = {split: module.LocalSearchEngineV42(corpora[split], geosearch_url="") for split in groups}
    engines = {split: CachedRetriever(engine) for split, engine in native_engines.items()}
    train_chains = {chain_id(row) for row in groups["train"] if row.get("search_queries")}
    dev_chains = {chain_id(row) for row in groups["dev"] if row.get("search_queries")}
    train_ancestors = {ancestor_id(row) for row in groups["train"] if row.get("search_queries")}
    dev_ancestors = {ancestor_id(row) for row in groups["dev"] if row.get("search_queries")}
    if train_chains & dev_chains or train_ancestors & dev_ancestors:
        raise ValueError("Underlying train/dev chain ancestry overlaps")
    report = {"version": 2, "input_sha256": sha(r6_blob), "public_bundle_sha256": sha(public_blob),
              "model_updates": 0, "gpu_model_evaluations": 0,
              "source_train_dev_chain_overlap": 0, "source_train_dev_ancestor_overlap": 0,
              "splits": {}, "scope": "Action/format views of existing source splits, not new knowledge coverage.",
              "fact_verification": "Snapshot consistency only; upstream facts were not re-queried individually.",
              "distractor_policy": "Exactly the same final relation, same original split, distinct normalized literal values, actual retriever support; no mixed-relation fallback.",
              "runtime_observation_protocol": protocol_source_audit(frozen_root)}
    rows_by_split, used_sources = {}, {}
    for split in ("train", "dev"):
        selected = select_chains(groups[split], {"jemhop": 200 if split == "train" else 32,
                                                "wikidata": 128 if split == "train" else 32})
        pools = relation_pools(groups[split], native_engines[split].documents)
        rows, rejected = create_rows(selected, engines[split], split, pools, set(meta["vocab"]), table)
        if not rows:
            raise ValueError("No eligible same-relation four-choice rows")
        rows_by_split[split] = rows
        report["splits"][split] = {"selected_source_chains": len(selected), "rejected_chains": rejected}
        if split == "dev":
            rows_by_split["free_dev"] = create_free_rows(selected, engines[split], set(meta["vocab"]), table)
            report["splits"]["free_dev"] = {"selected_source_chains": len(selected), "rejected_chains": {}}
    source_lookup = {split: {row["id"]: row for row in rows if row.get("id")} for split, rows in groups.items()}
    transition_counts, native_replay_calls = {}, 0
    for output_split, rows in rows_by_split.items():
        source_split = "train" if output_split == "train" else "dev"
        seen_chains, transitions, final_matches = set(), 0, 0
        fresh = module.LocalSearchEngineV42(corpora[source_split], geosearch_url="")
        for row in rows:
            validate_row(row, set(meta["vocab"]), table)
            source = source_lookup[source_split][row["source_problem_id"]]
            if source["split"] != source_split or canonical_row_sha(source) != row["source_row_sha256"]:
                raise ValueError("Source identity or original split differs")
            used_sources[(source_split, source["id"])] = source
            for option in row["option_provenance"]:
                origin = source_lookup[source_split][option["source_problem_id"]]
                if canonical_row_sha(origin) != option["source_row_sha256"]:
                    raise ValueError("Option source row differs")
                used_sources[(source_split, origin["id"])] = origin
            if row["source_chain_id"] not in seen_chains:
                actual = [fresh.search(query) for query in row["search_queries"]]
                native_replay_calls += len(actual)
                if actual != row["reference_observations"]:
                    raise ValueError("Independent native retrieval replay differs")
                triples = validate_chain(source, actual)
                transitions += len(triples) - 1
                final_matches += 1
                seen_chains.add(row["source_chain_id"])
        transition_counts[output_split] = {"unique_chains_replayed": len(seen_chains),
                                          "verified_query_entity_transitions": transitions,
                                          "exact_final_value_matches": final_matches}
    train_used_ancestors = {ancestor_id(row) for (split, _), row in used_sources.items() if split == "train"}
    if train_used_ancestors & dev_ancestors:
        raise ValueError("Added train question or distractor derives from old dev ancestry")
    report["added_train_used_source_ancestor_overlap_with_all_old_dev"] = 0
    report["frozen_evaluation_overlap"] = frozen_overlap_audit(rows_by_split, frozen_root)
    report["native_replay"] = {"query_calls": native_replay_calls, "by_split": transition_counts,
                               "generation_unique_query_calls": {key: engine.native_calls for key, engine in engines.items()}}
    for split, rows in rows_by_split.items():
        path = output / f"knowledge_{split}.jsonl"
        dump(path, rows)
        report["splits"][split].update({
            "file": path.name, "rows": len(rows), "underlying_chains": len({row["source_chain_id"] for row in rows}),
            "answer_values": len({row["reference_answer_text"] for row in rows}),
            "source_families": dict(Counter(row["source_family"] for row in rows)),
            "source_family_chains": dict(Counter(source_family(source_lookup["train" if split == "train" else "dev"][row_id])
                                                 for row_id in {row["source_problem_id"] for row in rows})),
            "answer_positions": dict(Counter(row["answer_index"] for row in rows if row["answer_index"] is not None)),
            "sha256": sha(path.read_bytes()), "size_bytes": path.stat().st_size,
            "max_characters_plus_eos": max(len(row["full_text"]) + 1 for row in rows),
            "unknown_characters": [], "normalization_changes": 0,
            "context_tokens_including_eos": sum(len(row["full_text"]) + 1 for row in rows),
            "supervised_target_tokens_including_eos": sum(len(row["target"]) + 1 -
                sum(end - start for start, end in row["loss_mask_spans"]) for row in rows),
        })
    source_file = support / "used_source_rows.jsonl"
    dump(source_file, [{"source_split": split, "row": row} for (split, _), row in sorted(used_sources.items())])
    report["source_records"] = {"file": "knowledge_support/used_source_rows.jsonl", "rows": len(used_sources),
                                "sha256": sha(source_file.read_bytes())}
    report["support_files"] = {str(path.relative_to(output)): {"sha256": sha(path.read_bytes()), "size_bytes": path.stat().st_size}
                               for path in sorted(support.iterdir()) if path.is_file()}
    report["limitations"] = [
        "Same-relation distractors remove easy cross-type options but do not prove absence of aliases or all semantic ambiguity.",
        "The four-choice dev set is very small because its Wikidata country/capital pools each contain only three distinct values.",
        "Free-answer dev reuses 64 existing development chains; it is a new view, not 64 independent new fact chains.",
        "Train/source-dev ancestry is checked within this public bundle. Other historical training sources may contain the same facts.",
        "JEMHopQA statements are rendered annotation triples, not newly verified Wikipedia quotations. Some facts may be outdated.",
        "No model capability or GPU training has been evaluated by this CPU data build.",
    ]
    (output / "knowledge_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, default=ROOT / "frozen")
    args = parser.parse_args()
    print(json.dumps(build(args.r6_bundle, args.output, args.frozen_root), ensure_ascii=False, indent=2))
