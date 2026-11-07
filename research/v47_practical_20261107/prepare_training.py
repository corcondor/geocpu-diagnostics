"""CPU preparation for an isolated practical-curriculum continuation.

The pinned 30,500-step dataset is copied byte for byte, then new training rows
are appended. Evaluation inputs are read only. Runtime edits are explicit,
anchor-checked changes to a copied, SHA-256-pinned original source.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import difflib
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
from typing import List


PARENT_TRAIN_SHA = "358f96521122606ce398eb6f5f30f51e71048728e52e6b8ee9f55de854fcd5fc"
PARENT_ROWS = 422500
ORIGINAL_RUNTIME_SHA = "548a5c509df3629a099acb97f31fe9f84155a46202008c351f45f57d2ad9eb08"
NEW_GROUP = "practical_multistep_v1"
ACCEPTED_GROUPS = {"practical_python_multistep_v1", "practical_knowledge_multistep_v1"}
VARIANTS = ("multistep", "answer_only", "replay_only")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1 << 20), b""):
            h.update(data)
    return h.hexdigest()


def require_sha(path: Path, expected: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise ValueError(f"missing or changed pinned input: {path}")


def rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row {path}:{n}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row {path}:{n}")
            yield value


def normalization_table(source: Path) -> dict:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found = [node.value for node in tree.body if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_BASE_NORM_DICT" for t in node.targets)]
    if len(found) != 1:
        raise ValueError("original normalizer literal changed")
    mapping = ast.literal_eval(found[0])
    for number in range(0xFF01, 0xFF5F):
        char = chr(number)
        if char not in ("（", "）", "！", "？", "：", "～"):
            mapping[char] = chr(number - 0xFEE0)
    return str.maketrans(mapping)


def _prompt_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("prompt", "question", "original_prompt") and isinstance(item, str):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _prompt_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _prompt_strings(item)


def heldout_sets(paths: list[Path], normalize: dict) -> tuple[set[str], set[str], dict]:
    prompts: set[str] = set()
    ids: set[str] = set()
    hashes = {}
    for path in paths:
        hashes[str(path)] = sha(path)
        data = list(rows(path)) if path.suffix == ".jsonl" else json.loads(path.read_text(encoding="utf-8"))
        prompts.update(x.translate(normalize) for x in _prompt_strings(data))
        records = data if isinstance(data, list) else data.get("cases", [])
        for row in records:
            for key in ("source_problem_id", "source_chain_id"):
                if row.get(key):
                    ids.add(str(row[key]))
    return prompts, ids, hashes


def validate_new_row(row: dict, vocab: set[str], normalize: dict,
                     heldout_prompts: set[str], heldout_ids: set[str]) -> None:
    prompt, full = row.get("prompt"), row.get("full_text")
    if not isinstance(prompt, str) or not isinstance(full, str) or not full.startswith(prompt) or len(full) <= len(prompt):
        raise ValueError("new row has an invalid prompt/target boundary")
    if len(full) + 1 > 2048 or full.translate(normalize) != full or set(full) - vocab:
        raise ValueError(f"new row changes under the original tokenizer or exceeds 2048: {row.get('id')}")
    if row.get("split") != "train" or not row.get("source") or not row.get("license"):
        raise ValueError("new row requires train split, source and license")
    if row.get("_v46_group") not in ACCEPTED_GROUPS or not row.get("id"):
        raise ValueError("unknown new curriculum group or missing ID")
    if prompt in heldout_prompts or any(str(row.get(k, "")) in heldout_ids for k in ("source_problem_id", "source_chain_id")):
        raise ValueError(f"new training row overlaps heldout: {row['id']}")
    spans = row.get("loss_mask_spans", [])
    previous = len(prompt)
    for span in spans:
        if not isinstance(span, list) or len(span) != 2 or any(type(x) is not int for x in span):
            raise ValueError("mask spans must be integer offset pairs")
        a, b = span
        if not previous <= a < b <= len(full):
            raise ValueError("mask overlaps, leaves the target region, or is out of bounds")
        observed = full[a:b].strip()
        if observed.startswith("=>"):
            observed = observed[2:].strip()
        if not (observed.startswith("```output\n") and observed.endswith("```")
                or observed.startswith("[RESULT") and observed.endswith("]")):
            raise ValueError("mask includes text other than a tool observation")
        previous = b
    # Every runtime-inserted observation must be covered; model-authored
    # actions and final answers remain supervised.
    patterns = (r"(?ms)^```output[^\n]*\n.*?^```[ \t]*(?:\n|$)", r"\[RESULT[^\n]*?\]")
    for pattern in patterns:
        for match in re.finditer(pattern, full[len(prompt):]):
            a, b = len(prompt) + match.start(), len(prompt) + match.end()
            # A fenced match may include the following newline not emitted in
            # the source mask; the fence itself must be completely masked.
            while b > a and full[b - 1] in "\r\n":
                b -= 1
            if not any(s <= a and b <= e for s, e in spans):
                raise ValueError("unmasked runtime observation")
    final = row.get("final_answer_line")
    if final is None:
        reference = row.get("reference_final_answer")
        if reference is None:
            raise ValueError("answer-only control needs an explicit reference answer")
        final = "答え: " + str(reference)
    if not isinstance(final, str) or not final or any(marker in final for marker in ("```", "[RESULT", "<SEARCH")):
        raise ValueError("final answer line is not independent of tool protocol")


def manifest_train(path: Path, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    item = files.get("train")
    # The knowledge builder keys files by basename, the Python builder by split.
    if item is None:
        item = files.get(path.name)
    if item is None:
        item = manifest.get("splits", {}).get("train")
    if not isinstance(item, dict):
        raise ValueError(f"manifest has no train entry: {manifest_path}")
    expected = item.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("training manifest lacks SHA-256")
    require_sha(path, expected)
    return item


def prepare_dataset(parent: Path, additions: list[tuple[Path, Path]], *, meta: Path,
                    normalizer_source: Path, heldouts: list[Path], output: Path,
                    manifest_output: Path, variant: str = "multistep",
                    expected_parent_sha: str = PARENT_TRAIN_SHA,
                    expected_parent_rows: int = PARENT_ROWS) -> dict:
    if variant not in VARIANTS:
        raise ValueError("unknown training variant")
    if output.exists() or manifest_output.exists() or output.resolve() == parent.resolve():
        raise FileExistsError("training preparation needs new output files")
    require_sha(parent, expected_parent_sha)
    normalize = normalization_table(normalizer_source)
    vocab = set(json.loads(meta.read_text(encoding="utf-8"))["vocab"])
    heldout_prompts, heldout_ids, heldout_hashes = heldout_sets(heldouts, normalize)
    parent_prompts = set()
    n = 0
    for row in rows(parent):
        n += 1
        parent_prompts.add(row["prompt"].translate(normalize))
    if n != expected_parent_rows:
        raise ValueError("parent row count changed")
    with parent.open("rb") as f:
        f.seek(-1, 2)
        if f.read(1) != b"\n":
            raise ValueError("parent JSONL lacks a final newline")
    accepted = []
    seen_ids = set()
    input_hashes = {}
    counts = Counter()
    for data, manifest_path in additions:
        item = manifest_train(data, manifest_path)
        input_hashes[str(data)] = item["sha256"]
        new_rows = list(rows(data))
        if item.get("rows") is not None and len(new_rows) != item["rows"]:
            raise ValueError("new curriculum row count changed")
        for row in new_rows:
            validate_new_row(row, vocab, normalize, heldout_prompts, heldout_ids)
            if row["id"] in seen_ids or row["prompt"] in parent_prompts:
                raise ValueError(f"duplicate new ID/prompt or parent prompt: {row['id']}")
            seen_ids.add(row["id"])
            parent_prompts.add(row["prompt"])
            if variant == "replay_only":
                continue
            row = dict(row)
            row["_practical_source"] = row["_v46_group"]
            row["_v46_group"] = NEW_GROUP
            row["training_variant"] = variant
            if variant == "answer_only":
                final = row.get("final_answer_line") or "答え: " + str(row["reference_final_answer"])
                row["target"] = "\n" + final.strip()
                row["full_text"] = row["prompt"] + row["target"]
                row["loss_mask_spans"] = []
            counts[row["_practical_source"]] += 1
            accepted.append(row)
    output.parent.mkdir(parents=True, exist_ok=True)
    with parent.open("rb") as old, output.open("xb") as new:
        shutil.copyfileobj(old, new, length=1 << 20)
        for row in accepted:
            new.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    report = {"parent_sha256": expected_parent_sha, "parent_rows_preserved_byte_exact": n,
              "new_rows": len(accepted), "new_group": NEW_GROUP, "new_sources": dict(counts),
              "variant": variant, "input_files_sha256": input_hashes,
              "heldout_files_sha256": heldout_hashes,
              "output_sha256": sha(output), "output_rows": n + len(accepted),
              "model_updates": 0, "new_teacher_scope": "all replay rows, excluding practical_multistep_v1",
              "limitation": "exact prompt/source-ID exclusions do not prove absence of semantic paraphrases"}
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    with manifest_output.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return report


def _replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"pinned runtime anchor count changed: {old[:90]!r}")
    return text.replace(old, new, 1)


def patched_runtime(original: str) -> str:
    """The original optimizer, losses, architecture and OOM retry remain intact."""
    if hashlib.sha256(original.encode()).hexdigest() != ORIGINAL_RUNTIME_SHA:
        raise ValueError("runtime must be the pinned 30500 source")
    out = original.replace("python_tir_30500", NEW_GROUP)
    out = _replace_once(out, "        self.groups = []\n", "        self.groups = []\n        self.origins = []\n        self.supervised_counts = []\n")
    out = _replace_once(out,
        '                is_v43 = (item.get("_v46_group") == "original_parent"\n'
        '                          and not str(item.get("source", "")).startswith("bridge_py:"))',
        f'                is_v43 = str(item.get("_v46_group", "legacy")) != "{NEW_GROUP}"')
    out = _replace_once(out,
        '                    self.groups.append(str(item.get("_v46_group", "legacy")))',
        '                    self.groups.append(str(item.get("_v46_group", "legacy")))\n'
        '                    self.origins.append(str(item.get("_practical_source", item.get("_v46_group", "legacy"))))\n'
        '                    self.supervised_counts.append(practical_supervised_count(chunk, pl, local_spans))')
    for group, indices in (("numina_tir", "numina_indices"), (NEW_GROUP, "new_indices")):
        anchor = f'                dataset.groups.extend(["{group}"] * len(originals))'
        out = _replace_once(out, anchor, anchor +
            f'\n                dataset.origins.extend([dataset.origins[i] for i in {indices}])'
            f'\n                dataset.supervised_counts.extend([dataset.supervised_counts[i] for i in {indices}])')
    out = _replace_once(out, '    args = parser.parse_args()\n',
        '    parser.add_argument("--practical_variant", choices=("multistep", "answer_only", "replay_only"), default="multistep")\n'
        '    parser.add_argument("--telemetry_jsonl", required=True)\n'
        '    parser.add_argument("--min_replay_share", type=float, default=0.75)\n'
        '    args = parser.parse_args()\n'
        '    if not 0.75 <= args.min_replay_share <= 1.0:\n'
        '        raise RuntimeError("replay supervised-token floor must be at least 0.75")\n')
    out = _replace_once(out,
        f'            if not new_indices:\n                raise RuntimeError("explicit new dataset phase has no {NEW_GROUP} rows")',
        f'            if not new_indices and args.practical_variant != "replay_only":\n                raise RuntimeError("explicit new dataset phase has no {NEW_GROUP} rows")\n'
        '            if args.practical_variant == "replay_only" and (new_indices or args.new_curriculum_repeat != 1 or args.new_group_min_exposures != 0):\n'
        '                raise RuntimeError("replay-only requires no new rows, repeat=1 and exposure floor=0")')
    out = _replace_once(out,
        '                if args.stop_at_step <= start_step or args.new_group_min_exposures < 1000:',
        '                if args.stop_at_step <= start_step or (args.new_group_min_exposures < 1000 and args.practical_variant != "replay_only"):')
    out = out.replace("applied to v43 rows only", "applied to ALL replay rows; new practical rows excluded")
    out = out.replace("KL(v43)", "KL(parent30500/replay)")
    out = _replace_once(out, "        log_interval = 50", "        log_interval = 10")
    out = _replace_once(out, "        exposure_counts = {key: 0 for key in set(dataset.groups)}\n",
        "        exposure_counts = {key: 0 for key in set(dataset.groups)}\n"
        "        phase_sources = {}\n        phase_source_tokens = {}\n        phase_source_supervised = {}\n"
        "        phase_supervised = 0\n        phase_replay_supervised = 0\n        phase_input_tokens = 0\n"
        "        phase_padded_tokens = 0\n")
    out = _replace_once(out, "            planned = 0\n", "            planned = 0\n            planned_sources = {}\n            planned_supervised = 0\n            planned_replay = 0\n")
    anchor = f'                planned += sum(dataset.groups[index] == "{NEW_GROUP}" for index in indices)'
    out = _replace_once(out, anchor, anchor +
        '\n                for index in indices:\n'
        '                    tokens = dataset.supervised_counts[index]\n'
        '                    planned_supervised += tokens\n'
        f'                    planned_replay += tokens if dataset.groups[index] != "{NEW_GROUP}" else 0\n'
        '                    origin = dataset.origins[index]\n'
        '                    planned_sources[origin] = planned_sources.get(origin, 0) + 1')
    out = _replace_once(out, "            planned_new_exposures = planned\n",
        '            planned_new_exposures = planned\n'
        '            planned_replay_share = planned_replay / max(1, planned_supervised)\n'
        '            practical_telemetry(args.telemetry_jsonl, {"event": "preview", "new_presentations": planned,\n'
        '                "source_presentations": planned_sources, "supervised_tokens": planned_supervised,\n'
        '                "replay_supervised_tokens": planned_replay, "replay_supervised_share": planned_replay_share})\n'
        '            if planned_replay_share < args.min_replay_share:\n'
        '                raise RuntimeError("deterministic phase replay share is below 75%; no update performed")\n')
    out = _replace_once(out,
        '                exposure_counts[group] = exposure_counts.get(group, 0) + 1\n',
        '                exposure_counts[group] = exposure_counts.get(group, 0) + 1\n'
        '                origin = dataset.origins[idx]\n'
        '                tokens = dataset.supervised_counts[idx]\n'
        '                inputs = len(dataset.samples[idx][0]) - 1\n'
        '                phase_sources[origin] = phase_sources.get(origin, 0) + 1\n'
        '                phase_source_tokens[origin] = phase_source_tokens.get(origin, 0) + inputs\n'
        '                phase_source_supervised[origin] = phase_source_supervised.get(origin, 0) + tokens\n'
        '                phase_supervised += tokens\n'
        f'                phase_replay_supervised += tokens if group != "{NEW_GROUP}" else 0\n'
        '                phase_input_tokens += inputs\n'
        '            phase_padded_tokens += int(batch["input_ids"].numel())\n')
    out = _replace_once(out,
        '            if step % log_interval == 0:\n                avg_loss = running_loss / log_interval',
        '            if step % log_interval == 0 or step == args.stop_at_step:\n'
        '                window_steps = min(log_interval, step - start_step) if step % log_interval == 0 else max(1, step % log_interval)\n'
        '                avg_loss = running_loss / window_steps')
    out = _replace_once(out, '                kl_txt = ""\n', '                kl_txt = ""\n                mean_kl = sum(kl_log) / len(kl_log) if kl_log else None\n')
    anchor = '                      f"Throughput: {steps_per_sec:.2f} step/s | ETA: {eta_sec/60:.1f}m{ch_txt}", flush=True)'
    out = _replace_once(out, anchor, anchor +
        '\n                practical_telemetry(args.telemetry_jsonl, {"event": "train_step", "step": step,\n'
        '                    "start_step": start_step, "stop_step": args.stop_at_step, "loss": avg_loss, "kl": mean_kl,\n'
        '                    "learning_rate": lr, "elapsed_seconds": elapsed, "eta_seconds": eta_sec,\n'
        '                    "source_presentations": phase_sources, "source_input_tokens": phase_source_tokens,\n'
        '                    "source_supervised_tokens": phase_source_supervised, "supervised_tokens": phase_supervised,\n'
        '                    "replay_supervised_tokens": phase_replay_supervised, "input_tokens": phase_input_tokens,\n'
        '                    "padded_tokens": phase_padded_tokens, "replay_supervised_share": phase_replay_supervised/max(1,phase_supervised),\n'
        '                    "vram_allocated_bytes": torch.cuda.memory_allocated(dev) if dev.type == "cuda" else 0,\n'
        '                    "vram_peak_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0,\n'
        '                    "oom_retries": policy.oom_retries, "variant": args.practical_variant})')
    out = _replace_once(out,
        '                     "source_exposures": exposure_counts,\n',
        '                     "source_exposures": exposure_counts,\n'
        '                     "practical_phase_sources": phase_sources,\n'
        '                     "practical_supervised_tokens": phase_supervised,\n'
        '                     "practical_replay_supervised_tokens": phase_replay_supervised,\n'
        '                     "practical_variant": args.practical_variant,\n')
    helper = '''\n\ndef practical_supervised_count(chunk, prompt_len, spans):
    first = max(1, int(prompt_len))
    return len(chunk) - first - sum(max(0, min(e, len(chunk)) - max(s, first)) for s, e in spans)


def practical_telemetry(path, event):
    record = dict(event, timestamp_utc=datetime.now(timezone.utc).isoformat())
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
    with open(path, "a", encoding="utf-8") as sink:
        sink.write(line + "\\n")
        sink.flush()
        os.fsync(sink.fileno())
    print("GECPU_TRAIN_EVENT " + line, flush=True)
'''
    out = _replace_once(out, "\n\ndef main():\n", helper + "\n\ndef main():\n")
    # Exact zero model/loss/optimizer changes apart from the replay mask and
    # logged sampler controls are checked by the CPU AST tests.
    compile(out, "runtime_practical_v1.py", "exec")
    return out


def write_runtime(original_path: Path, destination: Path, diff_path: Path) -> dict:
    require_sha(original_path, ORIGINAL_RUNTIME_SHA)
    if destination.exists() or diff_path.exists():
        raise FileExistsError("runtime candidate/diff already exist")
    original = original_path.read_text(encoding="utf-8")
    updated = patched_runtime(original)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(updated, encoding="utf-8")
    diff_path.write_text("".join(difflib.unified_diff(original.splitlines(keepends=True), updated.splitlines(keepends=True),
                               fromfile="runtime_30500.py (unchanged original)", tofile="runtime_practical_v1.py (candidate)")),
                         encoding="utf-8")
    return {"original_sha256": ORIGINAL_RUNTIME_SHA, "candidate_sha256": sha(destination),
            "diff_sha256": sha(diff_path), "model_updates": 0}


def original_classes(base_source: Path, runtime_source: Path):
    """Extract only pure-Python sampler classes; never import the GPU runtime."""
    env = {"random": random, "math": math, "List": List, "PreTokenizedMemoryDataset": object}
    for path, name in ((base_source, "DynamicTokenBucketBatchSampler"),
                       (runtime_source, "ResumableTokenBucketSampler")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name]
        if len(nodes) != 1:
            raise ValueError(f"original sampler class missing: {name}")
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env["ResumableTokenBucketSampler"]


def split_sequence(tokens, prompt_len, spans, max_len=2048):
    # Same span/window arithmetic as pinned ToolMaskedDataset.split_sequence.
    sequence = list(tokens) + [3]
    target = max(1, min(int(prompt_len), len(tokens)))
    first = True
    while target < len(sequence):
        start = max(0, target - (max_len - 1 if first else 512))
        end = min(len(sequence), start + max_len)
        if end <= target:
            raise ValueError("cannot cover supervised target within context")
        local = tuple((max(a, start) - start, min(b, end) - start)
                      for a, b in spans if max(a, start) < min(b, end))
        yield end - start, target - start, local
        target = end
        first = False


class LengthDataset:
    def __init__(self):
        self.samples, self.groups, self.origins, self.supervised_counts = [], [], [], []

    def __len__(self):
        return len(self.samples)


def preview_plan(data: Path, *, meta: Path, normalizer_source: Path,
                 base_sampler_source: Path, runtime_source: Path, steps: int,
                 repeat: int, min_new: int = 1000, min_replay_share: float = 0.75) -> dict:
    if not 1 <= repeat <= 16 or steps <= 0 or min_new not in (0, 1000):
        raise ValueError("invalid pilot repeat/steps/exposure floor")
    normalize = normalization_table(normalizer_source)
    vocab = set(json.loads(meta.read_text(encoding="utf-8"))["vocab"])
    dataset = LengthDataset()
    for row in rows(data):
        raw = row.get("full_text") or row.get("prompt", "") + row.get("target", "")
        full = raw.translate(normalize)
        prompt = row.get("prompt", "").translate(normalize)
        ntokens = sum(c in vocab for c in full)
        pl = sum(c in vocab for c in prompt)
        spans = row.get("loss_mask_spans") or []
        if spans and (full != raw or ntokens != len(full)):
            raise ValueError("old or new mask would be dropped by the original runtime")
        group = str(row.get("_v46_group", "legacy"))
        for length, prompt_length, local_spans in split_sequence(range(ntokens), pl, spans):
            supervised = length - prompt_length - sum(max(0, min(b, length) - max(a, prompt_length)) for a, b in local_spans)
            dataset.samples.append((range(length),))
            dataset.groups.append(group)
            dataset.origins.append(str(row.get("_practical_source", group)))
            dataset.supervised_counts.append(supervised)
    new_indices = [i for i, group in enumerate(dataset.groups) if group == NEW_GROUP]
    for _ in range(repeat - 1):
        for field in ("samples", "groups", "origins", "supervised_counts"):
            values = getattr(dataset, field)
            values.extend(values[i] for i in new_indices)
    sampler = original_classes(base_sampler_source, runtime_source)(dataset, max_batch_size=64,
                        max_tokens=36864, megabucket_mult=32, seed=460007)
    iterator = iter(sampler)
    new = supervised = replay = input_tokens = padded = 0
    sources, source_tokens = Counter(), Counter()
    for _ in range(steps):
        try:
            indices = next(iterator)
        except StopIteration:
            iterator = iter(sampler)
            indices = next(iterator)
        padded += len(indices) * max(32, math.ceil(max(len(dataset.samples[i][0]) for i in indices) / 32) * 32)
        for index in indices:
            tokens = dataset.supervised_counts[index]
            supervised += tokens
            new += int(dataset.groups[index] == NEW_GROUP)
            replay += tokens if dataset.groups[index] != NEW_GROUP else 0
            input_tokens += len(dataset.samples[index][0]) - 1
            origin = dataset.origins[index]
            sources[origin] += 1
            source_tokens[origin] += tokens
    share = replay / max(1, supervised)
    report = {"steps": steps, "new_repeat": repeat, "new_presentations": new,
              "required_new_presentations": min_new, "replay_supervised_share": share,
              "supervised_tokens": supervised, "replay_supervised_tokens": replay,
              "input_tokens": input_tokens, "padded_token_capacity": padded,
              "step_token_budget": 36864, "source_presentations": dict(sources),
              "source_supervised_tokens": dict(source_tokens),
              "sampler_seed": 460007, "sampler_reset": True, "model_updates": 0,
              "passed": new >= min_new and share >= min_replay_share}
    if not report["passed"]:
        raise ValueError("pilot preview rejected before training: " + json.dumps(report, ensure_ascii=False))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-source", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--diff-output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_runtime(args.runtime_source, args.runtime_output, args.diff_output), indent=2))


if __name__ == "__main__":
    main()
