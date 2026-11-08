#!/usr/bin/env python3
"""CPU-only, one-pass deterministic exposure simulation for several step budgets.

Matches the frozen v47 preview_plan pipeline line for line, but tokenizes the
corpus and builds the sampler once, records a snapshot at every candidate
budget, and (optionally) stops at the first budget the caller accepts.
assert_preview_parity() re-runs the untouched v47 preview_plan on one budget
and requires an identical report before anything is trusted.
"""
from collections import Counter
import math
import json
from pathlib import Path


def _in_vocab_counter(vocab):
    """Same number as sum(c in vocab for c in s); the all-in-vocabulary case is checked at C speed."""
    def count(s):
        if vocab.issuperset(s):
            return len(s)
        return sum(c in vocab for c in s)
    return count


def preview_budgets(prep, data: Path, *, meta: Path, normalizer_source: Path,
                    base_sampler_source: Path, runtime_source: Path, budgets,
                    repeat=1, min_new=1000, min_replay_share=0.75, accept=None):
    """Snapshots {steps: report} for every budget reached. With accept(report)->bool the
    simulation stops at the first budget it accepts (later budgets are not simulated)."""
    budgets = sorted(set(int(s) for s in budgets))
    if not budgets or budgets[0] <= 0 or not 1 <= repeat <= 16 or min_new not in (0,1000):
        raise ValueError("Invalid preflight budgets")
    normalize = prep.normalization_table(normalizer_source)
    vocab = set(json.loads(Path(meta).read_text(encoding="utf-8"))["vocab"])
    count = _in_vocab_counter(vocab)
    dataset = prep.LengthDataset()
    for row in prep.rows(data):
        raw = row.get("full_text") or row.get("prompt","") + row.get("target","")
        full = raw.translate(normalize)
        prompt = row.get("prompt","").translate(normalize)
        ntokens = count(full)
        pl = count(prompt)
        spans = row.get("loss_mask_spans") or []
        if spans and (full != raw or ntokens != len(full)):
            raise ValueError("Original runtime would discard a loss mask")
        group = str(row.get("_v46_group","legacy"))
        for length,prompt_length,local_spans in prep.split_sequence(range(ntokens),pl,spans):
            supervised = length - prompt_length - sum(
                max(0,min(b,length)-max(a,prompt_length)) for a,b in local_spans)
            dataset.samples.append((range(length),))
            dataset.groups.append(group)
            dataset.origins.append(str(row.get("_practical_source",group)))
            dataset.supervised_counts.append(supervised)
    new_indices = [i for i,g in enumerate(dataset.groups) if g == prep.NEW_GROUP]
    for _ in range(repeat-1):
        for field in ("samples","groups","origins","supervised_counts"):
            values = getattr(dataset,field)
            values.extend(values[i] for i in new_indices)
    sampler = prep.original_classes(base_sampler_source,runtime_source)(
        dataset,max_batch_size=64,max_tokens=36864,megabucket_mult=32,seed=460007)
    iterator = iter(sampler)
    new = supervised = replay = input_tokens = padded = 0
    sources,source_tokens = Counter(), Counter()
    snapshots = {}
    checkpoints = set(budgets)
    for step in range(1,budgets[-1]+1):
        try:
            indices = next(iterator)
        except StopIteration:
            iterator = iter(sampler)
            indices = next(iterator)
        padded += len(indices)*max(32,math.ceil(max(len(dataset.samples[i][0]) for i in indices)/32)*32)
        for index in indices:
            tokens = dataset.supervised_counts[index]
            supervised += tokens
            new += int(dataset.groups[index] == prep.NEW_GROUP)
            replay += tokens if dataset.groups[index] != prep.NEW_GROUP else 0
            input_tokens += len(dataset.samples[index][0])-1
            origin = dataset.origins[index]
            sources[origin] += 1
            source_tokens[origin] += tokens
        if step in checkpoints:
            share = replay/max(1,supervised)
            snapshots[step] = {
                "steps":step,"new_repeat":repeat,
                "new_presentations":new,"required_new_presentations":min_new,
                "replay_supervised_share":share,"supervised_tokens":supervised,
                "replay_supervised_tokens":replay,"input_tokens":input_tokens,
                "padded_token_capacity":padded,"step_token_budget":36864,
                "source_presentations":dict(sources),
                "source_supervised_tokens":dict(source_tokens),
                "sampler_seed":460007,"sampler_reset":True,
                "model_updates":0,
                "passed":new >= min_new and share >= min_replay_share
            }
            if accept is not None and accept(snapshots[step]):
                break
    return snapshots


def assert_preview_parity(prep, snapshots, data, *, meta, normalizer_source,
                          base_sampler_source,runtime_source,compare_steps):
    """Independent frozen-v47 oracle on one representative step budget."""
    frozen = prep.preview_plan(
        data,meta=meta,normalizer_source=normalizer_source,
        base_sampler_source=base_sampler_source,runtime_source=runtime_source,
        steps=compare_steps,repeat=1,min_new=1000)
    observed = snapshots[compare_steps]
    if frozen != observed:
        bad = [k for k in frozen if frozen.get(k) != observed.get(k)]
        raise RuntimeError("CPU multi-budget preview diverges from pinned v47 code: "+str(bad))
    return True
