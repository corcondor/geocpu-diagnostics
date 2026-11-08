"""Robust re-measurements of the legacy tests that turned out to be fragile (all CPU-testable, model injected).

* T5  : same 400 items and the same answer extractor, but a generation cap, and every raw output saved.
* T6  : the 21 distinct questions only have their answers in knowledge_corpus_v42 (21/21); the corpus the stage used has 2/21.
        Run against the v42 corpus and record separately whether the answer was RETRIEVED and whether it was USED.
* MCQ : JMMLU with the options rotated (4 cyclic rotations). The model answered 'A' for ~80 % of questions, so a plain run
        measures position bias. Rotation accuracy shows the bias; the debiased score averages over rotations.
* Stats: Wilson intervals and an exact McNemar test for paired per-item correctness (same items, two checkpoints).
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Callable, Optional

OPTION_LINE = re.compile(r"(?m)^([ABCD]): (.*)$")
LETTERS = "ABCD"


# ----------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (100 * (centre - half), 100 * (centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p for discordant pairs: b = only A right, c = only B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired_report(correct_a: list, correct_b: list) -> dict:
    """Same items in the same order. Returns accuracies with Wilson CIs and the exact McNemar p."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired comparison needs the same items")
    n = len(correct_a)
    b = sum(1 for x, y in zip(correct_a, correct_b) if x and not y)       # only A right
    c = sum(1 for x, y in zip(correct_a, correct_b) if y and not x)       # only B right
    ka, kb = sum(map(bool, correct_a)), sum(map(bool, correct_b))
    return {"n": n, "a": ka, "b": kb, "a_ci": wilson(ka, n), "b_ci": wilson(kb, n), "only_a": b, "only_b": c,
            "diff_pp": 100 * (kb - ka) / n if n else float("nan"), "mcnemar_p": mcnemar_exact(b, c)}


# ----------------------------------------------------------------------------- MCQ with rotated options
def parse_mcq(prompt: str):
    """JMMLU prompt -> (head, [4 option texts], tail) or None when it does not have exactly four option lines."""
    ms = list(OPTION_LINE.finditer(prompt))
    if len(ms) != 4 or [m.group(1) for m in ms] != list(LETTERS):
        return None
    head = prompt[:ms[0].start()]
    tail = prompt[ms[-1].end():]
    return head, [m.group(2) for m in ms], tail


def rotated_prompt(parsed, k: int) -> str:
    """Rotation k: new option i is the old option (i+k) % 4. The gold letter moves accordingly."""
    head, opts, tail = parsed
    new = [opts[(i + k) % 4] for i in range(4)]
    return head + "\n".join(f"{LETTERS[i]}: {new[i]}" for i in range(4)) + tail


def gold_after_rotation(gold_index: int, k: int) -> int:
    return (gold_index - k) % 4


def mcq_rotation_eval(items: list, scorer: Callable[[list], list], rotations: int = 4, batch: int = 8,
                      cache_path=None, progress: Optional[Callable[[int, int], None]] = None, progress_every: int = 100) -> dict:
    """items: [{'id','prompt','target'}]; scorer(list[str]) -> list of 4 logits (A..D) per prompt.
    Returns per-rotation accuracy, debiased accuracy (scores mapped back to the original options and summed), the
    distribution of the letter the model picked, per-item correctness for paired tests, and the raw letter logits.

    cache_path: every scored batch is appended to this JSONL, and a restart skips what is already there (a Kaggle session
    that dies after an hour of scoring loses nothing). progress(done, total) is called every progress_every batches.
    A row with a non-finite logit is counted wrong (never silently turned into 'A') and reported in `nonfinite`;
    `ties` counts rows whose top logit is shared by two letters (bf16 readout; ties resolve to the first letter like argmax)."""
    usable = [(it, parse_mcq(it["prompt"])) for it in items]
    usable = [(it, p) for it, p in usable if p is not None and it["target"].strip() in LETTERS]
    rot_correct = [[] for _ in range(rotations)]
    debiased, picks = [], {l: 0 for l in LETTERS}
    jobs = [(i, k, rotated_prompt(p, k)) for i, (it, p) in enumerate(usable) for k in range(rotations)]
    logits = {}
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                logits[(row["id"], row["k"])] = list(row["v"])
    todo = [j for j in jobs if (usable[j[0]][0]["id"], j[1]) not in logits]
    sink = cache.open("a", encoding="utf-8") if cache is not None else None
    try:
        for n_batch, s in enumerate(range(0, len(todo), batch), 1):
            chunk = todo[s:s + batch]
            out = scorer([c[2] for c in chunk])
            if len(out) != len(chunk):
                raise ValueError("scorer returned a different number of rows than prompts")
            for (i, k, _), vec in zip(chunk, out):
                key = (usable[i][0]["id"], k)
                logits[key] = [float(x) for x in vec]
                if sink is not None:
                    sink.write(json.dumps({"id": key[0], "k": k, "v": logits[key]}) + "\n")
            if sink is not None:
                sink.flush()
            if progress is not None and n_batch % progress_every == 0:
                progress(len(jobs) - len(todo) + s + len(chunk), len(jobs))
    finally:
        if sink is not None:
            sink.close()
    nonfinite = ties = 0
    for i, (it, p) in enumerate(usable):
        gold = LETTERS.index(it["target"].strip())
        total = [0.0] * 4
        bad_item = False
        for k in range(rotations):
            v = logits[(it["id"], k)]
            if not all(math.isfinite(x) for x in v):
                nonfinite += 1
                bad_item = True
                rot_correct[k].append(False)
                continue
            top = max(v)
            if sum(1 for x in v if x == top) > 1:
                ties += 1
            lse = top + math.log(sum(math.exp(x - top) for x in v))
            lp = [x - lse for x in v]                        # log-softmax over the four letters
            pick = max(range(4), key=lambda j: lp[j])
            picks[LETTERS[pick]] += 1
            rot_correct[k].append(pick == gold_after_rotation(gold, k))
            for new_pos in range(4):                         # new position -> original option index
                total[(new_pos + k) % 4] += lp[new_pos]
        debiased.append((not bad_item) and max(range(4), key=lambda j: total[j]) == gold)
    n = len(usable)
    picked_total = max(1, sum(picks.values()))
    return {"n": n, "rotation_accuracy": [100 * sum(r) / n if n else float("nan") for r in rot_correct],
            "debiased_accuracy": 100 * sum(debiased) / n if n else float("nan"), "debiased_ci": wilson(sum(debiased), n),
            "picked_letter_share": {l: v / picked_total for l, v in picks.items()},
            "nonfinite": nonfinite, "ties": ties,
            "ids": [it["id"] for it, _ in usable], "debiased_correct": debiased, "rotation_correct": rot_correct,
            "letter_logits": [[logits[(it["id"], k)] for k in range(rotations)] for it, _ in usable]}


# ----------------------------------------------------------------------------- T5 (capped) and T6 (aligned corpus)
def extract_choice_abc(text: str) -> str:
    """The ORIGINAL T5 extractor, verbatim (so the only change versus the original run is the generation cap)."""
    m = re.search(r"正解[:：]\s*([ABC])", text)
    if m:
        return m.group(1)
    m2 = re.findall(r"\b([ABC])\b", text[-40:])
    if m2:
        return m2[-1]
    for ch in reversed(text):
        if ch in ("A", "B", "C"):
            return ch
    return ""


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def run_items(items: list, generate: Callable[[dict], dict], score: Callable[[dict, dict], dict], out_path, limit=None) -> list:
    """Resumable: one JSON line per finished item (id, generation, score). Never regenerates a finished item."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {}
    if out_path.exists():
        for row in read_jsonl(out_path):
            done[row["id"]] = row
    rows = []
    with out_path.open("a", encoding="utf-8") as sink:
        for n, item in enumerate(items):
            if limit is not None and n >= limit:
                break
            if item["id"] in done:
                rows.append(done[item["id"]])
                continue
            gen = generate(item)
            row = {"id": item["id"], **score(item, gen)}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            rows.append(row)
    return rows


def t5_score(item: dict, gen: dict, cap: Optional[int] = None) -> dict:
    """Original extractor and correctness rule. Also records how the generation ended, because the run uses a token cap:
    hit_cap = it neither ended with EOS nor stopped before the cap, so the cap may have cut the answer off."""
    text = gen["output_text"].strip()
    pred = extract_choice_abc(text)
    target = item["target"].strip()
    tokens = gen.get("tokens") or gen.get("generated_tokens") or []
    eos = bool(gen.get("ended_with_eos"))
    return {"operator_type": item.get("operator_type"), "group_id": item.get("group_id"), "target": target, "pred": pred,
            "correct": pred == target, "has_answer_marker": bool(re.search(r"正解[:：]\s*[ABC]", text)),
            "ended_with_eos": eos, "n_tokens": len(tokens), "hit_cap": bool(cap is not None and not eos and len(tokens) >= cap),
            "chars": len(text), "text": text[:400]}


def model_text(out: str, trace: list) -> str:
    """What the MODEL wrote: the output minus every tool result the runtime injected (' => {result} ') and minus the
    <SEARCH ...> commands themselves (a query that spells the answer is not an answer)."""
    for t in trace:
        r = str(t.get("result", "")) if isinstance(t, dict) else str(t)
        if r:
            out = out.replace(f" => {r} ", " ").replace(r, " ")
    return re.sub(r"<SEARCH[ \t]+[^<>]*>", " ", out)


def t6_score(item: dict, gen: dict) -> dict:
    """answer_in_output is the legacy rule (target in output_text) and is CONTAMINATED: the injected search result is part of
    output_text, so it is true whenever the answer was retrieved. The honest measures are answer_written (the model's own text
    contains the answer) and used_after_retrieval (it wrote the answer AFTER the first result that contained it)."""
    target = item["target"].strip()
    out = gen["output_text"].strip()
    trace = gen.get("search_trace") or []
    results = [str(t.get("result", "")) if isinstance(t, dict) else str(t) for t in trace]
    retrieved = any(target in r for r in results)
    written = target in model_text(out, trace)
    used = False
    for r in results:
        if r and target in r and r in out:
            tail = out[out.index(r) + len(r):]
            used = target in model_text(tail, trace)
            break
    tokens = gen.get("generated_tokens") or gen.get("tokens") or []
    return {"hop_type": item.get("hop_type"), "target": target, "search_calls": int(gen.get("search_invocations", 0)),
            "answer_in_output": target in out, "answer_retrieved": bool(retrieved), "answer_written": bool(written),
            "used_after_retrieval": bool(retrieved and used), "chars": len(out), "text": out, "trace": trace,
            "ended_with_eos": bool(gen.get("ended_with_eos")), "n_tokens": len(tokens)}
