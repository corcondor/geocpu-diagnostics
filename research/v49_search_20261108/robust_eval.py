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


def mcq_rotation_eval(items: list, scorer: Callable[[list], list], rotations: int = 4, batch: int = 8) -> dict:
    """items: [{'id','prompt','target'}]; scorer(list[str]) -> list of 4 logits (A..D) per prompt.
    Returns per-rotation accuracy, debiased accuracy (scores mapped back to the original options and summed), the
    distribution of the letter the model picked, and per-item correctness for paired tests."""
    usable = [(it, parse_mcq(it["prompt"])) for it in items]
    usable = [(it, p) for it, p in usable if p is not None and it["target"].strip() in LETTERS]
    rot_correct = [[] for _ in range(rotations)]
    debiased, picks = [], {l: 0 for l in LETTERS}
    jobs = [(i, k, rotated_prompt(p, k)) for i, (it, p) in enumerate(usable) for k in range(rotations)]
    logits = {}
    for s in range(0, len(jobs), batch):
        chunk = jobs[s:s + batch]
        out = scorer([c[2] for c in chunk])
        for (i, k, _), vec in zip(chunk, out):
            logits[(i, k)] = list(vec)
    for i, (it, p) in enumerate(usable):
        gold = LETTERS.index(it["target"].strip())
        total = [0.0] * 4
        for k in range(rotations):
            v = logits[(i, k)]
            m = max(v)
            lse = m + math.log(sum(math.exp(x - m) for x in v))
            lp = [x - lse for x in v]                        # log-softmax over the four letters
            pick = max(range(4), key=lambda j: lp[j])
            picks[LETTERS[pick]] += 1
            rot_correct[k].append(pick == gold_after_rotation(gold, k))
            for new_pos in range(4):                         # new position -> original option index
                total[(new_pos + k) % 4] += lp[new_pos]
        debiased.append(max(range(4), key=lambda j: total[j]) == gold)
    n = len(usable)
    return {"n": n, "rotation_accuracy": [100 * sum(r) / n if n else float("nan") for r in rot_correct],
            "debiased_accuracy": 100 * sum(debiased) / n if n else float("nan"), "debiased_ci": wilson(sum(debiased), n),
            "picked_letter_share": {l: v / max(1, n * rotations) for l, v in picks.items()},
            "ids": [it["id"] for it, _ in usable], "debiased_correct": debiased, "rotation_correct": rot_correct}


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


def t5_score(item: dict, gen: dict) -> dict:
    text = gen["output_text"].strip()
    pred = extract_choice_abc(text)
    target = item["target"].strip()
    return {"operator_type": item.get("operator_type"), "group_id": item.get("group_id"), "target": target, "pred": pred,
            "correct": pred == target, "has_answer_marker": bool(re.search(r"正解[:：]\s*[ABC]", text)),
            "chars": len(text), "text": text[:400]}


def t6_score(item: dict, gen: dict) -> dict:
    target = item["target"].strip()
    out = gen["output_text"].strip()
    trace = gen.get("search_trace") or []
    retrieved = any(target in str(t.get("result", t)) for t in trace)
    return {"hop_type": item.get("hop_type"), "target": target, "search_calls": int(gen.get("search_invocations", 0)),
            "answer_in_output": target in out, "answer_retrieved": bool(retrieved),
            "used_after_retrieval": bool(retrieved and target in out), "chars": len(out), "text": out[:400]}
