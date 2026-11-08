"""Teacher-forced measurement of the DECISIVE characters of the legacy T5 task (no generation, deterministic, one forward pass per sequence).

Why: T5's result is decided by a handful of characters inside a ~420-character response whose format the model was trained on
(instruction -> three state lines 真/偽/不明 -> ALU line -> 'に合致するのは X である。\\n正解: X').  Free generation (cap, stop rule, answer extractor) hides
which step fails.  We feed the GOLD response up to each decisive character and read the model's next-character logits over A/B/C:

  S1   first state line  : the letter of the entailed option      (decided by the option that cites premise (1) in the original items)
  S2   second state line : the letter of the contradicted option  (cites premise (2))
  S3   third state line  : the letter of the undetermined option  (cites premise (3))
  FIN  'に合致するのは'   : copy the state-line letter that the instruction asks for
  COPY '正解: '           : repeat the previous answer letter

Variants (the content of every option and the truth label of every option are unchanged in all of them):
  orig    the original items
  renum1  premise sentences renumbered 1->2->3->1, option citations follow  (number <-> label correspondence broken)
  renum2  renumbered the other way round                                   (broken again, differently)
  nocite  the '（k）に関して、' prefix removed from the options
and the option letters are cyclically renamed (3 renamings) to remove letter-position bias.
A model that only uses the number shortcut scores ~100 % on orig and ~0 % on renum1/2 (it follows the number to the WRONG option);
a model that compares content keeps its score.  'follows_number_rule' reports the shortcut directly.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional

LETTERS = "ABC"
POSITIONS = ("S1", "S2", "S3", "FIN", "COPY")
TYPE_SLOT = {"positive_entailment": "T", "negation_contradiction": "F", "non_entailment_undeducible": "U", "not_necessarily_true": "U"}
SLOT_NUMBER = {"T": 1, "F": 2, "U": 3}                       # the shortcut: the option citing premise k has label k (1->T, 2->F, 3->U)
VARIANTS = {"orig": {1: 1, 2: 2, 3: 3}, "renum1": {1: 2, 2: 3, 3: 1}, "renum2": {1: 3, 2: 1, 3: 2}, "nocite": {1: 1, 2: 2, 3: 3}}
RENAMINGS = ({"A": "A", "B": "B", "C": "C"}, {"A": "B", "B": "C", "C": "A"}, {"A": "C", "B": "A", "C": "B"})
PROMPT_END = "思考の連鎖と回答:"


def load_templates(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))["instructions"]
    return {ins: v["template"] for ins, v in d.items()}


def parse_prompt(prompt: str) -> dict:
    head, rest = prompt.split("【本文】\n", 1)
    body, rest = rest.split("\n【指示】\n", 1)
    instruction, rest = rest.split("\n【選択肢】\n", 1)
    premises = {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"（(\d)）(.*?)(?=\s*（\d）|$)", body, re.S)}
    options = {m.group(1): (int(m.group(2)), m.group(3)) for m in re.finditer(r"(?m)^([ABC])\. （(\d)）に関して、(.*)$", rest)}
    if sorted(premises) != [1, 2, 3] or sorted(options) != ["A", "B", "C"] or not rest.rstrip().endswith(PROMPT_END):
        raise ValueError("unexpected T5 prompt shape")
    return {"head": head, "premises": premises, "instruction": instruction.strip(), "options": options}


def render_prompt(head, premises, instruction, options, cite=True) -> str:
    body = " ".join(f"（{k}）{premises[k]}" for k in (1, 2, 3))
    opts = "\n".join(f"{L}. " + (f"（{options[L][0]}）に関して、" if cite else "") + options[L][1] for L in LETTERS)
    return f"{head}【本文】\n{body}\n【指示】\n{instruction}\n【選択肢】\n{opts}\n{PROMPT_END}"


def render_target(template: str, letters: dict):
    """-> (target text, {position: char offset of the decisive letter inside the target})."""
    out, offsets, i = [], {}, 0
    pieces = re.split(r"(\{T\}|\{F\}|\{U\}|\{ANS\})", template)
    n_ans = 0
    pos = 0
    for p in pieces:
        if p in ("{T}", "{F}", "{U}", "{ANS}"):
            slot = p[1:-1]
            if slot == "ANS":
                name = "FIN" if n_ans == 0 else "COPY"
                n_ans += 1
            else:
                name = {"T": "S1", "F": "S2", "U": "S3"}[slot]
            offsets[name] = pos
            out.append(letters[slot])
            pos += 1
        else:
            out.append(p)
            pos += len(p)
    return "".join(out), offsets


def build_variant(item: dict, templates: dict, variant: str, renaming: dict) -> dict:
    """One probe: full text (prompt + gold response), decisive offsets, gold letters, and the number-rule letter at each position."""
    p = parse_prompt(item["prompt"])
    pi = VARIANTS[variant]
    op_type = item["operator_type"]
    slot_of_old_letter = {old: slot for slot, old in item["map_type_to_letter"].items()}      # old option letter -> T/F/U
    premises = {pi[k]: text for k, text in p["premises"].items()}
    new_options = {}
    label_of_new_letter = {}
    for old, (k, text) in p["options"].items():
        new = renaming[old]
        new_options[new] = (pi[k], text)
        label_of_new_letter[new] = slot_of_old_letter[old]
    prompt = render_prompt(p["head"], premises, p["instruction"], new_options, cite=(variant != "nocite"))
    letter_of_slot = {slot: new for new, slot in label_of_new_letter.items()}
    ans_slot = TYPE_SLOT[op_type]
    letters = {"T": letter_of_slot["T"], "F": letter_of_slot["F"], "U": letter_of_slot["U"], "ANS": letter_of_slot[ans_slot]}
    target, offs = render_target(templates[p["instruction"]], letters)
    text = prompt + target
    gold = {"S1": letters["T"], "S2": letters["F"], "S3": letters["U"], "FIN": letters["ANS"], "COPY": letters["ANS"]}
    # the shortcut's choice: the option that cites premise number k (k = 1,2,3 for T,F,U; FIN/COPY: the number of the asked-for slot)
    cited = {num: new for new, (num, _t) in new_options.items()} if variant != "nocite" else {}
    rule = {pos: None for pos in POSITIONS}
    if cited:
        rule = {"S1": cited[1], "S2": cited[2], "S3": cited[3], "FIN": cited[SLOT_NUMBER[ans_slot]], "COPY": cited[SLOT_NUMBER[ans_slot]]}
    positions = {pos: len(prompt) + o for pos, o in offs.items()}
    for pos, o in positions.items():
        assert text[o] == gold[pos], (item["id"], variant, pos)
    return {"id": item["id"], "group_id": item["group_id"], "operator_type": op_type, "variant": variant, "renaming": "".join(renaming[x] for x in LETTERS),
            "text": text, "positions": positions, "gold": gold, "number_rule": rule}


def build_all(items: list, templates: dict, variants=tuple(VARIANTS), renamings=RENAMINGS) -> list:
    return [build_variant(it, templates, v, r) for it in items for v in variants for r in renamings]


def score_all(probes: list, scorer: Callable, batch: int = 4, progress: Optional[Callable[[int, int], None]] = None, cache_path=None) -> list:
    """scorer(texts, offsets_per_text) -> per text, per position (S1,S2,S3,FIN,COPY order) a list of 3 logits (A,B,C).
    Resumable through cache_path (JSONL, one row per finished probe)."""
    done = {}
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                done[(row["id"], row["variant"], row["renaming"])] = row
    todo = [p for p in probes if (p["id"], p["variant"], p["renaming"]) not in done]
    sink = cache.open("a", encoding="utf-8") if cache is not None else None
    try:
        for n_batch, s in enumerate(range(0, len(todo), batch), 1):
            chunk = todo[s:s + batch]
            out = scorer([c["text"] for c in chunk], [[c["positions"][pos] for pos in POSITIONS] for c in chunk])
            for c, logits in zip(chunk, out):
                row = {k: c[k] for k in ("id", "group_id", "operator_type", "variant", "renaming", "gold", "number_rule")}
                row["logits"] = {pos: [float(x) for x in lg] for pos, lg in zip(POSITIONS, logits)}
                done[(row["id"], row["variant"], row["renaming"])] = row
                if sink is not None:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            if sink is not None:
                sink.flush()
            if progress is not None and n_batch % 50 == 0:
                progress(len(probes) - len(todo) + s + len(chunk), len(probes))
    finally:
        if sink is not None:
            sink.close()
    return [done[(p["id"], p["variant"], p["renaming"])] for p in probes]


def _pick(logits):
    top = max(logits)
    return LETTERS[logits.index(top)] if all(x == x for x in logits) else None


def summarize(rows: list) -> dict:
    """Accuracy per (variant, position), the share of 'argmax follows the number shortcut', and the same for each operator type.
    Also the distribution of predicted letters (position bias) and a context-level (group) accuracy."""
    acc = defaultdict(lambda: [0, 0])
    rule = defaultdict(lambda: [0, 0])
    by_op = defaultdict(lambda: [0, 0])
    letters = defaultdict(lambda: defaultdict(int))
    group = defaultdict(lambda: defaultdict(list))
    all_ok = defaultdict(lambda: [0, 0])
    for r in rows:
        ok_pos = {}
        for pos in POSITIONS:
            pick = _pick(r["logits"][pos])
            ok = pick == r["gold"][pos]
            ok_pos[pos] = ok
            acc[(r["variant"], pos)][0] += ok
            acc[(r["variant"], pos)][1] += 1
            by_op[(r["variant"], r["operator_type"], pos)][0] += ok
            by_op[(r["variant"], r["operator_type"], pos)][1] += 1
            letters[(r["variant"], pos)][pick] += 1
            if r["number_rule"].get(pos):
                rule[(r["variant"], pos)][0] += pick == r["number_rule"][pos]
                rule[(r["variant"], pos)][1] += 1
            group[(r["variant"], pos)][r["group_id"]].append(ok)
        chain = ok_pos["S1"] and ok_pos["S2"] and ok_pos["FIN"]
        all_ok[r["variant"]][0] += chain
        all_ok[r["variant"]][1] += 1
    def pct(v):
        return None if v[1] == 0 else round(100 * v[0] / v[1], 2)
    out = {"accuracy": {f"{v}/{p}": pct(c) for (v, p), c in sorted(acc.items())},
           "n": {f"{v}/{p}": c[1] for (v, p), c in sorted(acc.items())},
           "follows_number_rule": {f"{v}/{p}": pct(c) for (v, p), c in sorted(rule.items())},
           "chain_S1_S2_FIN_all_correct": {v: pct(c) for v, c in sorted(all_ok.items())},
           "by_operator": {f"{v}/{o}/{p}": pct(c) for (v, o, p), c in sorted(by_op.items())},
           "predicted_letters": {f"{v}/{p}": dict(d) for (v, p), d in sorted(letters.items())},
           "context_level_accuracy": {f"{v}/{p}": round(100 * sum(sum(x) / len(x) for x in g.values()) / len(g), 2) for (v, p), g in sorted(group.items())}}
    return out
