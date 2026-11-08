"""Extract the exact response template of the four T5 instructions from the real training rows (parent training file).

For each of the four instructions used by the T5 test, collect training rows with the identical instruction text, split the target into
fixed text and the four letter slots {T} {F} {U} {ANS}, and verify that EVERY collected row is reproduced bit-exactly by the template.

  python extract_t5_templates.py --train <train_40500_exact.jsonl> --t5 <eval_v41_...heldout.jsonl> --out t5_templates.json [--per 60]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

STATE_LINE = re.compile(r"(?m)^- ([ABC]): (.*)$")


def instruction_of(prompt: str):
    m = re.search(r"【指示】\n(.*?)\n【選択肢】", prompt, re.S)
    return m.group(1).strip() if m else None


def options_of(prompt: str):
    return {m.group(1): (m.group(2), m.group(3)) for m in re.finditer(r"(?m)^([ABC])\. （(\d)）に関して、(.*)$", prompt)}


def split_target(target: str):
    """-> (template with {T},{F},{U},{ANS}, letters dict) or None when the target does not have the expected shape."""
    states = list(STATE_LINE.finditer(target))
    if len(states) != 3:
        return None
    t_letter, f_letter, u_letter = (m.group(1) for m in states)
    if len({t_letter, f_letter, u_letter}) != 3:
        return None
    m_fin = re.search(r"に合致するのは ([ABC]) である。\n正解: ([ABC])\s*$", target)
    if not m_fin or m_fin.group(1) != m_fin.group(2):
        return None
    ans = m_fin.group(1)
    out = []
    last = 0
    for slot, m in zip(("T", "F", "U"), states):
        out.append(target[last:m.start(1)])
        out.append("{" + slot + "}")
        last = m.end(1)
    out.append(target[last:m_fin.start(1)])
    out.append("{ANS}")
    last = m_fin.end(1)
    rest = target[last:]
    rest = rest.replace("\n正解: " + ans, "\n正解: {ANS}")
    out.append(rest)
    tpl = "".join(out)
    return tpl, {"T": t_letter, "F": f_letter, "U": u_letter, "ANS": ans}


def render(tpl: str, letters: dict) -> str:
    return tpl.replace("{T}", letters["T"]).replace("{F}", letters["F"]).replace("{U}", letters["U"]).replace("{ANS}", letters["ANS"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--t5", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per", type=int, default=60)
    ap.add_argument("--max-lines", type=int, default=450000)
    args = ap.parse_args(argv)
    t5 = [json.loads(x) for x in args.t5.read_text(encoding="utf-8").splitlines() if x.strip()]
    wanted = {}
    for r in t5:
        wanted.setdefault(instruction_of(r["prompt"]), r["operator_type"])
    assert len(wanted) == 4, wanted.keys()
    rows = {ins: [] for ins in wanted}
    n = 0
    with args.train.open(encoding="utf-8") as f:
        for line in f:
            n += 1
            if n > args.max_lines or all(len(v) >= args.per for v in rows.values()):
                break
            if "JNLI Grounded" not in line:
                continue
            row = json.loads(line)
            ins = instruction_of(row["prompt"])
            if ins in rows and len(rows[ins]) < args.per:
                rows[ins].append(row)
    result = {"scanned_lines": n, "instructions": {}}
    for ins, op in wanted.items():
        rs = rows[ins]
        parsed = [split_target(r["target"]) for r in rs]
        ok = [p for p in parsed if p is not None]
        if not ok:
            raise SystemExit(f"no usable training rows for {op}")
        tpl = ok[0][0]
        bad = 0
        for r, p in zip(rs, parsed):
            if p is None or render(tpl, p[1]) != r["target"]:
                bad += 1
        # the prompt/target seam, and the ordering of T/F/U lines
        seams = {(r["prompt"][-12:], r["target"][:3]) for r in rs}
        result["instructions"][ins] = {"operator_type": op, "rows_checked": len(rs), "rows_not_reproduced": bad, "template": tpl,
                                       "prompt_end_and_target_start": sorted(map(list, seams))[:3]}
        print(op, "rows", len(rs), "not reproduced", bad, file=sys.stderr)
        if bad:
            raise SystemExit(f"template for {op} does not reproduce {bad}/{len(rs)} real rows")
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("written", args.out)


if __name__ == "__main__":
    main()
