#!/usr/bin/env python3
"""Build the tiny test fixtures from the Hugging Face datasets-server first-rows API (read-only JSON, <= 100 rows per call).

    python make_fixtures.py --fetch          # 6 small GET requests to datasets-server.huggingface.co, saved under --raw
    python make_fixtures.py                  # select rows from the saved first-rows JSON and write fixtures/*

Selection: every record is run through its converter on its own; one record per distinct outcome (emitted family or reject
reason) is kept, smallest first, inside a per-dataset byte budget (<= 20 rows each, <= 100 KB in total).  Rows are real, public
and unmodified (JaQuAD is re-nested into the raw SQuAD-like file layout the reader expects; ToolACE / hotpot / messages rows keep
their original field structure).  Nothing here touches the full datasets.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert as C                                              # noqa: E402

JOBS = {
    "coding": ("llm-jp/Synthetic-JP-EN-Coding-Dataset", "default"),
    "tengentoppa": ("DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k", "default"),
    "jaquad": ("SkelterLabsInc/JaQuAD", "default"),
    "hotpot": ("hotpotqa/hotpot_qa", "distractor"),
    "toolace": ("Team-ACE/ToolACE", "default"),
    "wizard": ("llm-jp/wizardlm8x22b-logical-math-coding-sft-ja", "default"),
}
BUDGET = {"coding": 20000, "tengentoppa": 12000, "jaquad": 11000, "hotpot": 22000, "toolace": 14000, "wizard": 10000}
MAX_ROWS = 20


def fetch(raw_dir: Path):
    raw_dir.mkdir(parents=True, exist_ok=True)
    for key, (ds, cfg) in JOBS.items():
        url = "https://datasets-server.huggingface.co/first-rows?" + urllib.parse.urlencode({"dataset": ds, "config": cfg, "split": "train"})
        req = urllib.request.Request(url, headers={"User-Agent": "geocpu-fixture-probe"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
        (raw_dir / f"{key}.json").write_bytes(data)
        print(key, len(data), "bytes")


def size(o) -> int:
    return len(json.dumps(o, ensure_ascii=False).encode("utf-8"))


def outcome(key, rec, ck):
    spec = C.DATASETS[key]
    st = C.Stats(key)
    rows = list(spec.convert([(0, rec)], st, ck, spec))
    if rows:
        return rows[0]["family"]
    return "reject:" + next(iter(st.rejects))


def select(key, recs, ck):
    info = [(outcome(key, r, ck), size(r), i) for i, r in enumerate(recs)]
    best = {}
    for lab, sz, i in info:
        if lab not in best or sz < best[lab][0]:
            best[lab] = (sz, i)
    order = sorted(best.items(), key=lambda kv: (kv[0].startswith("reject:"), kv[1][0]))
    chosen, used = [], 0
    for lab, (sz, i) in order:
        if len(chosen) < MAX_ROWS and used + sz <= BUDGET[key]:
            chosen.append(i)
            used += sz
    rest = sorted((sz, i) for lab, sz, i in info if i not in chosen)
    for sz, i in rest:
        if len(chosen) < MAX_ROWS and used + sz <= BUDGET[key]:
            chosen.append(i)
            used += sz
    chosen.sort()
    return chosen


def reassemble_jaquad(recs):
    """first-rows view (one record per question) -> raw file layout {data: [{title, paragraphs: [{context, qas: [...]}]}]}."""
    arts = {}
    for r in recs:
        art = arts.setdefault(r["title"], {"title": r["title"], "paragraphs": {}})
        para = art["paragraphs"].setdefault(r["context"], {"context": r["context"], "qas": []})
        a = r["answers"]
        para["qas"].append({"id": r["id"], "question": r["question"], "question_type": r["question_type"],
                            "answers": [{"text": t, "answer_start": s, "answer_type": ty}
                                        for t, s, ty in zip(a["text"], a["answer_start"], a["answer_type"])]})
    return {"data": [{"title": t, "paragraphs": list(a["paragraphs"].values())} for t, a in arts.items()]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--raw", default=str(HERE.parents[2] / "fr_raw"))
    ap.add_argument("--out", default=str(HERE / "fixtures"))
    a = ap.parse_args()
    raw = Path(a.raw)
    if a.fetch:
        fetch(raw)
        return
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    ck = C.Checks(use_external=False, run_code=False)                 # (no code is run to pick fixture rows)
    total = 0
    report = {}
    for key in JOBS:
        recs = [r["row"] for r in json.loads((raw / f"{key}.json").read_text(encoding="utf-8"))["rows"]]
        if key == "jaquad":
            # select whole contexts so the re-nested file keeps every question of a chosen paragraph
            by_ctx = {}
            for r in recs:
                by_ctx.setdefault((r["title"], r["context"]), []).append(r)
            labs = {}
            for ctxkey, rs in by_ctx.items():
                labs[ctxkey] = (sorted({outcome(key, r, ck) for r in rs}), size(rs))
            picked, used, seen = [], 0, set()
            for ctxkey, (ls, sz) in sorted(labs.items(), key=lambda kv: kv[1][1]):
                new = set(ls) - seen
                if (new or len(picked) < 3) and used + sz <= BUDGET[key] and len(picked) < MAX_ROWS:
                    picked.append(ctxkey)
                    used += sz
                    seen |= set(ls)
            sel = [r for ctxkey in picked for r in by_ctx[ctxkey]][:MAX_ROWS]
            payload, name = reassemble_jaquad(sel), "jaquad_train_fixture.json"
        elif key == "tengentoppa":
            idx = select(key, recs, ck)
            sel = [recs[i] for i in idx]
            (out / "tengentoppa_fixture.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sel), encoding="utf-8")
            name, payload = None, None
        else:
            idx = select(key, recs, ck)
            sel = [recs[i] for i in idx]
            payload, name = sel, f"{key}_fixture.json"
        if name:
            (out / name).write_bytes(json.dumps(payload, ensure_ascii=False, indent=None).encode("utf-8"))
        report[key] = {"rows": len(sel), "outcomes": sorted({outcome(key, r, ck) for r in sel})}
    for p in sorted(out.iterdir()):
        if p.is_file():
            total += p.stat().st_size
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("total fixture bytes:", total)
    assert total <= 100_000, total


if __name__ == "__main__":
    main()
