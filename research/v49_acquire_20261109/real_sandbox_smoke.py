#!/usr/bin/env python3
"""Does checks.code_sandbox_check reproduce what GeoCPU's own executor stored?  Real rows of the earlier v47 data
(japanese_math_python_dev.jsonl, 830 KB: python fence followed by an output fence written by the executor) are re-run:
rows with exactly one python block and one output block must give the same stdout.  ASCII JSON result.
    python -I -X utf8 real_sandbox_smoke.py OUT.json [N_ROWS]"""
import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C                                                                  # noqa: E402

SCR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PATH = os.path.join(SCR, "v47_handoff", "bundle", "data", "japanese_math_python_dev.jsonl")


def main(out_path, n_rows):
    rows = list(C.iter_rows(PATH, limit=n_rows))
    res = {"rows": len(rows)}
    one = []
    cats = collections.Counter()
    for r in rows:
        blocks = C.extract_code_blocks(r["full_text"])
        py = [b for b in blocks if b.lang in C._PY_LANGS]
        outs = [b for b in blocks if b.lang == "output"]
        cats[f"py={len(py)},out={len(outs)}"] += 1
        if len(py) == 1 and len(outs) == 1:
            one.append((r["id"], py[0].code, outs[0].code))
    res["block_shapes"] = dict(cats.most_common(8))
    res["single_block_rows"] = len(one)
    t = time.time()
    results = C.code_sandbox_check_many([c for _, c, _ in one], timeout=10, workers=2)
    res["seconds"] = round(time.time() - t, 1)
    res["seconds_per_snippet"] = round((time.time() - t) / max(1, len(one)), 3)
    status = collections.Counter()
    examples = collections.defaultdict(list)
    for (rid, code, expect), (ok, out, err) in zip(one, results):
        if ok:
            k = "ok_equal" if C.answers_equal(out.strip(), expect.strip()) or out.strip() == expect.strip() else "ok_but_output_differs"
        else:
            k = "refused_or_failed:" + ":".join(err.split(":")[:2])
        status[k] += 1
        if len(examples[k]) < 2 and not k == "ok_equal":
            examples[k].append({"id": rid, "err": err[:120], "got": out.strip()[:80], "expected": expect.strip()[:80], "code": code[:160]})
    res["status"] = dict(status.most_common())
    res["examples"] = dict(examples)
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(res, f, ensure_ascii=True, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 150)
