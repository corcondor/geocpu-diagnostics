#!/usr/bin/env python3
"""How many code_text rows (and which languages) can the coding dataset give?  1,000 rows read through the datasets-server `rows` API
at 10 spread offsets (in memory only), fenced-code languages counted over ALL rows and over the rows that pass the converter
(<= 1,900 characters, every fence tagged, no python).  Python is not executed here.

    python coding_language_sample.py --out coding_language_sample.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert as C                                              # noqa: E402

API = "https://datasets-server.huggingface.co"
DS, CFG = "llm-jp/Synthetic-JP-EN-Coding-Dataset", "default"


def get(path, **params):
    """GET with 4 tries (the datasets-server answers 502 / 429 now and then)."""
    url = API + path + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "geocpu-yield-probe"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == 3:
                raise
            time.sleep(3 * (attempt + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "coding_language_sample.json"))
    ap.add_argument("--offsets", type=int, default=10)
    a = ap.parse_args()
    total = next(s["num_rows"] for s in get("/size", dataset=DS, config=CFG)["size"]["splits"] if s["split"] == "train")
    ck = C.Checks(run_code=False, use_external=False)
    spec = C.DATASETS["coding"]
    recs = []
    for i in range(a.offsets):
        off = min(int(total * (0.05 + 0.9 * i / max(1, a.offsets - 1))), total - 100)
        recs.extend(r["row"] for r in get("/rows", dataset=DS, config=CFG, split="train", offset=off, length=100)["rows"])
    all_langs, rows_with_lang, first_lang = Counter(), Counter(), Counter()
    for r in recs:
        asst = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
        blocks = C.parse_fences(C.clean_text(asst)) or []
        langs = {b["lang"] for b in blocks if b["lang"]}
        for b in blocks:
            all_langs[b["lang"] or "(untagged)"] += 1
        for lg in langs:
            rows_with_lang[lg] += 1
        code_blocks = [b["lang"] for b in blocks if b["lang"] in C.CODE_LANGS]
        if code_blocks:
            first_lang[code_blocks[0]] += 1
    st = C.Stats("coding")
    rows = list(spec.convert(enumerate(recs), st, ck, spec))
    kept = Counter(r["code_lang"] for r in rows if r["family"] == "code_text")
    n = len(recs)
    out = {"note": "indicative only; 10 x 100 rows at 5..95 % of the train split; python not executed", "n_rows": n, "train_rows_total": total,
           "rows_by_first_code_language_any_length": dict(first_lang.most_common()), "rows_containing_language_any_length": dict(rows_with_lang.most_common(20)),
           "code_text_rows_kept": sum(kept.values()), "code_text_by_language_kept_le_1900": dict(kept.most_common()),
           "code_text_est_full_dataset": int(round(sum(kept.values()) / n * total)),
           "code_text_est_by_language_full_dataset": {k: int(round(v / n * total)) for k, v in kept.most_common()},
           "cpp_or_rust_rows_any_length": first_lang["cpp"] + first_lang["rust"], "cpp_or_rust_kept": kept["cpp"] + kept["rust"],
           "rejects": dict(st.rejects.most_common(10))}
    Path(a.out).write_bytes(json.dumps(out, ensure_ascii=False, indent=1).encode("utf-8"))


if __name__ == "__main__":
    main()
