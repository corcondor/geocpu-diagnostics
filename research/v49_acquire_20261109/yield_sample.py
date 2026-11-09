#!/usr/bin/env python3
"""Indicative yield of the converters from a SMALL spread sample (3 x 100 rows per dataset) read through the Hugging Face
datasets-server `rows` API (read-only JSON, kept in memory, nothing is stored).  It exists so that the Colab run can be planned
before the full download is processed; it is NOT a substitute for the checks on the real files (adoption counts are decided after
those checks).  Wilson 95 % intervals are given because 300 rows is a small sample and the datasets are not shuffled.

    python yield_sample.py --out yield_estimate_sample.json [--no-external-checks]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import convert as C                                              # noqa: E402

API = "https://datasets-server.huggingface.co"
JOBS = {k: (C.DATASETS[k].hf, cfg) for k, cfg in (("jaquad", "default"), ("coding", "default"), ("tengentoppa", "default"),
                                                    ("hotpot", "distractor"), ("toolace", "default"), ("wizard", "default"))}
FRACTIONS = (0.10, 0.50, 0.90)
LENGTH = 100


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


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def num_train_rows(ds: str, cfg: str) -> int:
    d = get("/size", dataset=ds, config=cfg)
    for sp in d["size"]["splits"]:
        if sp["config"] == cfg and sp["split"] == "train":
            return int(sp["num_rows"])
    raise KeyError("no train split in /size for " + ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "yield_estimate_sample.json"))
    ap.add_argument("--no-external-checks", action="store_true")
    ap.add_argument("--dump", default=None, help="write 2 readable kept rows per family here (for reading by eye)")
    a = ap.parse_args()
    ck = C.Checks(use_external=not a.no_external_checks)
    dump = []
    result = {"note": "indicative only: 3 x 100 rows per dataset at 10 / 50 / 90 % of the train split, datasets-server rows API; "
                      "Wilson 95 % intervals; the real run decides adoption", "backends": ck.backends, "max_chars": ck.max_chars, "datasets": {}}
    for key, (ds, cfg) in JOBS.items():
        try:
            total = num_train_rows(ds, cfg)
            recs, offsets = [], []
            for f in FRACTIONS:
                off = min(int(total * f), max(0, total - LENGTH))
                offsets.append(off)
                recs.extend(r["row"] for r in get("/rows", dataset=ds, config=cfg, split="train", offset=off, length=LENGTH)["rows"])
        except Exception as e:                                    # noqa: BLE001
            result["datasets"][key] = {"error": f"{type(e).__name__}: {e}"}
            print(key, "FETCH FAILED", e, file=sys.stderr)
            continue
        spec = C.DATASETS[key]
        st = C.Stats(key)
        rows = list(spec.convert(enumerate(recs), st, ck, spec))
        seen_f = {}
        for r in rows:
            if seen_f.get(r["family"], 0) < 2:
                seen_f[r["family"]] = seen_f.get(r["family"], 0) + 1
                dump.append(r)
        s = C.summarize(st, ck)
        n = s["n_in"]
        fam = {f: {"n": c, "rate": round(c / n, 4), "ci95": wilson(c, n), "est_rows_full_dataset": int(round(c / n * total))} for f, c in s["families"].items()}
        result["datasets"][key] = {
            "train_rows_total": total, "offsets": offsets, "n_sample": n, "n_kept": s["n_out"], "keep_rate": round(s["n_out"] / n, 4) if n else None,
            "keep_ci95": wilson(s["n_out"], n), "est_kept_full_dataset": int(round(s["n_out"] / n * total)) if n else None,
            "families": fam, "top_rejects": dict(list(s["rejects"].items())[:12]), "extra": s["extra"],
            "chars_p50_p90_max_of_kept": ([sorted(len(r["full_text"]) for r in rows)[len(rows) // 2], sorted(len(r["full_text"]) for r in rows)[int(len(rows) * 0.9)],
                                           max(len(r["full_text"]) for r in rows)] if rows else None),
            "oov_rows_over_0_chars": sum(1 for r in rows if r["n_oov"]),
        }
        for extra_key in ("arith_error_rate", "language", "verification", "code_text_languages", "fit"):
            if extra_key in s:
                result["datasets"][key][extra_key] = s[extra_key]
        print(key, n, "->", s["n_out"], file=sys.stderr)
    Path(a.out).write_bytes(json.dumps(result, ensure_ascii=False, indent=1).encode("utf-8"))
    if a.dump:
        Path(a.dump).write_text(C.format_dump(dump), encoding="utf-8")


if __name__ == "__main__":
    main()
