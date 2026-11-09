#!/usr/bin/env python3
"""Smoke run of checks.py on REAL small files that already sit on the owner's PC (nothing is downloaded, nothing > 5 MB is read):
  * the real model vocabulary (base_metadata.json, 70 KB),
  * the first 2,000 rows of the JGLUE-based family (train.jsonl is 169 MB: only the first rows are streamed),
  * the small eval manifests of the r6 bundle (benchmark_8100_manifest.jsonl, 4.9 MB, is left out to protect the PC's RAM),
  * a speed/memory measurement of MinHash on synthetic 2,000-character rows.
    python -I -X utf8 real_smoke_checks.py OUT.json
The result is written as ASCII JSON (the Windows console shows Japanese as mojibake)."""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checks as C                                                                  # noqa: E402

SCR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
BASE_META = os.path.join(SCR, "v47_handoff", "bundle", "r6", "base_metadata.json")
CONTROLLER = os.path.join(SCR, "v47_handoff", "bundle", "r6", "qwenputer_controller_v30.py")
EVAL_DIR = os.path.join(SCR, "v47_handoff", "bundle", "r6", "data")
JGLUE = os.path.join(SCR, "wt_v48", "research", "v49_data_20261108", "families", "jglue_human", "data", "train.jsonl")
T8 = os.path.join(SCR, "v47_handoff", "bundle", "code", "tier8_readonly_v44fix.py")


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:                                                                # noqa: BLE001
        return None


def main(out_path):
    res = {}
    t0 = time.time()
    with open(BASE_META, "rb") as f:
        v = C.vocab_from_json_bytes(f.read(), BASE_META)
    res["vocab"] = v.as_dict()
    res["vocab"]["sample_of_oov_relevant"] = {"has_nl": "\n" in v.chars, "has_ideographic_space": "　" in v.chars, "has_tab": "\t" in v.chars}
    with open(CONTROLLER, encoding="utf-8") as f:
        res["runtime_normalizer_matches_controller"] = C.verify_runtime_normalizer(f.read())
    rows = list(C.iter_rows(JGLUE, limit=2000))
    res["jglue_rows_read"] = len(rows)
    o = C.oov_stats(rows, v, group_key=lambda r: r.get("family"))
    res["oov"] = {k: o[k] for k in ("n_rows", "rows_with_any_oov", "rows_over_threshold", "oov_frac", "rows_changed_by_normalizer", "n_distinct_oov_chars")}
    res["oov_top"] = o["top_oov"][:5]
    ln = C.length_stats(rows, group_key=lambda r: r.get("family"))
    res["length_ALL"] = {k: ln["ALL"][k] for k in ("n", "mean", "max", "p50", "p95", "over_2048", "over_1900", "full_text_mismatch")}
    res["exact_dups"] = {k: C.exact_dups(rows)[k] for k in ("n_rows", "within_dups")}
    t = time.time()
    nd = C.near_dups_within(rows, threshold=0.8)
    res["near_dups_2000_rows"] = {"n_dups": nd["n_dups"], "n_clusters_gt1": nd["n_clusters_gt1"], "largest_cluster": nd["largest_cluster"],
                                  "backend": nd["hasher_backend"], "seconds": round(time.time() - t, 2), "n_examples": len(nd["examples"])}
    if nd["examples"]:
        res["near_dup_example_ids"] = [(e["id"], e["similar_to_id"], e["est_jaccard"]) for e in nd["examples"]]
    # eval index from the small manifests (+ T8 prompts)
    r0 = rss_mb()
    t = time.time()
    names = [n for n in sorted(os.listdir(EVAL_DIR)) if n.endswith(".jsonl") and n in C.EVAL_REQUIRED_JSONL and n != "benchmark_8100_manifest.jsonl"]
    srcs = [os.path.join(EVAL_DIR, n) for n in names] + [T8]
    idx = C.build_eval_index(srcs)
    res["eval_index"] = {"files": names + ["tier8_readonly_v44fix.py"], "distinct_13grams": len(idx), "normalised_texts": len(idx.docs),
                         "seconds": round(time.time() - t, 2), "rss_mb_before": r0, "rss_mb_after": rss_mb(),
                         "per_source": {k: v2 for k, v2 in idx.per_source.items()}}
    t = time.time()
    dc = C.decontaminate(rows, idx, boilerplate_eval_df=20)                           # a shingle shared by 20+ eval items is template boilerplate
    res["decontaminate_2000_rows_df50"] = {k: dc[k] for k in ("n_rows", "rows_with_any_shared_shingle", "rows_boilerplate_only", "rows_with_real_overlap",
                                                                "rows_flagged", "shared_count_hist")}
    res["decontaminate_2000_rows_df50"]["boilerplate_n"] = dc["boilerplate"]["n_shingles"]
    res["decontaminate_2000_rows_df50"]["boilerplate_top3"] = dc["boilerplate"]["top"][:3]
    res["decontaminate_2000_rows_df50"]["seconds"] = round(time.time() - t, 2)
    res["decontaminate_examples"] = [{"id": e["id"], "n": e["n_shared_shingles"], "shingles": e["shared_shingles"], "eval": [m["source"] for m in e["eval_matches"]]}
                                     for e in dc["examples"]]
    # speed of MinHash on synthetic 2,000-char rows
    rng = random.Random(3)
    alphabet = "".join(chr(0x3041 + i) for i in range(80)) + "".join(chr(0x4E00 + i * 7) for i in range(300))
    texts = ["".join(rng.choice(alphabet) for _ in range(2000)) for _ in range(150)]
    speed = {}
    for name, flag in (("pure", False), ("numpy", True)):
        try:
            h = C.MinHasher(use_numpy=flag)
        except Exception as e:                                                       # noqa: BLE001
            speed[name] = "unavailable: " + type(e).__name__
            continue
        n = 30 if not flag else 150
        t = time.time()
        for tx in texts[:n]:
            h.signature(tx)
        speed[name] = {"ms_per_row": round((time.time() - t) / n * 1000, 2), "rows_timed": n}
    res["minhash_speed_2000_char_rows"] = speed
    t = time.time()
    r1 = rss_mb()
    big = C.MinHashIndex(snippet_chars=120)
    for tx in texts * 20:                                                            # 3,000 rows
        big.add_text(tx, "x")
    t_add = time.time() - t
    t = time.time()
    big.build()
    res["minhash_index_3000_rows"] = {"add_seconds": round(t_add, 2), "build_seconds": round(time.time() - t, 2), "rss_mb_delta": (rss_mb() - r1) if r1 else None,
                                      "bytes_per_row_estimate": (round((rss_mb() - r1) * 2 ** 20 / 3000) if r1 else None)}
    res["total_seconds"] = round(time.time() - t0, 1)
    with open(out_path, "w", encoding="ascii") as f:
        json.dump(res, f, ensure_ascii=True, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "real_smoke_output.json")
