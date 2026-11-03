#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v44fix training data: GPT's public curriculum converted to the v43 prompt format + the REAL v43 data
(ALU and SEARCH rows kept, v43's own row weighting kept) as the anchor.

Root cause this fixes: v44 trained and evaluated in "ユーザー: ...\\nアシスタント: ", a format v43 never saw.
With identical weights and code, that format alone turns v43's correct ALU arithmetic into loops/garbage.
So every row here uses a format that already exists in v43's training data, including v43's separator
(in all 268,032 v43 rows the character right after the prompt is a space):
  tir      -> "問題: {problem}\\n思考:"  target = " " + solution + "\\n答え: {last \\boxed{}}"
  code     -> "指示: {problem}\\n回答:"  target = " " + solution
  japanese -> "指示: {question}[\\n文脈: {context}]\\n回答:"  target = " " + answer  (v43's Databricks_Dolly format)
  anchor   -> v43 rows verbatim, sampled from the RAW row list (no dedup): v43 learned SEARCH from 66 unique rows
              repeated 29,000 times, so deduplicating would erase it.

GPT's records carry no "question" field; the question is recovered from their prompt wrapper.
Public buckets are deduplicated and their repeats capped; every bucket reports unique rows, exposures and repeat factor.

  python build_v44fix_data.py --public-dir /content/public_curriculum --v43-train train_v43_knowledge_unified.jsonl \\
      --eval-dir data --vocab base_metadata.json --out-dir /content/v44fix_data
"""
import argparse, hashlib, json, random, re
from collections import Counter, defaultdict
from pathlib import Path

try:  # the v43 training loader normalizes full_text before tokenizing; check vocabulary the same way
    from qwenputer_controller_v30 import canonical_normalize_text as normalize
except Exception:  # pragma: no cover
    def normalize(text):
        return text

ALU_RE = re.compile(r"<(SET|ADD|SUB|MUL|DIV|MOD|INC|DEC|CMP|AND|OR|XOR|BXOR|NOT|SHL|SHR|LOAD|STORE|JMP|JZ|JNZ|MOV)\b")
SEARCH_RE = re.compile(r"<SEARCH[ \t]")
PY_FENCE = re.compile(r"(?m)^```python")
MAX_CHARS = 2046  # v43 positions=2048; the loader appends EOS only when len < max_len
GPT_HEAD, GPT_TAIL = "ユーザー: ", "\nアシスタント: "


def sha(*parts):
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


def norm_q(text):
    return re.sub(r"\s+", " ", (text or "")).strip()


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j, depth = i + len("\\boxed{"), 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i + len("\\boxed{"):j - 1].strip() if depth == 0 else None


def gpt_question(rec):
    p = rec.get("prompt")
    if not (isinstance(p, str) and p.startswith(GPT_HEAD) and p.endswith(GPT_TAIL)):
        return None
    return p[len(GPT_HEAD):-len(GPT_TAIL)]


def convert(rec):
    """GPT public record -> ((prompt, target), question) in v43 format, or (None, reason)."""
    bucket, target, q = rec.get("bucket"), rec.get("target"), gpt_question(rec)
    if q is None or not isinstance(target, str):
        return None, "schema"
    body = " " + target.lstrip(" ")
    if bucket == "tir":
        ans = last_boxed(target)
        if not ans:
            return None, "tir_no_boxed_answer"
        return (("問題: " + q.strip() + "\n思考:", body.rstrip() + "\n答え: " + ans), q), None
    if bucket == "code":
        return (("指示: " + q.strip() + "\n回答:", body), q), None
    if bucket == "japanese":
        if q.startswith("本文:\n") and "\n質問:\n" in q:
            ctx, question = q[len("本文:\n"):].split("\n質問:\n", 1)
            return (("指示: " + question.strip() + "\n文脈: " + ctx.strip() + "\n回答:", body), question), None
        return (("指示: " + q.strip() + "\n回答:", body), q), None
    return None, "bucket_not_public"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--public-dir", required=True)
    ap.add_argument("--v43-train", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out-dir", required=True)
    # conservative: v44 lost v43's skills with 40% TIR and no protocol anchor; keep v43 rows at ~60% of the mixture
    ap.add_argument("--mix", default="anchor=150000,tir=60000,code=20000,japanese=15000")
    ap.add_argument("--max-repeat", type=float, default=2.0, help="cap for PUBLIC buckets only")
    ap.add_argument("--heldout-tir", type=int, default=300)
    ap.add_argument("--min-search-share-ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=44)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    vocab = set(k for k in json.loads(Path(args.vocab).read_text(encoding="utf-8"))["vocab"] if len(k) == 1)
    mix = {k: int(v) for k, v in (p.split("=") for p in args.mix.split(","))}

    eval_q = set()
    for p in Path(args.eval_dir).glob("*.jsonl"):
        if "knowledge_corpus" in p.name or p.name.startswith("train_"):
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                for key in ("prompt", "question"):
                    if isinstance(r.get(key), str):
                        eval_q.add(norm_q(r[key]))

    rejects, read = Counter(), Counter()
    pools = defaultdict(dict)
    with open(Path(args.public_dir) / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            b = rec.get("bucket")
            if b not in ("tir", "code", "japanese"):
                continue
            read[b] += 1
            conv, why = convert(rec)
            if why:
                rejects[f"{b}:{why}"] += 1; continue
            (prompt, target), q = conv
            full = prompt + target
            if prompt.startswith("ユーザー:") or "\nアシスタント:" in prompt:
                rejects[f"{b}:still_v44_format"] += 1; continue
            if len(full) > MAX_CHARS:
                rejects[f"{b}:exceeds_v43_context_eos_lost"] += 1; continue
            if any(c not in vocab for c in normalize(full)):
                rejects[f"{b}:out_of_vocab_char"] += 1; continue
            if norm_q(q) in eval_q or norm_q(prompt) in eval_q:
                rejects[f"{b}:evaluation_overlap"] += 1; continue
            pools[b][sha(prompt, target)] = {"prompt": prompt, "target": target, "full_text": full,
                                             "source": f"v44fix_public_{b}", "bucket": b,
                                             "provenance": rec.get("provenance", {})}

    # anchor: v43 rows verbatim, RAW list (keeps v43's intentional repetition, e.g. SEARCH 66 unique x 29,000)
    raw = []
    with open(args.v43_train, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            full = r.get("full_text") or (r.get("prompt", "") + r.get("target", ""))
            raw.append({"prompt": r.get("prompt", ""), "target": r.get("target", ""), "full_text": full,
                        "source": r.get("source", "v43"), "bucket": "anchor"})
    v43_search_share = sum(bool(SEARCH_RE.search(r["target"])) for r in raw) / max(1, len(raw))
    v43_src = Counter(r["source"] for r in raw)

    rows, report = [], {}
    for b, n in mix.items():
        if b == "anchor":
            pool = raw[:]
            rng.shuffle(pool)
            n_exp = n
            chosen = [pool[i % len(pool)] for i in range(n_exp)]
            unique = len({sha(r["full_text"]) for r in pool})
        else:
            pool = list(pools[b].values())
            rng.shuffle(pool)
            if not pool:
                raise SystemExit(f"bucket {b} is empty; refusing to build a mixture without it ({dict(rejects)})")
            n_exp = min(n, int(len(pool) * args.max_repeat))
            chosen = [pool[i % len(pool)] for i in range(n_exp)]
            unique = len(pool)
        rows += chosen
        report[b] = {"pool_rows": len(pool), "unique_rows": unique, "requested": n, "exposures": n_exp,
                     "repeat_factor_vs_pool": round(n_exp / len(pool), 3),
                     "alu_rows": sum(bool(ALU_RE.search(r["target"])) for r in chosen),
                     "search_rows": sum(bool(SEARCH_RE.search(r["target"])) for r in chosen),
                     "python_fence_rows": sum(bool(PY_FENCE.search(r["target"])) for r in chosen)}
    anchor_search_share = report["anchor"]["search_rows"] / max(1, report["anchor"]["exposures"])
    anchor_src = Counter(r["source"] for r in rows if r["bucket"] == "anchor")
    rng.shuffle(rows)
    total = len(rows)
    for b in report:
        report[b]["fraction_of_mixture"] = round(report[b]["exposures"] / total, 4)

    with open(out / "train.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    held = []
    for split in ("dev", "test"):
        p = Path(args.public_dir) / f"{split}.jsonl"
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("bucket") != "tir":
                continue
            conv, why = convert(rec)
            if why or len(conv[0][0] + conv[0][1]) > MAX_CHARS:
                continue
            held.append({"prompt": conv[0][0], "answer": last_boxed(rec["target"]), "reference": conv[0][1], "split": split})
    rng.shuffle(held)
    held = held[:args.heldout_tir]
    with open(out / "tir_heldout.jsonl", "w", encoding="utf-8") as f:
        for r in held:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    no_space = sum(1 for r in rows if not r["full_text"][len(r["prompt"]):len(r["prompt"]) + 1] == " ")
    v44_format_rows = sum(1 for r in rows if r["prompt"].startswith("ユーザー:"))
    manifest = {"schema": "geocpu_v44fix_v2", "total_rows": total, "buckets": report, "public_read": dict(read),
                "rejections": dict(rejects), "heldout_tir_rows": len(held),
                "rows_in_v44_user_assistant_format": v44_format_rows,
                "rows_without_v43_space_after_prompt": no_space,
                "search_share": {"v43": round(v43_search_share, 4), "anchor": round(anchor_search_share, 4)},
                "anchor_source_shares_vs_v43": {s: [round(anchor_src[s] / max(1, report["anchor"]["exposures"]), 4),
                                                    round(c / len(raw), 4)] for s, c in v43_src.most_common(25)},
                "max_repeat_cap_public": args.max_repeat, "seed": args.seed,
                "train_sha256": hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=1), flush=True)
    if v44_format_rows:
        raise SystemExit("v44 user/assistant format leaked into training data")
    if no_space:
        raise SystemExit(f"{no_space} rows lack v43's space after the prompt")
    if v43_search_share > 0 and anchor_search_share < args.min_search_share_ratio * v43_search_share:
        raise SystemExit(f"SEARCH share collapsed: anchor {anchor_search_share:.4f} vs v43 {v43_search_share:.4f}")


if __name__ == "__main__":
    main()
