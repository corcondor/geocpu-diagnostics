#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v44fix gate: did the new training keep what v43 could do, and does Python TIR start to work?

Measured with v43's own evaluators so numbers are comparable with v43's report (SWE 96%, state 87.5%):
  swe    eval_step_control_heldout.run_heldout_evaluation   (200 held-out SWE items)
  state  runtime_cotraining_v41_247M.evaluate_tier2_decomposed_suite (200 items)
  alu    8 arithmetic prompts in v43 format (answers checked by hand)
  loops  10 in-domain v43-format prompts: loop rate / EOS rate (no loop guard, 400 tokens)
  tir    held-out NuminaMath-TIR in v43 format: final answer accuracy + Python call rate

  python gate_eval_v44fix.py --ckpt X.pt --out gate.json [--baseline baseline.json] [--tir data/tir_heldout.jsonl]
"""
import argparse, hashlib, json, re, sys, time
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

ALU = [("数式 (87 * 14 + 23) を 13 で割った余りを求めよ。", "6"), ("数式 (56 * 9 + 31) を 7 で割った余りを求めよ。", "3"),
       ("数式 (123 + 456) * 2 を計算せよ。", "1158"), ("数式 (95 * 12 - 40) を 9 で割った余りを求めよ。", "2"),
       ("数式 (73 * 8 + 19) を 11 で割った余りを求めよ。", "9"), ("数式 (64 + 129) * 3 を計算せよ。", "579"),
       ("数式 (88 * 7 - 50) を 6 で割った余りを求めよ。", "2"), ("数式 (35 * 35 + 5) を 12 で割った余りを求めよ。", "6")]
LOOP_PROMPTS = [
    "質問: 日本国憲法で戦争の放棄を定めているのは第何条か？\n思考:",
    "問題: 数式 (45 * 6 + 17) を 8 で割った余りを求めよ。\n思考:",
    "算数文章題: りんごが 12 個あり、3 人で同じ数ずつ分けると 1 人何個になりますか?\n思考:",
    "前提: 少年が公園で犬と走っています。\n仮説: 少年が外にいます。\n問: 前提に対して仮説はどのような論理関係ですか？\n思考:",
    "指示: 日本の四季について簡潔に説明してください。\n回答:",
    "文脈: 富士山は静岡県と山梨県にまたがる日本最高峰の山である。\n質問: 富士山はどの県にまたがっているか?\n答:",
    "状況: 佐藤が鍵を持っています。 佐藤は鍵を田中に渡しました。\n質問: 現在、鍵は誰が持っていますか？\n思考:",
    "問: 雨の日に使うものは?\n選択肢: [0] 傘 [1] 扇子 [2] 帽子 [3] 手袋 [4] 靴下\n思考:",
    "指示: 次の文を丁寧語に直してください。「明日行く」\n回答:",
    "質問: フランスの首都はどこか？\n思考:",
]


def loop_period(toks):
    t = toks[-200:]
    for p in range(1, 90):
        if len(t) > 3 * p and all(t[i] == t[i - p] for i in range(p, len(t))):
            return p
    return None


def last_answer(text):
    m = re.findall(r"(?:答え|答)\s*[:：]\s*([^\n]+)", text)
    return m[-1].strip() if m else None


def canon(ans):
    if ans is None:
        return None
    s = ans.strip().strip("$").replace(" ", "").replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").rstrip(".。")
    s = re.sub(r"^\\text\{(.*)\}$", r"\1", s)
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    try:
        if m:
            v = Fraction(int(m.group(2)), int(m.group(3)))
            return str(-v if m.group(1) else v)
        if re.fullmatch(r"-?\d+(\.\d+)?(/\d+)?", s):
            return str(Fraction(s))
    except (ValueError, ZeroDivisionError):
        pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--tir", default=None, help="tir_heldout.jsonl from build_v44fix_data.py")
    ap.add_argument("--n-tir", type=int, default=100)
    ap.add_argument("--n-swe", type=int, default=200)
    ap.add_argument("--skip-state", action="store_true")
    ap.add_argument("--skip-search", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    import torch
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from local_search_engine_v42 import LocalSearchEngineV42
    from integrated_model_v44fix import IntegratedModelV44Fix
    import eval_step_control_heldout
    from runtime_cotraining_v41_247M import evaluate_tier2_decomposed_suite

    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    meta = json.loads((ROOT / "base_metadata.json").read_bytes())
    vocab = meta["vocab"]; inv = {int(v): k for k, v in vocab.items()}
    cfg = dict(meta["config"]); cfg["scan_backend"] = "jit"
    model = IntegratedModelV44Fix(base=GeoCPU_CED_v2(CEDConfig(**cfg)),
                                  search_engine=LocalSearchEngineV42(corpus_path=ROOT / "data" / "knowledge_corpus_v43.jsonl"))
    ck = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    bad_missing = [k for k in res.missing_keys if not k.startswith(("alu.", "ram."))]
    if bad_missing or res.unexpected_keys:
        raise SystemExit(f"checkpoint/model mismatch: missing={bad_missing[:5]} unexpected={res.unexpected_keys[:5]}")
    model.to(dev).eval()
    t0 = time.monotonic()
    h = hashlib.sha256()
    for p in (Path(__file__), ROOT / "integrated_model_v44fix.py", Path(args.tir) if args.tir else None):
        if p is not None and p.exists():
            h.update(p.read_bytes())
    out = {"ckpt": args.ckpt, "step": ck.get("step") if isinstance(ck, dict) else None, "fingerprint": h.hexdigest()}

    swe = eval_step_control_heldout.run_heldout_evaluation(
        manifest_path=str(ROOT / "data" / "eval_swe_step_control_heldout.jsonl"), arm="learned", device=str(dev),
        model=model, vocab=vocab, inv_vocab=inv, max_samples=args.n_swe)
    n_swe = sum(v.get("total", 0) for v in (swe.get("by_loop_count") or {}).values()) or min(args.n_swe, 200)
    out["swe"] = {"correct": swe.get("correct_val"), "n": n_swe, "acc": swe.get("correct_val", 0) / n_swe,
                  "strict_with_eos": swe.get("strict_with_eos")}
    model.eval()
    if not args.skip_state:
        st = evaluate_tier2_decomposed_suite(model, ROOT / "data" / "eval_v38_decomposed_state_bench200.jsonl", vocab, inv, dev)
        out["state"] = {"correct": st.get("correct"), "n": st.get("total"), "acc": (st.get("accuracy") or 0) / 100.0}
        model.eval()
    if not args.skip_search:
        import eval_v42_search_retrieval
        sp = ROOT / "data" / "eval_v42_search_retrieval_aligned.jsonl"
        if not sp.exists():
            raise SystemExit(f"search manifest missing: {sp} (extract data/ from the v43 zip)")
        t6 = eval_v42_search_retrieval.run_tier6_search_evaluation(manifest_path=sp, device=str(dev), model=model,
                                                                   vocab=vocab, inv_vocab=inv)
        out["search"] = {"acc": (t6.get("accuracy") or 0) / 100.0, "n": t6.get("total"),
                         "call_rate": (t6.get("search_invocation_rate") or 0) / 100.0}
        model.eval()

    rows = []
    with torch.no_grad():
        for q, a in ALU:
            g = model.generate_v44fix(f"問題: {q}\n思考:", vocab, inv, max_tokens=300, enable_search=False)
            rows.append({"ok": canon(last_answer(g["output_text"])) == canon(a), "cpu": g["cpu_invocations"]})
    out["alu"] = {"correct": sum(r["ok"] for r in rows), "n": len(rows), "acc": sum(r["ok"] for r in rows) / len(rows),
                  "used_alu": sum(r["cpu"] > 0 for r in rows)}

    loops = []
    with torch.no_grad():
        for p in LOOP_PROMPTS:
            g = model.generate_v44fix(p, vocab, inv, max_tokens=400)
            loops.append({"prompt": p[:30], "eos": g["ended_with_eos"], "loop": loop_period(g["generated_tokens"]) is not None
                          if len(g["generated_tokens"]) >= 60 else False, "text": g["output_text"][:160]})
    out["loops"] = {"loop_rate": sum(r["loop"] for r in loops) / len(loops), "eos_rate": sum(r["eos"] for r in loops) / len(loops),
                    "samples": loops}

    if args.tir and Path(args.tir).exists():
        items = [json.loads(l) for l in Path(args.tir).read_text(encoding="utf-8").splitlines() if l.strip()][:args.n_tir]
        tr = []
        with torch.no_grad():
            for it in items:
                try:
                    g = model.generate_v44fix(it["prompt"], vocab, inv, max_tokens=1536, enable_search=False)
                except Exception as exc:  # one bad item must not discard SWE/state/search results already computed
                    tr.append({"ok": False, "py": 0, "py_ok": 0, "stop": "exception:" + type(exc).__name__})
                    continue
                pred = last_answer(g["output_text"])
                tr.append({"ok": pred is not None and canon(pred) == canon(it["answer"]), "py": g["python_invocations"],
                           "py_ok": sum(t["success"] for t in g["python_trace"]), "stop": g["stop_reason"]})
        out["tir"] = {"n": len(tr), "acc": sum(r["ok"] for r in tr) / max(1, len(tr)),
                      "python_call_rate": sum(r["py"] > 0 for r in tr) / max(1, len(tr)),
                      "python_success_calls": sum(r["py_ok"] for r in tr),
                      "stop_reasons": {s: sum(r["stop"] == s for r in tr) for s in {r["stop"] for r in tr}}}

    if args.baseline and Path(args.baseline).exists():
        b = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        if b.get("fingerprint") != out["fingerprint"]:
            raise SystemExit("baseline was produced by a different gate script or TIR set; recompute the baseline")
        checks = {"swe_drop<=0.05": out["swe"]["acc"] >= b["swe"]["acc"] - 0.05,
                  "alu_drop<=1/8": out["alu"]["acc"] >= b["alu"]["acc"] - 0.125,
                  "loop_rate_rise<=0.10": out["loops"]["loop_rate"] <= b["loops"]["loop_rate"] + 0.10}
        for k in ("state", "search"):
            if (k in out) != (k in b):
                raise SystemExit(f"'{k}' measured in only one of gate/baseline; run both with the same flags")
            if k in out:
                checks[f"{k}_drop<=0.05"] = out[k]["acc"] >= b[k]["acc"] - 0.05
        if "tir" in out:
            # the new tool must at least start to appear; otherwise the long run would spend hours learning nothing new
            checks["python_call_rate>0"] = out["tir"]["python_call_rate"] > 0
        out["gate"] = {"checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}
    out["sec"] = round(time.monotonic() - t0, 1)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "loops"} | {"loops": {k: v for k, v in out["loops"].items() if k != "samples"}},
                     ensure_ascii=False, indent=1), flush=True)
    if out.get("gate", {}).get("verdict") == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
