#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v43 baseline health check (read-only on originals).

Same v43 weights (step15000, bit-identical to the v44 parent), v43 code (IntegratedModelV42,
the generate path v43's own evals used). Vary only:
  fmt: v43 ("質問:/問題: ...\n思考: ")  vs  v44 ("ユーザー: ...\nアシスタント: ")
  rp : 1.0 vs 1.12 (v43 fmt only)
Known reference: v44 fmt + v44 code (tir_engine) = 4/4 loops (before_public_answers.json).
=> v43fmt vs v44fmt here isolates FORMAT; v44fmt-here vs v44-before isolates CODE path.

Greedy, 300 tokens, fp32 on CPU. Also reports where v30's static-loop guard
(8tok x4 / 16tok x4) would have cut, and the loop period.
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).parent
V43 = HERE / "v43"
sys.path.insert(0, str(V43))
CKPT = Path("G:/マイドライブ/GeoCPU_Frontier_CoTraining_v43/checkpoints/checkpoint_step_15000.pt")
BEFORE = Path("G:/マイドライブ/GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103/run_1/training/before_public_answers.json")
MAXTOK = int(os.environ.get("MAXTOK", "300"))


def loop_period(toks):
    t = toks[-200:]
    for p in range(1, 90):
        if len(t) > 3 * p and all(t[i] == t[i - p] for i in range(p, len(t))):
            return p
    return None


def guard_cut(toks):
    for n in range(32, len(toks) + 1):
        g = toks[:n]
        if g[-8:] == g[-16:-8] == g[-24:-16] == g[-32:-24]:
            return n
        if n >= 64 and g[-16:] == g[-32:-16] == g[-48:-32] == g[-64:-48]:
            return n
    return None


def main():
    import torch
    torch.set_num_threads(max(4, (os.cpu_count() or 8) // 2))
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from integrated_model_v42 import IntegratedModelV42
    from local_search_engine_v42 import LocalSearchEngineV42

    t0 = time.monotonic()
    meta = json.loads((V43 / "base_metadata.json").read_bytes())
    vocab = meta["vocab"]; inv_vocab = {int(v): k for k, v in vocab.items()}
    cfg_dict = dict(meta["config"]); cfg_dict["scan_backend"] = "jit"
    try:
        cfg = CEDConfig(**cfg_dict)
    except TypeError:
        cfg_dict.pop("scan_backend"); cfg = CEDConfig(**cfg_dict)
    model = IntegratedModelV42(base=GeoCPU_CED_v2(cfg), search_engine=LocalSearchEngineV42(corpus_path=None))
    try:
        ck = torch.load(str(CKPT), map_location="cpu", weights_only=True)
    except Exception:
        ck = torch.load(str(CKPT), map_location="cpu", weights_only=False)   # user's own training artifact
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    res = model.load_state_dict(sd, strict=False)
    model.eval()
    load_info = {"missing_keys": len(res.missing_keys), "unexpected_keys": len(res.unexpected_keys),
                 "missing_sample": res.missing_keys[:8], "unexpected_sample": res.unexpected_keys[:8],
                 "ckpt_step": ck.get("step") if isinstance(ck, dict) else None,
                 "load_sec": round(time.monotonic() - t0, 1)}
    print(json.dumps({"stage": "loaded", **load_info}, ensure_ascii=False), flush=True)

    before = json.load(open(BEFORE, encoding="utf-8"))["records"]
    qs = []
    for r in before:
        p = r["prompt"]
        q = p[len("ユーザー: "):] if p.startswith("ユーザー: ") else p
        q = q[: -len("\nアシスタント: ")] if q.endswith("\nアシスタント: ") else q
        qs.append(("public%d" % r["case_index"], "質問", q))
    qs.append(("alu_mod", "問題", "数式 (87 * 14 + 23) を 13 で割った余りを求めよ。"))   # 1241 mod 13 = 6
    qs.append(("ja_history", "質問", "江戸幕府を開いた人物は誰か？"))                     # 徳川家康

    conds = [("v43fmt", 1.0), ("v44fmt", 1.0), ("v43fmt", 1.12)]
    out = {"load": load_info, "maxtok": MAXTOK, "runs": []}
    for name, head, q in qs:
        for fmt, rp in conds:
            prompt = f"{head}: {q}\n思考: " if fmt == "v43fmt" else f"ユーザー: {q}\nアシスタント: "
            t1 = time.monotonic()
            g = model.generate_with_qwenputer_v42(prompt, vocab=vocab, inv_vocab=inv_vocab, max_tokens=MAXTOK,
                                                  repetition_penalty=rp, enable_search=False)
            toks = g["generated_tokens"]
            rec = {"case": name, "fmt": fmt, "rp": rp, "n_tokens": len(toks), "eos": g["ended_with_eos"],
                   "loop_period": loop_period(toks) if len(toks) >= 60 else None,
                   "v30_guard_would_cut_at": guard_cut(toks), "cpu_invocations": g["cpu_invocations"],
                   "sec": round(time.monotonic() - t1, 1), "text": g["output_text"][:400]}
            out["runs"].append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "text"}, ensure_ascii=False), flush=True)
            print("   TEXT:", repr(rec["text"][:200]), flush=True)
            (HERE / "v43_baseline_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    out["sec"] = round(time.monotonic() - t0, 1)
    (HERE / "v43_baseline_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"stage": "done", "sec": out["sec"]}), flush=True)


if __name__ == "__main__":
    main()
