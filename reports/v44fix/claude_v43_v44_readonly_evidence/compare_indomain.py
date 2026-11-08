#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Did v44 training destroy v43's in-domain abilities? (read-only on originals)

Both checkpoints run in v43 code (IntegratedModelV42, which has the ALU), v43 prompt format.
  SWE step-control: first N items of eval_swe_step_control_heldout.jsonl, scored like v43's eval
                    (CPU final R0 if the ALU was used, else the last emitted token).
  ALU arithmetic:   4 "問題: 数式 ... \n思考: " prompts with known answers.
"""
import json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).parent
V43 = HERE / "v43"
sys.path.insert(0, str(V43))
CKPTS = {
    "v43_step15000": Path("G:/マイドライブ/GeoCPU_Frontier_CoTraining_v43/checkpoints/checkpoint_step_15000.pt"),
    "v44_step25000": Path("G:/マイドライブ/GeoCPU_Frontier_CoTraining_v44_PythonTIR_Hotfix_20261103/run_1/training/"
                          "checkpoint_step_25000_aa6c44ab47a14297b5c6481a5fb1374d_27867b30ef624c8985a22b81ae7789fa.pt"),
}
N_SWE = int(os.environ.get("N_SWE", "12"))
ALU = [("数式 (87 * 14 + 23) を 13 で割った余りを求めよ。", "6"),
       ("数式 (56 * 9 + 31) を 7 で割った余りを求めよ。", "3"),
       ("数式 (123 + 456) * 2 を計算せよ。", "1158"),
       ("数式 (95 * 12 - 40) を 9 で割った余りを求めよ。", "2")]


def final_answer(text):
    m = re.findall(r"答え\s*[:：]\s*([-0-9]+)", text)
    return m[-1] if m else None


def main():
    import torch
    torch.set_num_threads(max(4, (os.cpu_count() or 8) // 2))
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from integrated_model_v42 import IntegratedModelV42
    from local_search_engine_v42 import LocalSearchEngineV42
    meta = json.loads((V43 / "base_metadata.json").read_bytes())
    vocab = meta["vocab"]; inv = {int(v): k for k, v in vocab.items()}
    cfg_dict = dict(meta["config"]); cfg_dict["scan_backend"] = "jit"
    model = IntegratedModelV42(base=GeoCPU_CED_v2(CEDConfig(**cfg_dict)),
                               search_engine=LocalSearchEngineV42(corpus_path=V43 / "data" / "knowledge_corpus_v43.jsonl"))
    init_fixed = {k: v.clone() for k, v in model.state_dict().items() if k.startswith(("alu.", "ram."))}
    swe = [json.loads(l) for l in (V43 / "data" / "eval_swe_step_control_heldout.jsonl").read_text(encoding="utf-8").splitlines()][:N_SWE]

    out = {}
    for name, path in CKPTS.items():
        t0 = time.monotonic()
        ck = torch.load(str(path), map_location="cpu", mmap=True, weights_only=True)
        sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
        res = model.load_state_dict(sd, strict=False)
        # ALU/RAM are fixed deterministic circuits; v44 has none, so the v43 constants must remain.
        fixed_ok = all(torch.equal(model.state_dict()[k], v) for k, v in init_fixed.items()) if name != "v43_step15000" else True
        model.eval()
        info = {"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys),
                "missing_are_alu_ram_only": all(k.startswith(("alu.", "ram.")) for k in res.missing_keys),
                "fixed_circuits_intact": fixed_ok, "load_sec": round(time.monotonic() - t0, 1)}
        print(json.dumps({"ckpt": name, **info}, ensure_ascii=False), flush=True)
        rows = []
        with torch.no_grad():
            for it in swe:
                g = model.generate_with_qwenputer(it["prompt"], vocab, inv, max_tokens=400, arm="learned", repetition_penalty=1.0)
                trace = g.get("cpu_trace", []) or []
                r0 = [s["val"] for s in trace if s.get("dest") == "R0"]
                raw = g["output_text"].strip()
                cleaned = re.sub(r"<[^>]+>(?:\s*=>\s*\[[^\]]+\])?", "", raw).strip().split()
                ans = str(r0[-1]) if (g.get("cpu_invocations", 0) > 0 and r0) else (cleaned[-1] if cleaned else "")
                rows.append({"task": "swe", "ok": ans == it["target"].strip(), "cpu": g.get("cpu_invocations", 0),
                             "eos": g.get("ended_with_eos"), "text": raw[:160]})
                print(f"  {name} SWE ok={rows[-1]['ok']} cpu={rows[-1]['cpu']} | {raw[:90]!r}", flush=True)
            for q, a in ALU:
                g = model.generate_with_qwenputer_v42(f"問題: {q}\n思考: ", vocab=vocab, inv_vocab=inv, max_tokens=300,
                                                      repetition_penalty=1.0, enable_search=False)
                fa = final_answer(g["output_text"])
                rows.append({"task": "alu", "ok": fa == a, "cpu": g["cpu_invocations"], "eos": g["ended_with_eos"],
                             "text": g["output_text"][:200]})
                print(f"  {name} ALU ok={fa == a} ({fa} vs {a}) cpu={g['cpu_invocations']} | {g['output_text'][:90]!r}", flush=True)
        summ = {t: {"correct": sum(r["ok"] for r in rows if r["task"] == t), "n": sum(1 for r in rows if r["task"] == t),
                    "used_alu": sum(1 for r in rows if r["task"] == t and r["cpu"] > 0)} for t in ("swe", "alu")}
        out[name] = {"load": info, "summary": summ, "rows": rows, "sec": round(time.monotonic() - t0, 1)}
        print(json.dumps({"ckpt": name, "summary": summ}, ensure_ascii=False), flush=True)
        (HERE / "compare_indomain_result.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        del ck, sd


if __name__ == "__main__":
    main()
