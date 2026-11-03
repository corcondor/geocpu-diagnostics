#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real 247M weights: generate_v44fix (python on) must equal v43's generate_with_qwenputer_v42 token for token
when the model writes no python. Read-only on the v43 checkpoint."""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))
CKPT = Path("G:/マイドライブ/GeoCPU_Frontier_CoTraining_v43/checkpoints/checkpoint_step_15000.pt")


def main():
    import torch
    torch.set_num_threads(6)
    from GeoCPU_CED_v1 import CEDConfig
    from GeoCPU_CED_v2 import GeoCPU_CED_v2
    from local_search_engine_v42 import LocalSearchEngineV42
    from integrated_model_v44fix import IntegratedModelV44Fix
    meta = json.loads((ROOT / "base_metadata.json").read_bytes())
    vocab = meta["vocab"]; inv = {int(v): k for k, v in vocab.items()}
    cfg = dict(meta["config"]); cfg["scan_backend"] = "jit"
    model = IntegratedModelV44Fix(base=GeoCPU_CED_v2(CEDConfig(**cfg)),
                                  search_engine=LocalSearchEngineV42(corpus_path=ROOT / "data" / "knowledge_corpus_v43.jsonl"))
    ck = torch.load(str(CKPT), map_location="cpu", mmap=True, weights_only=True)
    res = model.load_state_dict(ck["model_state_dict"] if "model_state_dict" in ck else ck, strict=False)
    model.eval()
    print(json.dumps({"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys)}), flush=True)
    prompts = ["問題: 数式 (87 * 14 + 23) を 13 で割った余りを求めよ。\n思考:",
               "質問: 日本国憲法で戦争の放棄を定めているのは第何条か？\n思考:"]
    ok = True
    with torch.no_grad():
        for p in prompts:
            t0 = time.monotonic()
            a = model.generate_with_qwenputer_v42(p, vocab=vocab, inv_vocab=inv, max_tokens=200, enable_search=True)
            b = model.generate_v44fix(p, vocab, inv, max_tokens=200, enable_search=True, enable_python=True)
            same = a["generated_tokens"] == b["generated_tokens"] and a["output_text"] == b["output_text"]
            ok &= same
            print(json.dumps({"prompt": p[:24], "identical": same, "cpu": [a["cpu_invocations"], b["cpu_invocations"]],
                              "search": [a["search_invocations"], b["search_invocations"]], "python": b["python_invocations"],
                              "stop": b["stop_reason"], "sec": round(time.monotonic() - t0, 1),
                              "text": b["output_text"][:160]}, ensure_ascii=False), flush=True)
    print("PARITY_OK" if ok else "PARITY_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
