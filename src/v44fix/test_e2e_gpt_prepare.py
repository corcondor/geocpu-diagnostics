#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end: GPT's REAL writer (gpt/prepare_public_v44.prepare_public_curriculum) on tiny raw fixtures
-> build_v44fix_data.py. Guards against the schema mismatch that a hand-written fixture hid before."""
import hashlib, json, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
GPT = ROOT / "gpt"
sys.path.insert(0, str(GPT))
import prepare_public_v44 as P  # noqa: E402

def write_jsonl(p, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def pin(p):
    return {"path": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


class EndToEndGptPrepare(unittest.TestCase):
    def test_real_gpt_records_flow_through_builder(self):
        tmp = Path(tempfile.mkdtemp())
        src = tmp / "src"; src.mkdir()
        numina = []
        for i in range(12):
            prob = f"Compute {i} times 3."
            sol = f"We compute with Python.\n```python\nprint({i}*3)\n```\n```output\n{i * 3}\n```\nThe answer is $\\boxed{{{i * 3}}}$."
            numina.append({"problem": prob, "solution": sol,
                           "messages": [{"role": "user", "content": prob}, {"role": "assistant", "content": sol}]})
        write_jsonl(src / "numina_train.jsonl", numina)
        test_prob = "Official test only."
        write_jsonl(src / "numina_test.jsonl", [{"problem": test_prob, "solution": "x",
                                                 "messages": [{"role": "user", "content": test_prob}, {"role": "assistant", "content": "x"}]}])
        write_jsonl(src / "code_train.jsonl", [{"problem": f"Write a C function returning {i}.",
                                                "solution": f"```c\nint f(void){{return {i};}}\n```"} for i in range(6)])
        ja = [{"instruction": f"季節の名前を{i}つ挙げて。", "input": "", "output": "春と夏です。", "category": "brainstorming"} for i in range(1, 5)]
        # this question hashes into GPT's train split (canonical_question/split_question); others may go to dev/test
        ja.append({"instruction": "日本で一番高い山は？", "input": "富士山は日本一高い山です。", "output": "富士山です。", "category": "closed_qa"})
        (src / "japanese_train.json").write_text(json.dumps(ja, ensure_ascii=False), encoding="utf-8")
        files = {("numina", "train"): src / "numina_train.jsonl", ("numina", "test"): src / "numina_test.jsonl",
                 ("code", "train"): src / "code_train.jsonl", ("japanese", "train"): src / "japanese_train.json"}
        lock = json.loads((GPT / "public_source_lock_v44.json").read_text(encoding="utf-8"))
        for (name, split), p in files.items():
            lock["repos"][name]["files"][split] = pin(p)
        lock_path = tmp / "lock.json"
        lock_path.write_text(json.dumps(lock, ensure_ascii=False), encoding="utf-8")
        ev = tmp / "eval"; ev.mkdir()
        write_jsonl(ev / "eval_dummy.jsonl", [{"prompt": "unrelated evaluation question"}])
        v43_path = tmp / "v43_train_sample.jsonl"
        anchors = []
        for i in range(20):
            prompt, target = f"指示: 昔の言語問題{i}\n回答:", " 春です。"
            anchors.append({"prompt": prompt, "target": target, "full_text": prompt + target,
                            "source": "fixture_language_anchor"})
        write_jsonl(v43_path, anchors)

        class FixtureExecutor:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def reset(self):
                return {"reset": True}

            def execute(self, code):
                expression = code.strip().removeprefix("print(").removesuffix(")")
                answer = eval(expression, {"__builtins__": {}}, {})
                return {"success": True, "stdout": f"{answer}\n", "stderr": "", "error": None}

        pub = tmp / "public"
        man = P.prepare_public_curriculum(pub, ROOT / "base_metadata.json", ev, v43_path, source_files=files,
                                          source_lock_path=lock_path, verify_all=True,
                                          executor_factory=FixtureExecutor,
                                          target_counts={"tir": 20, "code": 8, "japanese": 8, "anchor": 8})
        self.assertTrue(man["training_eligible"], man.get("admission_blockers"))
        self.assertTrue(man["tool_verification"]["all_observations_locally_executed"])
        rec = json.loads((pub / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertNotIn("question", rec)                       # the real writer's schema
        self.assertTrue(rec["prompt"].startswith("ユーザー: "))

        out = tmp / "out"
        r = subprocess.run([sys.executable, str(ROOT / "build_v44fix_data.py"), "--public-dir", str(pub), "--v43-train", str(v43_path),
                            "--eval-dir", str(ev), "--vocab", str(ROOT / "base_metadata.json"), "--out-dir", str(out),
                            "--mix", "anchor=500,tir=10,code=8,japanese=8"],
                           capture_output=True, text=True, encoding="utf-8", cwd=str(ROOT))
        self.assertEqual(r.returncode, 0, r.stdout[-3000:] + r.stderr[-3000:])
        m = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        for b in ("tir", "code", "japanese"):
            self.assertGreater(m["buckets"][b]["unique_rows"], 0, m["rejections"])
        rows = [json.loads(l) for l in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        tir = [x for x in rows if x["bucket"] == "tir"]
        self.assertTrue(all(x["prompt"].startswith("問題: Compute ") and x["prompt"].endswith("\n思考:") for x in tir))
        self.assertTrue(all("```python" in x["target"] and "```output" in x["target"] for x in tir))
        self.assertTrue(all(x["target"].rstrip().split("\n")[-1].startswith("答え: ") for x in tir))
        self.assertTrue(all(x["full_text"][len(x["prompt"])] == " " for x in rows))
        ja_prompts = {x["prompt"] for x in rows if x["bucket"] == "japanese"}
        self.assertIn("指示: 日本で一番高い山は？\n文脈: 富士山は日本一高い山です。\n回答:", ja_prompts)
        self.assertTrue(all(not x["prompt"].startswith("ユーザー:") for x in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
