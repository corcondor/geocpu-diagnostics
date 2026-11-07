#!/usr/bin/env python3
"""GeoCPU v48 additive dataset: do NOT remove or replace existing training sets.

Only original, synthetic training data are written. T1-T8 and external corpora are untouched.
Use --verify for independent Python stdout and schema/vocabulary checks.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

DOMAINS = ("python", "logic", "evidence", "japanese")
OLD_PARENT = 422500
OLD_ADDITIONS = 848
TRAIN_PER_DOMAIN = 1024
DIFFICULTY = (1, 2, 3, 4, 3, 4, 3, 4)
FENCE = chr(96) * 3

def rng(domain, family, index, split):
    key = f"v48|{domain}|{family}|{index}|{split}"
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return random.Random(seed)

def py_case(base, mode, r, depth):
    n = 6 + 3 * depth
    values = [r.randint(1, 65) for _ in range(n)]
    threshold, divisor = r.randint(9, 31), r.randint(2, 7)
    if base == 0:
        filtered = [x for x in values if x >= threshold and x % divisor]
        answers = (sum(filtered), len(filtered), max(filtered, default=0), sum(x*x for x in filtered))
        exprs = ("sum(a)", "len(a)", "max(a, default=0)", "sum(x*x for x in a)")
        code = f"v={values!r}\na=[x for x in v if x>={threshold} and x%{divisor}!=0]\nprint({exprs[mode]})"
        text = f"数列{values!r}で、{threshold}以上かつ{divisor}の倍数ではない値だけを選び、" + ("合計", "個数", "最大値（該当なしなら0）", "二乗和")[mode] + "を求めてください。"
    elif base == 1:
        w = r.randint(2, min(6, n-1))
        windows = [sum(values[i:i+w]) for i in range(n-w+1)]
        answers = (max(windows), min(windows), sum(windows), max(windows)-min(windows))
        exprs = ("max(s)", "min(s)", "sum(s)", "max(s)-min(s)")
        code = f"v={values!r}\nw={w}\ns=[sum(v[i:i+w]) for i in range(len(v)-w+1)]\nprint({exprs[mode]})"
        text = f"数列{values!r}の連続する{w}個の和をすべて求め、" + ("最大値", "最小値", "すべての和", "最大値と最小値の差")[mode] + "を答えてください。"
    elif base == 2:
        rows = [(r.randint(1, 6), r.randint(1, 55)) for _ in range(n)]
        latest = dict(rows)
        vv = list(latest.values())
        answers = (sum(vv), sum(x >= threshold for x in vv), max(vv), sum(x%2==0 for x in vv))
        exprs = ("sum(d.values())", f"sum(x>={threshold} for x in d.values())", "max(d.values())", "sum(x%2==0 for x in d.values())")
        code = f"rows={rows!r}\nd={{}}\nfor k,v in rows:\n    d[k]=v\nprint({exprs[mode]})"
        text = f"履歴{rows!r}は(識別番号,新しい値)です。同一番号では最後の行だけを採用し、" + ("値の合計", f"{threshold}以上の件数", "最大値", "偶数値の件数")[mode] + "を求めてください。"
    elif base == 3:
        mid = n//2
        a, b = set(values[:mid]), set(values[mid:])
        answers = (len(a&b), len(a-b), len(a|b), len(a^b))
        exprs = ("len(a&b)", "len(a-b)", "len(a|b)", "len(a^b)")
        code = f"v={values!r}\na=set(v[:len(v)//2])\nb=set(v[len(v)//2:])\nprint({exprs[mode]})"
        text = f"数列{values!r}を前半Aと後半Bに分け、それぞれを集合としたときの" + ("共通要素数", "Aだけの要素数", "和集合の要素数", "片方だけの要素数")[mode] + "を答えてください。"
    elif base == 4:
        size = 3 + depth%2
        mat = [[r.randint(1, 40) for _ in range(size)] for _ in range(size)]
        answers = (sum(mat[i][i] for i in range(size)), sum(mat[i][size-i-1] for i in range(size)), sum(x>=threshold for row in mat for x in row), sum(max(row) for row in mat))
        exprs = ("sum(a[i][i] for i in range(len(a)))", "sum(a[i][len(a)-i-1] for i in range(len(a)))", f"sum(x>={threshold} for row in a for x in row)", "sum(max(row) for row in a)")
        code = f"a={mat!r}\nprint({exprs[mode]})"
        text = f"行列{mat!r}について" + ("主対角線の和", "逆対角線の和", f"{threshold}以上の値の件数", "各行の最大値の和")[mode] + "を求めてください。"
    elif base == 5:
        differences = [values[i+1]-values[i] for i in range(n-1)]
        answers = (sum(x>0 for x in differences), sum(x<0 for x in differences), sum(abs(x) for x in differences), max([0]+differences))
        exprs = ("sum(x>0 for x in d)", "sum(x<0 for x in d)", "sum(abs(x) for x in d)", "max([0]+d)")
        code = f"v={values!r}\nd=[v[i+1]-v[i] for i in range(len(v)-1)]\nprint({exprs[mode]})"
        text = f"時系列{values!r}について、前回との差から" + ("増加回数", "減少回数", "絶対変化量の合計", "最大増加幅（増加なしなら0）")[mode] + "を求めてください。"
    elif base == 6:
        rows = [(r.randint(0, 42), r.randint(1, 29)) for _ in range(n)]
        ends = [s+t for s,t in rows]
        answers = (max(ends), sum(x>threshold for x in ends), sum(max(0,x-threshold) for x in ends), sum(x<=threshold for x in ends))
        exprs = ("max(e)", f"sum(x>{threshold} for x in e)", f"sum(max(0,x-{threshold}) for x in e)", f"sum(x<={threshold} for x in e)")
        code = f"a={rows!r}\ne=[s+t for s,t in a]\nprint({exprs[mode]})"
        text = f"作業記録{rows!r}は(開始時刻,所要時間)です。終了時刻について" + ("最大値", f"{threshold}を超える件数", f"{threshold}を超過する時間の合計", f"{threshold}までの件数")[mode] + "を答えてください。"
    else:
        remainders = [x%divisor for x in values]
        answers = (sum(remainders), remainders.count(0), max(remainders), sum(x%2==0 for x in remainders))
        exprs = ("sum(a)", "a.count(0)", "max(a)", "sum(x%2==0 for x in a)")
        code = f"v={values!r}\na=[x%{divisor} for x in v]\nprint({exprs[mode]})"
        text = f"数列{values!r}を{divisor}で割った余りの列に変え、" + ("合計", "0の件数", "最大値", "偶数の余りの個数")[mode] + "を答えてください。"
    return text, str(answers[mode]), code

def make_row(domain, family, index, split):
    r = rng(domain, family, index, split)
    level = DIFFICULTY[index % 8]
    base, mode = family//4, family%4
    info = {}
    if domain in ("python", "logic"):
        question, answer, code = py_case(base%8, mode, r, level)
        if level >= 3:
            question += " 操作の順序と例外条件を確認し、途中の値を取り違えないでください。"
        trace = "必要な条件を先に確認し、最終値を検算します"
        if domain == "logic":
            code = None
    elif domain == "evidence":
        names = ("研究", "装置", "倉庫", "部署", "材料", "検査", "設計", "記録", "作業")
        name = names[base%len(names)]
        hops = level+1+(1 if family>=32 else 0)
        nodes = [f"{name}{i}号{r.randint(10,99)}" for i in range(hops+1)]
        value, offset = r.randint(20,90), r.randint(2,12)
        docs = [f"{nodes[i]}の参照先は{nodes[i+1]}。" for i in range(hops)]
        docs.append(f"{nodes[-1]}の登録値は{value}。")
        docs += [f"{name}別{j}号{r.randint(10,99)}の登録値は{r.randint(10,99)}。" for j in range(level+2)]
        r.shuffle(docs)
        answer = str((value, value+offset, value-offset, value%offset)[mode])
        question = "次は架空の資料です。\n" + "\n".join(f"資料{i+1}: {x}" for i,x in enumerate(docs))
        question += f"\n対象{nodes[0]}から参照先を順にたどり、終点の値を" + ("そのまま", f"{offset}だけ増やして", f"{offset}だけ減らして", f"{offset}で割った余りとして")[mode] + "答えてください。"
        trace = "参照順は" + "から".join(nodes) + f"です。終点の値は{value}です"
        code = None
        info = {"hops": hops, "source_is_synthetic": True}
    else:
        threshold = r.randint(14,32)
        tests = [
            (lambda x:x%2==0,"偶数である"),
            (lambda x:x%3==0,"3の倍数である"),
            (lambda x:x%5==0,"5の倍数である"),
            (lambda x:x>=threshold,f"{threshold}以上である"),
            (lambda x:x%2==0 and x>=threshold,f"偶数かつ{threshold}以上である"),
            (lambda x:x%7!=0,"7の倍数ではない"),
            (lambda x:x%4==1,"4で割った余りが1である"),
            (lambda x:threshold<=x<=threshold+20,f"{threshold}以上{threshold+20}以下である"),
            (lambda x:x//10>x%10,"十の位が一の位より大きい"),
        ]
        predicate, desc = tests[base%len(tests)]
        positive = [x for x in range(10,100) if predicate(x)]
        negative = [x for x in range(10,100) if not predicate(x)]
        is_positive = mode in (0,2)
        chosen = r.choice(positive if is_positive else negative)
        others = r.sample(negative if is_positive else positive, 3)
        options = [chosen] + others
        r.shuffle(options)
        answer = "ABCD"[options.index(chosen)]
        notice = (
            "条件を丁寧に読み取ってください。",
            "否定の表現を見落とさないでください。",
            "条件を同時に満たすか確認してください。",
            "結論を急がず選択肢を比較してください。",
        )[mode]
        question = f"選択肢: A={options[0]}, B={options[1]}, C={options[2]}, D={options[3]}。\n{notice}"
        question += f" 「{desc}」に" + ("該当する" if is_positive else "該当しない") + "ものを一つ選び、最後に記号で答えてください。"
        trace = f"各値を判定すると指定条件に合うのは{chosen}だけです"
        code = None
        info = {"options": options, "correct_option": options.index(chosen)}
    prompt = "問題: " + question + ("\n道具: Python\n思考:" if domain=="python" else "\n条件と根拠を順に考えてください。")
    steps, spans = [], []
    if code is not None:
        observation = f"\n{FENCE}output\n{answer}\n{FENCE}\n"
        target = f"\n{trace}。\n{FENCE}python\n{code}\n{FENCE}" + observation + f"答え: {answer}。実行結果を確認しました。"
        pos = len(prompt) + target.index(observation)
        spans = [[pos, pos+len(observation)]]
        steps = [{"index": 0, "code": code, "stdout": answer+"\n", "observation": observation, "success": True}]
        info.update({"python_code": code, "expected_stdout": answer+"\n"})
    else:
        target = f"\n{trace}。\n答え: {answer}。"
    return {
        "id": f"v48_{domain}_{split}_{family}_{index}",
        "source": "selfauthored:geocpu/v48-additive",
        "license": "CC0-1.0",
        "split": split, "training_category": domain,
        "family": f"{domain}_{family}", "structural_template": f"{domain}_{base}",
        "difficulty": level, "_v46_group": "v48_additive_only",
        "prompt": prompt, "target": target, "full_text": prompt+target,
        "loss_mask_spans": spans, "tool_steps": steps, "reference_final_answer": answer,
        **info,
    }

def generate(domain, split):
    if split == "train":
        return [make_row(domain,f,i,split) for f in range(32) for i in range(32)]
    if split == "dev":
        return [make_row(domain,f,i,split) for f in range(32) for i in range(1000,1004)]
    if split == "unseen_eval":
        return [make_row(domain,f,i,split) for f in range(32,36) for i in range(2000,2032)]
    raise ValueError(split)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent/"data")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    vocab_path = Path(__file__).resolve().parents[2] / "src/v44fix/base_metadata.json"
    vocab = set(json.loads(vocab_path.read_text(encoding="utf-8"))["vocab"])
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Existing data directory is not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {"version":"v48_additive_20261108", "old_parent_unchanged":OLD_PARENT,
                "old_v47_additions_unchanged":OLD_ADDITIONS, "files":{}, "categories":{}}
    for domain in DOMAINS:
        manifest["categories"][domain] = {}
        for split in ("train","dev","unseen_eval"):
            rows = generate(domain,split)
            seen = set()
            for row in rows:
                assert row["full_text"]==row["prompt"]+row["target"]
                assert row["prompt"] not in seen,(row["id"],"duplicate prompt")
                seen.add(row["prompt"])
                assert len(row["full_text"])+1<=2048
                assert set(row["full_text"])<=vocab,(row["id"],set(row["full_text"])-vocab)
                if domain=="python":
                    (a,b), = row["loss_mask_spans"]
                    assert row["full_text"][a:b]==row["tool_steps"][0]["observation"]
                    if args.verify:
                        result = subprocess.run([sys.executable,"-I","-c",row["python_code"]],capture_output=True,text=True,timeout=5)
                        assert result.returncode==0 and result.stdout==row["expected_stdout"],(row["id"],result.stderr,result.stdout)
            if split=="train":
                assert sum(r["difficulty"]>=3 for r in rows)==768
            filename = f"{domain}_{split}.jsonl"
            payload = "".join(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n" for row in rows).encode("utf-8")
            (args.output/filename).write_bytes(payload)
            manifest["files"][filename] = {"rows":len(rows),"sha256":hashlib.sha256(payload).hexdigest(),"bytes":len(payload)}
            manifest["categories"][domain][split] = {"rows":len(rows),"unique_prompts":len(seen)}
            print("VERIFIED",domain,split,len(rows),flush=True)
    (args.output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("ADDITIVE_DATA_COMPLETE",json.dumps({k:sum(v[s]["rows"] for s in ("train",)) for k,v in manifest["categories"].items()},ensure_ascii=False))

if __name__=="__main__":
    main()
