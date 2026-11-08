"""Compare two checkpoints measured by kaggle_runner.py: paired statistics and a plain-Japanese report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import robust_eval as R  # noqa: E402


def load(out: Path, prefix: str):
    d = Path(out) / prefix
    res = {"summary": json.loads((d / "summary.json").read_text(encoding="utf-8")) if (d / "summary.json").exists() else {}}
    if (d / "mcq.json").exists():
        res["mcq"] = json.loads((d / "mcq.json").read_text(encoding="utf-8"))
    for name in ("t5", "t6"):
        if (d / f"{name}.jsonl").exists():
            res[name] = R.read_jsonl(d / f"{name}.jsonl")
    return res


def verdict(rep: dict, higher_is_better=True) -> str:
    p, diff = rep["mcnemar_p"], rep["diff_pp"]
    if p >= 0.05:
        return "差は統計的に有意ではない"
    better = diff > 0 if higher_is_better else diff < 0
    return "有意に向上" if better else "**有意に低下（要調査）**"


def ci(t):
    return f"{t[0]:.1f}–{t[1]:.1f}%"


def report(a: dict, b: dict, name_a: str, name_b: str) -> str:
    L = [f"# 修正版の評価：{name_a}（基準） と {name_b}", "",
         "上限つき・出力保存・対応のある検定（同じ問題で比べる）。**元の評価（変更なし）の結果とは別物です。**", ""]
    if "mcq" in a and "mcq" in b:
        ma, mb = a["mcq"], b["mcq"]
        common = [i for i in ma["ids"] if i in set(mb["ids"])]
        ia = {i: k for k, i in enumerate(ma["ids"])}
        ib = {i: k for k, i in enumerate(mb["ids"])}
        rep = R.paired_report([ma["debiased_correct"][ia[i]] for i in common], [mb["debiased_correct"][ib[i]] for i in common])
        L += ["## JMMLU：選択肢の並びを 4 通りに回して測る", "",
              "| | " + name_a + " | " + name_b + " |", "|---|---:|---:|"]
        for k in range(4):
            L.append(f"| 回転 {k} の正解率 | {ma['rotation_accuracy'][k]:.2f}% | {mb['rotation_accuracy'][k]:.2f}% |")
        L.append(f"| **偏りを除いた正解率（{rep['n']}問）** | **{rep['a']/rep['n']*100:.2f}%**（{ci(rep['a_ci'])}） | **{rep['b']/rep['n']*100:.2f}%**（{ci(rep['b_ci'])}） |")
        for letter in "ABCD":
            L.append(f"| 「{letter}」を選んだ割合 | {100*ma['picked_letter_share'][letter]:.1f}% | {100*mb['picked_letter_share'][letter]:.1f}% |")
        L += ["", f"対応のある差：{rep['diff_pp']:+.2f} ポイント、McNemar の p={rep['mcnemar_p']:.3f}（{name_a} だけ正解 {rep['only_a']}、{name_b} だけ正解 {rep['only_b']}）→ {verdict(rep)}",
              "", "読み方：4 つの回転で正解率が大きく違えば、位置の偏りです。偏りを除いた正解率が 25% 前後なら、内容を見て選べていません。", ""]
    if "t5" in a and "t5" in b:
        ra, rb = {r["id"]: r for r in a["t5"]}, {r["id"]: r for r in b["t5"]}
        common = [i for i in ra if i in rb]
        rep = R.paired_report([ra[i]["correct"] for i in common], [rb[i]["correct"] for i in common])
        L += ["## T5 論理演算子（上限つき、元と同じ答えの取り出し方）", "",
              f"- {name_a}：{rep['a']}/{rep['n']}（{rep['a']/rep['n']*100:.1f}%、{ci(rep['a_ci'])}）、「正解:」の書式あり {100*sum(r['has_answer_marker'] for r in a['t5'])/len(a['t5']):.0f}%",
              f"- {name_b}：{rep['b']}/{rep['n']}（{rep['b']/rep['n']*100:.1f}%、{ci(rep['b_ci'])}）、「正解:」の書式あり {100*sum(r['has_answer_marker'] for r in b['t5'])/len(b['t5']):.0f}%",
              f"- 対応のある差：{rep['diff_pp']:+.1f} ポイント、p={rep['mcnemar_p']:.3f} → {verdict(rep)}",
              "- 偶然水準：答えの文字を当てるだけで 34.8%、演算子ごとの最頻値で 39.8%。", ""]
    if "t6" in a and "t6" in b:
        L += ["## T6 検索つき（v42 の資料に合わせた版、固有の問い 21 種）", "", "| | " + name_a + " | " + name_b + " |", "|---|---:|---:|"]
        def w(rows, key):
            return sum(r["weight"] for r in rows if r[key]), sum(r["weight"] for r in rows)
        for key, label in (("answer_retrieved", "答えが検索結果に出た"), ("answer_in_output", "答えが出力にある"), ("used_after_retrieval", "検索結果を使って答えた")):
            ka, na = w(a["t6"], key)
            kb, nb = w(b["t6"], key)
            L.append(f"| {label}（200 行換算） | {ka}/{na} | {kb}/{nb} |")
        sa = sum(r["weight"] for r in a["t6"] if r["search_calls"] > 0)
        sb = sum(r["weight"] for r in b["t6"] if r["search_calls"] > 0)
        L.append(f"| 検索を自分から使った | {sa}/200 | {sb}/200 |")
        L += ["", "注意：固有の問いは 21 種だけで、独立した標本は 21 です。差の検定はしません。", ""]
    return "\n".join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="v47_30500")
    ap.add_argument("--new", default="v48_61500")
    args = ap.parse_args(argv)
    text = report(load(args.out, args.base), load(args.out, args.new), args.base, args.new)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
