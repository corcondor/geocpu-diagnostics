"""Compare two checkpoints measured by kaggle_runner.py: paired statistics and a plain-Japanese report.

A side only counts as finished when its summary.json says so (the runner writes it last, with complete=true). Anything missing,
failed or partial is announced in a banner at the top and its statistics are skipped: a half-finished run never gets a verdict.
"""
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


def pct(k, n):
    return f"{100 * k / n:.1f}%" if n else "–"


def banners(a: dict, b: dict, name_a: str, name_b: str) -> list:
    out = []
    for name, side in ((name_a, a), (name_b, b)):
        s = side.get("summary") or {}
        if not s:
            out.append(f"!! 未完了：{name} には summary.json がありません（実行が最後まで終わっていません）。数値は比較できません。")
            continue
        if not s.get("complete"):
            out.append(f"!! 未完了：{name} は途中で止まったか失敗した課題があります："
                       + ", ".join(f"{t}={v.get('error', '?')}" for t, v in (s.get("tasks") or {}).items() if isinstance(v, dict) and "error" in v))
        if s.get("limit"):
            out.append(f"!! 試験実行：{name} は --limit {s['limit']} の少数実行です。本測定ではありません。")
    return out


def report(a: dict, b: dict, name_a: str, name_b: str) -> str:
    L = [f"# 修正版の評価：{name_a}（基準） と {name_b}", ""]
    notes = banners(a, b, name_a, name_b)
    L += notes + ([""] if notes else [])
    L += ["上限つき・出力保存・対応のある検定（同じ問題で比べる）。**元の評価（変更なし）の結果とは別物です。**", ""]
    cap = (a.get("summary") or {}).get("caps", {})

    if "mcq" in a and "mcq" in b and set(a["mcq"]["ids"]) != set(b["mcq"]["ids"]):
        L += ["!! JMMLU：問題の集合が一致しません。比較しません。", ""]
    elif "mcq" in a and "mcq" in b:
        ma, mb = a["mcq"], b["mcq"]
        common = [i for i in ma["ids"] if i in set(mb["ids"])]
        ia = {i: k for k, i in enumerate(ma["ids"])}
        ib = {i: k for k, i in enumerate(mb["ids"])}
        rep = R.paired_report([ma["debiased_correct"][ia[i]] for i in common], [mb["debiased_correct"][ib[i]] for i in common])
        L += ["## JMMLU：選択肢の並びを 4 通りに回して測る", "",
              "| | " + name_a + " | " + name_b + " |", "|---|---:|---:|"]
        for k in range(4):
            L.append(f"| 回転 {k} の正解率 | {ma['rotation_accuracy'][k]:.2f}% | {mb['rotation_accuracy'][k]:.2f}% |")
        L.append(f"| **偏りを除いた正解率（{rep['n']}問）** | **{pct(rep['a'], rep['n'])}**（{ci(rep['a_ci'])}） | **{pct(rep['b'], rep['n'])}**（{ci(rep['b_ci'])}） |")
        for letter in "ABCD":
            L.append(f"| 「{letter}」を選んだ割合 | {100 * ma['picked_letter_share'][letter]:.1f}% | {100 * mb['picked_letter_share'][letter]:.1f}% |")
        L.append(f"| 同点（A〜D の最大値が並んだ回数） | {ma.get('ties', '–')} | {mb.get('ties', '–')} |")
        L.append(f"| 数値が壊れた回数（NaN/inf） | {ma.get('nonfinite', '–')} | {mb.get('nonfinite', '–')} |")
        L += ["", f"対応のある差：{rep['diff_pp']:+.2f} ポイント、McNemar の p={rep['mcnemar_p']:.3f}（{name_a} だけ正解 {rep['only_a']}、{name_b} だけ正解 {rep['only_b']}）→ {verdict(rep)}",
              "", "読み方：4 つの回転で正解率が大きく違えば、位置の偏りです。偏りを除いた正解率が 25% 前後なら、内容を見て選べていません。"
              "回転 0 は元の評価と同じ並びなので、元の JMMLU の数字との照合用（同じ T4 で測った対照）です。", ""]
    elif ("mcq" in a) != ("mcq" in b):
        L += ["!! JMMLU は片方にしかないため比較しません。", ""]

    if a.get("t5") and b.get("t5") and {r["id"] for r in a["t5"]} != {r["id"] for r in b["t5"]}:
        L += [f"!! T5：問題の集合が一致しません（{name_a} {len(a['t5'])} 件、{name_b} {len(b['t5'])} 件）。途中までの結果は比較しません。", ""]
    elif a.get("t5") and b.get("t5"):
        ra, rb = {r["id"]: r for r in a["t5"]}, {r["id"]: r for r in b["t5"]}
        common = [i for i in ra if i in rb]
        rep = R.paired_report([ra[i]["correct"] for i in common], [rb[i]["correct"] for i in common])

        def split(rows, key):
            yes = [r for r in rows if r.get(key)]
            no = [r for r in rows if not r.get(key)]
            return (f"{pct(sum(r['correct'] for r in yes), len(yes))}（{len(yes)}件）", f"{pct(sum(r['correct'] for r in no), len(no))}（{len(no)}件）")
        L += [f"## T5 論理演算子（生成上限 {cap.get('t5', '?')} トークン、元と同じ答えの取り出し方）", "",
              f"- {name_a}：{rep['a']}/{rep['n']}（{pct(rep['a'], rep['n'])}、{ci(rep['a_ci'])}）、「正解:」の書式あり {100 * sum(r['has_answer_marker'] for r in a['t5']) / len(a['t5']):.0f}%、上限に当たった {100 * sum(r.get('hit_cap', False) for r in a['t5']) / len(a['t5']):.0f}%",
              f"- {name_b}：{rep['b']}/{rep['n']}（{pct(rep['b'], rep['n'])}、{ci(rep['b_ci'])}）、「正解:」の書式あり {100 * sum(r['has_answer_marker'] for r in b['t5']) / len(b['t5']):.0f}%、上限に当たった {100 * sum(r.get('hit_cap', False) for r in b['t5']) / len(b['t5']):.0f}%",
              f"- 対応のある差：{rep['diff_pp']:+.1f} ポイント、p={rep['mcnemar_p']:.3f} → {verdict(rep)}",
              "- 偶然水準：答えの文字を当てるだけで 34.8%、演算子ごとの最頻値で 39.8%。"]
        for name, rows in ((name_a, a["t5"]), (name_b, b["t5"])):
            w_yes, w_no = split(rows, "has_answer_marker")
            h_yes, h_no = split(rows, "hit_cap")
            L.append(f"- {name} の内訳：書式あり {w_yes} / 書式なし {w_no}；上限に当たった {h_yes} / 当たらない {h_no}")
        L += ["", "上限に当たった問題の正解率が低いなら、上限が答えを切っている可能性があります（元の評価は上限なし）。", ""]
    elif ("t5" in a) != ("t5" in b) or (a.get("t5") == [] or b.get("t5") == []):
        L += ["!! T5 は片方が空か欠けているため比較しません。", ""]

    if a.get("t6") and b.get("t6") and {r["id"] for r in a["t6"]} != {r["id"] for r in b["t6"]}:
        L += [f"!! T6：問いの集合が一致しません（{name_a} {len(a['t6'])} 件、{name_b} {len(b['t6'])} 件）。比較しません。", ""]
    elif a.get("t6") and b.get("t6"):
        L += ["## T6 検索つき（v42 の資料に合わせた版、固有の問い 21 種）", "", "| | " + name_a + " | " + name_b + " |", "|---|---:|---:|"]

        def w(rows, key):
            return sum(r["weight"] for r in rows if r[key]), sum(r["weight"] for r in rows)
        for key, label in (("answer_retrieved", "答えが検索結果に出た"),
                           ("answer_in_output", "答えが出力にある（元の採点。検索結果の注入も含むので、検索すれば必ず真）"),
                           ("answer_written", "答えをモデル自身が書いた（注入された検索結果と <SEARCH> は除く）"),
                           ("used_after_retrieval", "検索結果が出た後に、答えをモデルが書いた")):
            ka, na = w(a["t6"], key)
            kb, nb = w(b["t6"], key)
            L.append(f"| {label}（行数換算） | {ka}/{na} | {kb}/{nb} |")
        sa, na = sum(r["weight"] for r in a["t6"] if r["search_calls"] > 0), sum(r["weight"] for r in a["t6"])
        sb, nb = sum(r["weight"] for r in b["t6"] if r["search_calls"] > 0), sum(r["weight"] for r in b["t6"])
        L.append(f"| 検索を自分から使った | {sa}/{na} | {sb}/{nb} |")
        L += ["", "注意：固有の問いは 21 種だけで、独立した標本は 21 です。差の検定はしません。", ""]
    elif ("t6" in a) != ("t6" in b) or (a.get("t6") == [] or b.get("t6") == []):
        L += ["!! T6 は片方が空か欠けているため比較しません。", ""]
    return "\n".join(L)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default="v47_30500")
    ap.add_argument("--new", default="v48_61500")
    args = ap.parse_args(argv)
    text = report(load(args.out, args.base), load(args.out, args.new), args.base, args.new)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
