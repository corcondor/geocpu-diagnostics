"""v49 'show the work' rows for the logic domain.

v48 logic rows were  question -> constant sentence -> '答え: N。'  : the decisive ~2 characters are the result of a
computation that is written nowhere, so a char-level model can only learn the answer distribution.
Here the same questions get a scratchpad: element-by-element judgements, intermediate lists and running sums, each step
locally computable. The question text is exactly v48's (same generator), so the format the evaluation sees is unchanged.

Every row is checked three ways: v48's own answer, an independent prompt-only solver (solvers.py), and the last value of
the trace itself (computed here with separate code).
"""
from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v48_gen", HERE.parent / "v48_balanced_augmentation_20261108/generate_dataset.py")
V48 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V48)
FILLER = V48.py_case.__globals__.get("FILLER", " 操作の順序と例外条件を確認し、途中の値を取り違えないでください。")
LEVEL_FILLER = " 操作の順序と例外条件を確認し、途中の値を取り違えないでください。"


def running_sum(values):
    """'a+b=c, c+d=e'  (a single value is just itself, nothing is '0')."""
    if not values:
        return "0", 0
    acc, parts = values[0], []
    for x in values[1:]:
        parts.append(f"{acc}+{x}={acc + x}")
        acc += x
    return (", ".join(parts) if parts else str(acc)), acc


def joined(values, sep=","):
    return sep.join(str(x) for x in values) if values else "なし"


def trace_for(question: str):
    """question text (without the tool/cue line) -> (list of trace lines, final value as int)."""
    q = question.replace(LEVEL_FILLER, "")
    m = re.fullmatch(r"数列(\[.*?\])で、(\d+)以上かつ(\d+)の倍数ではない値だけを選び、(合計|個数|最大値（該当なしなら0）|二乗和)を求めてください。", q)
    if m:
        v, t, d, what = ast.literal_eval(m[1]), int(m[2]), int(m[3]), m[4]
        flags = [x >= t and x % d != 0 for x in v]
        a = [x for x, f in zip(v, flags) if f]
        lines = ["判定: " + " ".join(f"{x}{'可' if f else '不'}" for x, f in zip(v, flags)), f"対象: {joined(a)}"]
        if what == "合計":
            s, val = running_sum(a); lines.append(f"合計: {s}")
        elif what == "個数":
            val = len(a); lines.append(f"個数: {val}")
        elif what == "二乗和":
            sq = [x * x for x in a]
            lines.append("二乗: " + (", ".join(f"{x}*{x}={x * x}" for x in a) if a else "なし"))
            s, val = running_sum(sq); lines.append(f"二乗和: {s}")
        else:
            val = max(a, default=0); lines.append(f"最大: {val}" if a else "該当なしなので0")
        return lines, val
    m = re.fullmatch(r"数列(\[.*?\])の連続する(\d+)個の和をすべて求め、(最大値|最小値|すべての和|最大値と最小値の差)を答えてください。", q)
    if m:
        v, w, what = ast.literal_eval(m[1]), int(m[2]), m[3]
        s = [sum(v[i:i + w]) for i in range(len(v) - w + 1)]
        lines = [f"窓の和({w}個ずつ): {joined(s)}"]
        if what == "最大値":
            val = max(s); lines.append(f"最大: {val}")
        elif what == "最小値":
            val = min(s); lines.append(f"最小: {val}")
        elif what == "すべての和":
            r, val = running_sum(s); lines.append(f"和: {r}")
        else:
            hi, lo = max(s), min(s); val = hi - lo; lines.append(f"最大: {hi}, 最小: {lo}, 差: {hi}-{lo}={val}")
        return lines, val
    m = re.fullmatch(r"履歴(\[.*?\])は\(識別番号,新しい値\)です。同一番号では最後の行だけを採用し、(値の合計|\d+以上の件数|最大値|偶数値の件数)を求めてください。", q)
    if m:
        rows, what = ast.literal_eval(m[1]), m[2]
        last = {}
        for k, x in rows:
            last[k] = x
        keys = sorted(last)
        vals = [last[k] for k in keys]
        lines = ["最後の値: " + " ".join(f"{k}={last[k]}" for k in keys)]
        if what == "値の合計":
            r, val = running_sum(vals); lines.append(f"合計: {r}")
        elif what == "最大値":
            val = max(vals); lines.append(f"最大: {val}")
        elif what == "偶数値の件数":
            ev = [x for x in vals if x % 2 == 0]; val = len(ev); lines += [f"偶数: {joined(ev)}", f"件数: {val}"]
        else:
            t = int(re.match(r"\d+", what)[0]); ok = [x for x in vals if x >= t]; val = len(ok)
            lines += [f"{t}以上: {joined(ok)}", f"件数: {val}"]
        return lines, val
    m = re.fullmatch(r"数列(\[.*?\])を前半Aと後半Bに分け、それぞれを集合としたときの(共通要素数|Aだけの要素数|和集合の要素数|片方だけの要素数)を答えてください。", q)
    if m:
        v, what = ast.literal_eval(m[1]), m[2]
        A, B = sorted(set(v[:len(v) // 2])), sorted(set(v[len(v) // 2:]))
        sa, sb = set(A), set(B)
        pick = {"共通要素数": sorted(sa & sb), "Aだけの要素数": sorted(sa - sb), "和集合の要素数": sorted(sa | sb), "片方だけの要素数": sorted(sa ^ sb)}[what]
        name = {"共通要素数": "共通", "Aだけの要素数": "Aだけ", "和集合の要素数": "和集合", "片方だけの要素数": "片方だけ"}[what]
        val = len(pick)
        return [f"A: {joined(A)}", f"B: {joined(B)}", f"{name}: {joined(pick)}", f"個数: {val}"], val
    m = re.fullmatch(r"行列(\[\[.*?\]\])について(主対角線の和|逆対角線の和|\d+以上の値の件数|各行の最大値の和)を求めてください。", q)
    if m:
        a, what = ast.literal_eval(m[1]), m[2]
        n = len(a)
        if what == "主対角線の和":
            xs = [a[i][i] for i in range(n)]; r, val = running_sum(xs); return [f"対角: {joined(xs)}", f"和: {r}"], val
        if what == "逆対角線の和":
            xs = [a[i][n - 1 - i] for i in range(n)]; r, val = running_sum(xs); return [f"逆対角: {joined(xs)}", f"和: {r}"], val
        if what == "各行の最大値の和":
            xs = [max(row) for row in a]; r, val = running_sum(xs); return [f"各行の最大: {joined(xs)}", f"和: {r}"], val
        t = int(re.match(r"\d+", what)[0])
        per = [sum(x >= t for x in row) for row in a]
        r, val = running_sum(per)
        return [f"{t}以上の件数(行ごと): {joined(per)}", f"合計: {r}"], val
    m = re.fullmatch(r"時系列(\[.*?\])について、前回との差から(増加回数|減少回数|絶対変化量の合計|最大増加幅（増加なしなら0）)を求めてください。", q)
    if m:
        v, what = ast.literal_eval(m[1]), m[2]
        d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
        lines = [f"差: {joined(d)}"]
        if what == "増加回数":
            val = sum(x > 0 for x in d); lines.append(f"増加: {joined([x for x in d if x > 0])}"); lines.append(f"回数: {val}")
        elif what == "減少回数":
            val = sum(x < 0 for x in d); lines.append(f"減少: {joined([x for x in d if x < 0])}"); lines.append(f"回数: {val}")
        elif what == "絶対変化量の合計":
            ab = [abs(x) for x in d]; lines.append(f"絶対値: {joined(ab)}"); r, val = running_sum(ab); lines.append(f"合計: {r}")
        else:
            val = max([0] + d); lines.append(f"最大増加: {val}")
        return lines, val
    m = re.fullmatch(r"作業記録(\[.*?\])は\(開始時刻,所要時間\)です。終了時刻について(最大値|\d+を超える件数|\d+を超過する時間の合計|\d+までの件数)を答えてください。", q)
    if m:
        rows, what = ast.literal_eval(m[1]), m[2]
        e = [s + t for s, t in rows]
        lines = ["終了: " + (", ".join(f"{s}+{t}={s + t}" for s, t in rows))]
        if what == "最大値":
            val = max(e); lines.append(f"最大: {val}")
        else:
            t = int(re.match(r"\d+", what)[0])
            if what.endswith("を超える件数"):
                ok = [x for x in e if x > t]; val = len(ok); lines += [f"{t}超: {joined(ok)}", f"件数: {val}"]
            elif what.endswith("を超過する時間の合計"):
                ex = [max(0, x - t) for x in e]; lines.append(f"超過: {joined(ex)}"); r, val = running_sum(ex); lines.append(f"合計: {r}")
            else:
                ok = [x for x in e if x <= t]; val = len(ok); lines += [f"{t}以下: {joined(ok)}", f"件数: {val}"]
        return lines, val
    m = re.fullmatch(r"数列(\[.*?\])を(\d+)で割った余りの列に変え、(合計|0の件数|最大値|偶数の余りの個数)を答えてください。", q)
    if m:
        v, d, what = ast.literal_eval(m[1]), int(m[2]), m[3]
        rem = [x % d for x in v]
        lines = [f"余り({d}で割る): {joined(rem)}"]
        if what == "合計":
            r, val = running_sum(rem); lines.append(f"合計: {r}")
        elif what == "0の件数":
            val = rem.count(0); lines.append(f"件数: {val}")
        elif what == "最大値":
            val = max(rem); lines.append(f"最大: {val}")
        else:
            ev = [x for x in rem if x % 2 == 0]; val = len(ev); lines += [f"偶数: {joined(ev)}", f"個数: {val}"]
        return lines, val
    raise ValueError("unparsed logic question: " + q[:70])


def make_row(family: int, index: int, split: str):
    """One v49 logic row. Same question as v48 (domain 'logic', same family/index/split), plus the scratchpad."""
    base = V48.make_row("logic", family, index, split)
    question = base["prompt"][len("問題: "):].split("\n条件と根拠を順に考えてください。")[0]
    lines, final = trace_for(question)
    answer = base["reference_final_answer"]
    if str(final) != answer:
        raise AssertionError(f"trace ends at {final} but the v48 answer is {answer} ({base['id']})")
    target = "\n" + "\n".join(lines) + f"\n答え: {answer}。"
    row = dict(base)
    row.update({"id": base["id"].replace("v48_", "v49trace_", 1), "target": target, "full_text": base["prompt"] + target,
                "_v46_group": "v49_trace_logic", "trace_lines": len(lines)})
    return row


# ---------------------------------------------------------------- japanese (choose by a condition)
def explain(desc: str, x: int):
    """(visible working, truth) for one option under the stated condition. Every step is a small local computation."""
    tens, ones = x // 10, x % 10
    if desc == "偶数である":
        return f"一の位{ones}", ones % 2 == 0
    if desc == "3の倍数である":
        s = tens + ones
        return f"{tens}+{ones}={s}", s % 3 == 0
    if desc == "5の倍数である":
        return f"一の位{ones}", ones in (0, 5)
    if desc == "7の倍数ではない":
        q = x // 7
        return f"7*{q}={7 * q}, 余り{x - 7 * q}", x % 7 != 0
    if desc == "4で割った余りが1である":
        q = x // 4
        return f"4*{q}={4 * q}, 余り{x - 4 * q}", x % 4 == 1
    if desc == "十の位が一の位より大きい":
        return f"十の位{tens} 一の位{ones}", tens > ones
    m = re.fullmatch(r"偶数かつ(\d+)以上である", desc)
    if m:
        t = int(m[1])
        return f"一の位{ones}, {x}と{t}を比べる", x % 2 == 0 and x >= t
    m = re.fullmatch(r"(\d+)以上(\d+)以下である", desc)
    if m:
        lo, hi = int(m[1]), int(m[2])
        return f"{lo}から{hi}の間か", lo <= x <= hi
    m = re.fullmatch(r"(\d+)以上である", desc)
    if m:
        t = int(m[1])
        return f"{x}と{t}を比べる", x >= t
    raise ValueError("unknown condition: " + desc)


def make_japanese_row(family: int, index: int, split: str):
    """Same question as v48 'japanese', but every option is judged on the page before the letter is chosen."""
    base = V48.make_row("japanese", family, index, split)
    q = base["prompt"]
    m = re.search(r"選択肢: A=(\d+), B=(\d+), C=(\d+), D=(\d+)。", q)
    opts = [int(m[i]) for i in range(1, 5)]
    m2 = re.search(r"「(.+?)」に(該当する|該当しない)ものを一つ選び", q)
    desc, positive = m2[1], m2[2] == "該当する"
    lines, hits = [], []
    for letter, x in zip("ABCD", opts):
        work, truth = explain(desc, x)
        lines.append(f"{letter}={x}: {work} → {'条件に合う' if truth else '合わない'}")
        if truth == positive:
            hits.append(letter)
    if len(hits) != 1:
        raise AssertionError(f"{base['id']}: {len(hits)} options satisfy the question")
    lines.append(f"{'該当する' if positive else '該当しない'}もの: {hits[0]}")
    answer = base["reference_final_answer"]
    if hits[0] != answer:
        raise AssertionError(f"trace picks {hits[0]} but the v48 answer is {answer} ({base['id']})")
    target = "\n" + "\n".join(lines) + f"\n答え: {answer}。"
    row = dict(base)
    row.update({"id": base["id"].replace("v48_", "v49trace_", 1), "target": target, "full_text": q + target,
                "_v46_group": "v49_trace_japanese", "trace_lines": len(lines)})
    return row
