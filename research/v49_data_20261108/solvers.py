"""Independent solver: re-derives the answer of a logic/python prompt from the PROMPT TEXT ALONE.
Copied from the v48 data audit (research notes); nothing here imports a generator."""
import ast
import re

FILLER = " 操作の順序と例外条件を確認し、途中の値を取り違えないでください。"


def solve_py(prompt):
    q = prompt[len("問題: "):]
    q = q.split("\n道具: Python")[0].split("\n条件と根拠を順に考えてください。")[0].replace(FILLER, "")
    m = re.fullmatch(r"数列(\[.*?\])で、(\d+)以上かつ(\d+)の倍数ではない値だけを選び、(合計|個数|最大値（該当なしなら0）|二乗和)を求めてください。", q)
    if m:
        v, t, d, what = ast.literal_eval(m[1]), int(m[2]), int(m[3]), m[4]
        a = [x for x in v if x >= t and x % d != 0]
        return str({"合計": sum(a), "個数": len(a), "最大値（該当なしなら0）": max(a, default=0), "二乗和": sum(x * x for x in a)}[what])
    m = re.fullmatch(r"数列(\[.*?\])の連続する(\d+)個の和をすべて求め、(最大値|最小値|すべての和|最大値と最小値の差)を答えてください。", q)
    if m:
        v, w, what = ast.literal_eval(m[1]), int(m[2]), m[3]
        s = [sum(v[i:i + w]) for i in range(len(v) - w + 1)]
        return str({"最大値": max(s), "最小値": min(s), "すべての和": sum(s), "最大値と最小値の差": max(s) - min(s)}[what])
    m = re.fullmatch(r"履歴(\[.*?\])は\(識別番号,新しい値\)です。同一番号では最後の行だけを採用し、(値の合計|\d+以上の件数|最大値|偶数値の件数)を求めてください。", q)
    if m:
        rows, what = ast.literal_eval(m[1]), m[2]
        d = {}
        for k, val in rows:
            d[k] = val
        vals = list(d.values())
        if what == "値の合計":
            return str(sum(vals))
        if what == "最大値":
            return str(max(vals))
        if what == "偶数値の件数":
            return str(sum(x % 2 == 0 for x in vals))
        t = int(re.match(r"\d+", what)[0])
        return str(sum(x >= t for x in vals))
    m = re.fullmatch(r"数列(\[.*?\])を前半Aと後半Bに分け、それぞれを集合としたときの(共通要素数|Aだけの要素数|和集合の要素数|片方だけの要素数)を答えてください。", q)
    if m:
        v, what = ast.literal_eval(m[1]), m[2]
        a, b = set(v[:len(v) // 2]), set(v[len(v) // 2:])
        return str({"共通要素数": len(a & b), "Aだけの要素数": len(a - b), "和集合の要素数": len(a | b), "片方だけの要素数": len(a ^ b)}[what])
    m = re.fullmatch(r"行列(\[\[.*?\]\])について(主対角線の和|逆対角線の和|\d+以上の値の件数|各行の最大値の和)を求めてください。", q)
    if m:
        a, what = ast.literal_eval(m[1]), m[2]
        n = len(a)
        if what == "主対角線の和":
            return str(sum(a[i][i] for i in range(n)))
        if what == "逆対角線の和":
            return str(sum(a[i][n - 1 - i] for i in range(n)))
        if what == "各行の最大値の和":
            return str(sum(max(r) for r in a))
        t = int(re.match(r"\d+", what)[0])
        return str(sum(x >= t for r in a for x in r))
    m = re.fullmatch(r"時系列(\[.*?\])について、前回との差から(増加回数|減少回数|絶対変化量の合計|最大増加幅（増加なしなら0）)を求めてください。", q)
    if m:
        v, what = ast.literal_eval(m[1]), m[2]
        d = [v[i + 1] - v[i] for i in range(len(v) - 1)]
        return str({"増加回数": sum(x > 0 for x in d), "減少回数": sum(x < 0 for x in d), "絶対変化量の合計": sum(abs(x) for x in d), "最大増加幅（増加なしなら0）": max([0] + d)}[what])
    m = re.fullmatch(r"作業記録(\[.*?\])は\(開始時刻,所要時間\)です。終了時刻について(最大値|\d+を超える件数|\d+を超過する時間の合計|\d+までの件数)を答えてください。", q)
    if m:
        rows, what = ast.literal_eval(m[1]), m[2]
        e = [s + t for s, t in rows]
        if what == "最大値":
            return str(max(e))
        t = int(re.match(r"\d+", what)[0])
        if what.endswith("を超える件数"):
            return str(sum(x > t for x in e))
        if what.endswith("を超過する時間の合計"):
            return str(sum(max(0, x - t) for x in e))
        return str(sum(x <= t for x in e))
    m = re.fullmatch(r"数列(\[.*?\])を(\d+)で割った余りの列に変え、(合計|0の件数|最大値|偶数の余りの個数)を答えてください。", q)
    if m:
        v, d, what = ast.literal_eval(m[1]), int(m[2]), m[3]
        a = [x % d for x in v]
        return str({"合計": sum(a), "0の件数": a.count(0), "最大値": max(a), "偶数の余りの個数": sum(x % 2 == 0 for x in a)}[what])
    raise ValueError("unparsed python/logic prompt: " + q[:80])


def pred_from_desc(desc):
    if desc == "偶数である":
        return lambda x: x % 2 == 0
    if desc == "3の倍数である":
        return lambda x: x % 3 == 0
    if desc == "5の倍数である":
        return lambda x: x % 5 == 0
    if desc == "7の倍数ではない":
        return lambda x: x % 7 != 0
    if desc == "4で割った余りが1である":
        return lambda x: x % 4 == 1
    if desc == "十の位が一の位より大きい":
        return lambda x: x // 10 > x % 10
    m = re.fullmatch(r"偶数かつ(\d+)以上である", desc)
    if m:
        t = int(m[1])
        return lambda x: x % 2 == 0 and x >= t
    m = re.fullmatch(r"(\d+)以上(\d+)以下である", desc)
    if m:
        lo, hi = int(m[1]), int(m[2])
        return lambda x: lo <= x <= hi
    m = re.fullmatch(r"(\d+)以上である", desc)
    if m:
        t = int(m[1])
        return lambda x: x >= t
    raise ValueError("unknown predicate: " + desc)


def solve_japanese(prompt):
    m = re.search(r"選択肢: A=(\d+), B=(\d+), C=(\d+), D=(\d+)。", prompt)
    opts = [int(m[i]) for i in range(1, 5)]
    m = re.search(r"「(.+?)」に(該当する|該当しない)ものを一つ選び", prompt)
    pred, positive = pred_from_desc(m[1]), m[2] == "該当する"
    hits = [i for i, x in enumerate(opts) if pred(x) == positive]
    return ("ABCD"[hits[0]] if len(hits) == 1 else "AMBIGUOUS:%d" % len(hits)), len(hits)
