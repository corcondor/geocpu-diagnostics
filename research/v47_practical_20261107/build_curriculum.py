#!/usr/bin/env python3
"""Build verified, isolated practical Python teaching trajectories for GeoCPU.

Only code authored by this module is executed. No model, checkpoint, training
process, or original test is changed. The exact v47 CPU executor produces every
tool observation. This is a curriculum generator, not a model evaluation.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from fractions import Fraction
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import random
import re
import statistics
import sys
import tempfile
import textwrap
import unicodedata
import zipfile


VERSION = "practical_python_multistep_v1"
SEED = 20261107
MAX_CONTEXT = 2048
R6_SHA256 = "d3bbc47da537c9073c0fb113c3073d19a6ad2990201688943141aaba41391263"
GROUP = VERSION


@dataclass
class Spec:
    family: str
    prompt: str
    codes: list[str]
    expected_stdout: list[str]
    actions: list[str]
    answer: str
    unit: str
    conclusion: str
    context_code: str | None = None
    context_exception: str | None = None


def code(source: str) -> str:
    return textwrap.dedent(source).strip() + "\n"


def money(value: Decimal | Fraction | int) -> str:
    if isinstance(value, Fraction):
        value = Decimal(value.numerator) / Decimal(value.denominator)
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_EVEN))


def rng_for(family: str, scenario: int) -> random.Random:
    digest = hashlib.sha256(f"{SEED}/{family}/{scenario}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], "big"))


def spec_invoice(r: random.Random) -> Spec:
    rows = [(f"P{i}", r.randint(1, 12), r.randint(100, 900), "sent" if i % 3 else "cancel")
            for i in range(1, 8)]
    active = [x for x in rows if x[3] == "sent"]
    total = sum(Decimal(q) * Decimal(p) for _, q, p, _ in active)
    return Spec("invoice_filter", f"出荷表の行は(商品,数量,単価円,状態)です。\n{rows!r}\n状態sentの行だけを対象に、数量と単価から請求合計を求めてください。cancelの行は含めず、税や送料は加えません。",
        [code(f"rows = {rows!r}\nactive = [row for row in rows if row[3] == 'sent']\nprint(len(rows), len(active))"),
         code("total = sum(row[1] * row[2] for row in active)\nprint(total)")],
        [f"{len(rows)} {len(active)}\n", f"{total}\n"],
        ["出荷済みの行を取り出し、元の行数と対象の行数を確認します。", "対象の数量と単価を掛けて合計します。"],
        str(total), "円", "出荷済みの行だけを合計した請求額です。")


def spec_weighted(r: random.Random) -> Spec:
    rows = [(r.randint(5, 40), r.randint(50, 500)) for _ in range(5)]
    units = sum(q for q, _ in rows)
    total = sum(q * p for q, p in rows)
    answer = money(Fraction(total, units))
    return Spec("weighted_unit_cost", f"仕入れ表の各行は(個数,1個の価格円)です。\n{rows!r}\n全体の1個あたり平均価格を、小数点以下2桁で答えてください。行ごとの価格の単純平均ではなく、個数で重みを付けてください。端数はhalf-evenで丸めます。",
        [code(f"rows = {rows!r}\nunits = sum(q for q, price in rows)\nprint(units)"),
         code("from decimal import Decimal\ncost = sum(q * price for q, price in rows)\nprint(f'{Decimal(cost) / Decimal(units):.2f}')")],
        [f"{units}\n", answer + "\n"],
        ["各仕入れの個数を合計します。", "金額合計を総個数で割って、指定の桁数にします。"], answer, "円", "個数の違いを反映した平均価格です。")


def spec_inventory(r: random.Random) -> Spec:
    names = ["A", "B", "C", "D"]
    initial = {name: r.randint(12, 30) for name in names}
    changes = [(r.choice(names), r.randint(-8, 9)) for _ in range(8)]
    minimum = r.randint(18, 28)
    state = dict(initial)
    for name, change in changes:
        state[name] += change
    required = sum(max(0, minimum - count) for count in state.values())
    return Spec("inventory_reorder", f"商品の初期在庫は{initial!r}です。入出庫表の行は(商品,増減)で、負数は出庫です。\n{changes!r}\n各商品の在庫を最低{minimum}個にするため、全部で何個発注すればよいですか。余りを他商品へ移すことはできません。",
        [code(f"stock = {initial!r}\nchanges = {changes!r}\nfor name, change in changes:\n    stock[name] += change\nprint(len(stock))"),
         code(f"minimum = {minimum}\norder = sum(max(0, minimum - count) for count in stock.values())\nprint(order)")],
        ["4\n", f"{required}\n"],
        ["商品ごとに入出庫を反映し、在庫表を更新します。", "不足している商品だけ、必要個数を足します。"], str(required), "個", "余っている商品の在庫は不足分と相殺していません。")


def spec_units(r: random.Random) -> Spec:
    amounts = [(r.randint(1, 8), "m"), (r.randint(10, 90), "cm"), (r.randint(10, 900), "mm"),
               (r.randint(1, 5), "m"), (r.randint(5, 60), "cm")]
    scale = {"m": 1000, "cm": 10, "mm": 1}
    expected = sum(n * scale[u] for n, u in amounts)
    return Spec("mixed_length_units", f"部品の長さが次の(数値,単位)で記録されています。\n{amounts!r}\n合計をmmで求めてください。この問題では1m=1000mm、1cm=10mmです。",
        [code(f"lengths = {amounts!r}\nscale = {{'m': 1000, 'cm': 10, 'mm': 1}}\nconverted = [value * scale[unit] for value, unit in lengths]\nprint(len(converted))"),
         code("print(sum(converted))")],
        ["5\n", f"{expected}\n"],
        ["単位ごとの換算を使い、すべてmmにそろえます。", "同じ単位になった値を合計します。"], str(expected), "mm", "元の数値をそのまま加えず、単位をそろえて合計しました。")


def spec_duration(r: random.Random) -> Spec:
    pairs = []
    durations = []
    for _ in range(5):
        start = r.randint(0, 1439)
        duration = r.randint(50, 360)
        end = (start + duration) % 1440
        pairs.append((f"{start // 60:02d}:{start % 60:02d}", f"{end // 60:02d}:{end % 60:02d}"))
        durations.append(duration)
    threshold = r.choice([120, 150, 180])
    expected = sum(max(0, d - threshold) for d in durations)
    return Spec("overnight_duration", f"作業記録の各行は(開始時刻,終了時刻)です。\n{pairs!r}\n各作業は24時間未満で、終了が開始より早い表記なら翌日に終わります。1回につき{threshold}分を超えた時間の合計を分で求めてください。",
        [code(f"records = {pairs!r}\ndef minutes(value):\n    hour, minute = map(int, value.split(':'))\n    return hour * 60 + minute\ndurations = [(minutes(end) - minutes(start)) % 1440 for start, end in records]\nprint(len(durations))"),
         code(f"print(sum(max(0, value - {threshold}) for value in durations))")],
        ["5\n", f"{expected}\n"],
        ["時刻を分へ変え、日付をまたぐ記録も含めて時間を求めます。", "各作業で上限を超えた分だけを合計します。"], str(expected), "分", "作業ごとに上限を引いてから、正の時間だけを足しました。")


def spec_events(r: random.Random) -> Spec:
    records = [(f"R{i}", r.choice([200, 200, 404, 500, 503]), r.randint(10, 500)) for i in range(6)]
    records += [("R1", 500, r.randint(10, 500)), ("R4", 200, r.randint(10, 500))]
    last = {row[0]: row for row in records}
    expected = sum(v[2] for v in last.values() if v[1] >= 500)
    return Spec("request_log_dedup", f"応答ログの行は(要求ID,状態コード,時間ms)です。\n{records!r}\n同じIDがあれば後の行だけを採用してください。その後、状態コードが500以上の要求について時間の合計をmsで求めてください。",
        [code(f"records = {records!r}\nlatest = {{}}\nfor request_id, status, duration in records:\n    latest[request_id] = (status, duration)\nprint(len(latest))"),
         code("print(sum(duration for status, duration in latest.values() if status >= 500))")],
        ["6\n", f"{expected}\n"],
        ["同じIDの古い記録を置き換え、採用する要求数を確認します。", "最新の記録だけから条件に合う応答時間を合計します。"], str(expected), "ms", "古い記録や500未満の状態コードは合計に含めていません。")


def spec_refunds(r: random.Random) -> Spec:
    rows = [(r.choice(["east", "west", "north"]), r.randint(50, 900), r.choice(["sale", "sale", "return"])) for _ in range(10)]
    target = r.choice(["east", "west", "north"])
    expected = sum(amount if kind == "sale" else -amount for area, amount, kind in rows if area == target)
    distinct = len(set(x[0] for x in rows))
    return Spec("sales_returns_group", f"販売記録の行は(地域,金額円,区分)です。\n{rows!r}\nsaleは売上、returnは同じ金額の返品です。地域{target}の正味売上を円で求めてください。",
        [code(f"rows = {rows!r}\ntotals = {{}}\nfor area, amount, kind in rows:\n    change = amount if kind == 'sale' else -amount\n    totals[area] = totals.get(area, 0) + change\nprint(len(totals))"),
         code(f"print(totals.get({target!r}, 0))")],
        [f"{distinct}\n", f"{expected}\n"],
        ["返品を負の金額として、地域ごとに記録をまとめます。", "指定された地域の正味売上を取り出します。"], str(expected), "円", "指定地域の売上から同地域の返品だけを引いた金額です。")


def spec_percentile(r: random.Random) -> Spec:
    values = [r.randint(10, 900) for _ in range(r.randint(12, 20))]
    percent = r.choice([80, 90, 95])
    rank = (len(values) * percent + 99) // 100
    expected = sorted(values)[rank - 1]
    return Spec("nearest_rank_percentile", f"応答時間msは{values!r}です。{percent}パーセンタイルを求めてください。この問題では小さい順に並べ、件数×{percent}/100を切り上げた順位(1始まり)の値を使います。補間はしません。",
        [code(f"values = {values!r}\nordered = sorted(values)\nprint(len(ordered))"),
         code(f"rank = (len(ordered) * {percent} + 99) // 100\nprint(ordered[rank - 1])")],
        [f"{len(values)}\n", f"{expected}\n"],
        ["応答時間を小さい順に並べ、件数を確認します。", "指定された順位の定義で値を取り出します。"], str(expected), "ms", "補間を使わず、問題で指定された順位の値です。")


def spec_sensor(r: random.Random) -> Spec:
    low, high = 0, 100
    values = [str(r.randint(0, 100)) for _ in range(7)] + ["", "missing", "-3", "105"]
    r.shuffle(values)
    valid = sorted(int(x) for x in values if x not in ("", "missing") and low <= int(x) <= high)
    expected = money(Fraction(valid[len(valid) // 2]))
    return Spec("sensor_missing_median", f"センサーの読み値は文字列の表{values!r}です。空文字とmissingは欠測で、0以上100以下だけが有効です。有効値の中央値を小数点以下2桁で求めてください。",
        [code(f"raw = {values!r}\nvalid = []\nfor item in raw:\n    if item in ('', 'missing'):\n        continue\n    value = int(item)\n    if 0 <= value <= 100:\n        valid.append(value)\nprint(len(valid))"),
         code("from statistics import median\nprint(f'{median(valid):.2f}')")],
        ["7\n", expected + "\n"],
        ["欠測と範囲外の値を除き、有効な測定数を確認します。", "有効な値だけから中央値を求めます。"], expected, "", "欠測や範囲外の値を0として扱ってはいません。")


def spec_packages(r: random.Random) -> Spec:
    rows = [(r.choice(["A", "B", "C"]), r.randint(1, 19)) for _ in range(9)]
    capacity = {"A": r.randint(4, 8), "B": r.randint(5, 10), "C": r.randint(3, 7)}
    grouped = Counter()
    for name, qty in rows:
        grouped[name] += qty
    expected = sum((count + capacity[name] - 1) // capacity[name] for name, count in grouped.items())
    return Spec("shipping_boxes", f"注文の行は(商品,個数)です。\n{rows!r}\n1箱に入る個数は{capacity!r}です。同じ商品はまとめてよく、違う商品は同じ箱に入れません。最後の箱が満杯でなくても1箱と数えます。必要な箱数の合計はいくつですか。",
        [code(f"rows = {rows!r}\ncapacity = {capacity!r}\ncounts = {{}}\nfor name, count in rows:\n    counts[name] = counts.get(name, 0) + count\nprint(sum(counts.values()))"),
         code("boxes = sum((count + capacity[name] - 1) // capacity[name] for name, count in counts.items())\nprint(boxes)")],
        [f"{sum(grouped.values())}\n", f"{expected}\n"],
        ["同じ商品の注文をまとめ、総個数を確認します。", "商品ごとの箱数を切り上げてから合計します。"], str(expected), "箱", "異なる商品を同じ箱に混ぜずに求めた箱数です。")


def spec_repair_boundary(r: random.Random) -> Spec:
    threshold = r.randint(5, 30)
    values = [r.randint(0, 50) for _ in range(8)] + [threshold, threshold]
    expected = sum(v >= threshold for v in values)
    broken = code("def count_at_least(values, threshold):\n    return sum(value > threshold for value in values)\nassert count_at_least([2, 3, 4], 3) == 2")
    return Spec("repair_inclusive_boundary", f"count_at_leastはthreshold以上の要素数を返す関数ですが、下のテストに失敗しました。等しい値も含めるよう直し、{values!r}にthreshold={threshold}を指定した結果を答えてください。",
        [code("def count_at_least(values, threshold):\n    return sum(value >= threshold for value in values)\nassert count_at_least([2, 3, 4], 3) == 2\nassert count_at_least([], 3) == 0\nassert count_at_least([3, 3], 3) == 2\nprint('tests: 3 passed')"),
         code(f"print(count_at_least({values!r}, {threshold}))")],
        ["tests: 3 passed\n", f"{expected}\n"],
        ["境界値を含む比較へ直し、空の入力と同じ値の重なりも確認します。", "確認した関数を指定の入力に適用します。"], str(expected), "個", "しきい値と等しい値も個数に含めています。", broken, "AssertionError")


def spec_repair_group(r: random.Random) -> Spec:
    rows = [(r.choice(["A", "B", "C"]), r.randint(-50, 200)) for _ in range(9)]
    target = r.choice(["A", "B", "C"])
    expected = sum(amount for name, amount in rows if name == target)
    broken = code("def totals(rows):\n    result = {}\n    for key, amount in rows:\n        result[key] = amount\n    return result\nassert totals([('A', 2), ('A', 3)]) == {'A': 5}")
    return Spec("repair_group_accumulation", f"同じ店の金額を合計するtotalsが、下のテストに失敗しました。修正し、表{rows!r}に対する店{target}の合計額を答えてください。負数は返金です。店がなければ0とします。",
        [code("def totals(rows):\n    result = {}\n    for key, amount in rows:\n        result[key] = result.get(key, 0) + amount\n    return result\nassert totals([('A', 2), ('A', 3)]) == {'A': 5}\nassert totals([('B', 5), ('B', -2)]) == {'B': 3}\nassert totals([]) == {}\nprint('tests: 3 passed')"),
         code(f"result = totals({rows!r})\nprint(result.get({target!r}, 0))")],
        ["tests: 3 passed\n", f"{expected}\n"],
        ["後の金額で置き換えずに加算するよう直して、複数の入力で確認します。", "修正した関数で集計し、指定された店の合計を出します。"], str(expected), "円", "同じ店のすべての行を足し、返金を引いた金額です。", broken, "AssertionError")


def spec_repair_sort(r: random.Random) -> Spec:
    values = [str(r.randint(-20, 150)) for _ in range(9)]
    position = r.randint(2, 7)
    expected = sorted(map(int, values))[position - 1]
    broken = code("def order_numbers(values):\n    return sorted(values)\nassert order_numbers(['2', '10', '1']) == ['1', '2', '10']")
    return Spec("repair_numeric_sort", f"数値を表す文字列を並べるorder_numbersが、下のテストに失敗しました。小さい数から並ぶよう修正し、入力{values!r}で{position}番目(1始まり)となる数を答えてください。重なった値も別の要素と数えます。",
        [code("def order_numbers(values):\n    return sorted(values, key=int)\nassert order_numbers(['2', '10', '1']) == ['1', '2', '10']\nassert order_numbers(['0', '-3', '2']) == ['-3', '0', '2']\nassert order_numbers(['2', '2']) == ['2', '2']\nprint('tests: 3 passed')"),
         code(f"ordered = order_numbers({values!r})\nprint(ordered[{position - 1}])")],
        ["tests: 3 passed\n", f"{expected}\n"],
        ["文字列の順序ではなく、整数にした値で並べるように直します。", "重なりを残したまま、指定の順位の値を取り出します。"], str(expected), "", "文字列のまま比較した順序とは区別して求めました。", broken, "AssertionError")


def spec_intervals(r: random.Random) -> Spec:
    periods = []
    for _ in range(7):
        start = r.randint(0, 150)
        periods.append((start, start + r.randint(10, 50)))
    expected = sum(max(a, c) < min(b, d) for i, (a, b) in enumerate(periods) for c, d in periods[i + 1:])
    return Spec("schedule_conflicts", f"会議室の予約を(開始分,終了分)で示します。\n{periods!r}\n開始は含み、終了は含まないものとします。重なりがある予約の組を何組見つけられますか。同じ組を2回数えないでください。",
        [code(f"periods = {periods!r}\nperiods.sort()\nassert all(start < end for start, end in periods)\nprint(len(periods))"),
         code("conflicts = 0\nfor i, (start, end) in enumerate(periods):\n    for other_start, other_end in periods[i + 1:]:\n        if max(start, other_start) < min(end, other_end):\n            conflicts += 1\nprint(conflicts)")],
        ["7\n", f"{expected}\n"],
        ["予約を開始順に並べ、それぞれの時間が有効か確認します。", "終了と開始が接するだけの組を除いて、重なりを数えます。"], str(expected), "組", "1つの予約の組につき1回だけ数えています。")


def spec_priority(r: random.Random) -> Spec:
    jobs = [(chr(65 + i), r.randint(1, 4), r.randint(10, 70)) for i in range(7)]
    count = r.randint(2, 5)
    selected = sorted(jobs, key=lambda x: (-x[1], x[0]))[:count]
    expected = sum(x[2] for x in selected)
    return Spec("priority_job_selection", f"作業表の行は(名前,優先度,作業時間分)です。\n{jobs!r}\n優先度が大きい順、同じ優先度なら名前のアルファベット順に{count}件を選びます。選んだ作業の時間合計を求めてください。",
        [code(f"jobs = {jobs!r}\nordered = sorted(jobs, key=lambda row: (-row[1], row[0]))\nprint(len(ordered))"),
         code(f"selected = ordered[:{count}]\nprint(sum(row[2] for row in selected))")],
        ["7\n", f"{expected}\n"],
        ["優先度と同順位の扱いに従って作業を並べます。", "指定件数を選び、その時間だけを足します。"], str(expected), "分", "作業時間の短い順ではなく、指定された優先順位で選びました。")


def spec_negative(r: random.Random) -> Spec:
    records = [(f"P{i}", r.choice([0, 1]), r.randint(0, 12), r.randint(20, 300)) for i in range(8)]
    days = r.randint(3, 8)
    accepted = sum(active == 1 and age <= days for _, active, age, _ in records)
    expected = sum(cost for _, active, age, cost in records if not (active == 1 and age <= days))
    return Spec("negative_condition_filter", f"記録の行は(ID,有効フラグ,経過日数,金額円)です。\n{records!r}\n「有効フラグが1、かつ経過日数が{days}以下」という条件を満たさない行の金額を合計してください。条件を満たす行は含めないでください。",
        [code(f"rows = {records!r}\neligible = [active == 1 and age <= {days} for _, active, age, amount in rows]\nprint(sum(eligible))"),
         code("print(sum(row[3] for row, accepted in zip(rows, eligible) if not accepted))")],
        [f"{accepted}\n", f"{expected}\n"],
        ["まず条件を満たす行を判定し、その件数を確認します。", "今回は反対側が対象なので、条件を満たさない行だけを合計します。"], str(expected), "円", "有効かつ日数内という条件の全体を否定して選んだ合計です。")


# These four algorithms have no generated train or dev rows in this package.
def spec_interval_union(r: random.Random) -> Spec:
    periods = [(start, start + r.randint(8, 30)) for start in [r.randint(0, 100) for _ in range(7)]]
    covered = set()
    for start, end in periods:
        covered.update(range(start, end))
    expected = len(covered)
    return Spec("heldout_interval_union", f"機械の稼働時間を(開始分,終了分)で記録しました。\n{periods!r}\n重なる時間を二重に数えず、少なくとも1台が稼働した時間の合計を分で求めてください。開始は含み、終了は含みません。",
        [code(f"periods = sorted({periods!r})\nprint(len(periods))"),
         code("merged = []\nfor start, end in periods:\n    if merged and start <= merged[-1][1]:\n        merged[-1][1] = max(merged[-1][1], end)\n    else:\n        merged.append([start, end])\nprint(sum(end - start for start, end in merged))")],
        ["7\n", f"{expected}\n"],
        ["記録を開始時刻の順に整理します。", "重なる区間を結合してから時間を合計します。"], str(expected), "分", "同時に複数台が動いた時間を重ねて加えていません。")


def spec_rolling(r: random.Random) -> Spec:
    values = [r.randint(1, 50) for _ in range(12)]
    width = r.randint(3, 5)
    expected = max(sum(values[i:i + width]) for i in range(len(values) - width + 1))
    return Spec("heldout_rolling_window", f"日ごとの処理件数は順に{values!r}です。連続するちょうど{width}日間の合計が最大になるとき、その合計件数はいくつですか。日を並べ替えたり、間の日を飛ばしたりしないでください。",
        [code(f"values = {values!r}\nwidth = {width}\nassert len(values) >= width\nprint(len(values) - width + 1)"),
         code("prefix = [0]\nfor value in values:\n    prefix.append(prefix[-1] + value)\nprint(max(prefix[i + width] - prefix[i] for i in range(len(values) - width + 1)))")],
        [f"{len(values) - width + 1}\n", f"{expected}\n"],
        ["条件に合う連続区間がいくつあるか確認します。", "順序を保った区間ごとの合計を比べます。"], str(expected), "件", "連続する日だけを対象にした最大の合計です。")


def spec_dependency(r: random.Random) -> Spec:
    durations = {name: r.randint(5, 30) for name in "ABCDE"}
    deps = {"A": [], "B": [], "C": ["A"], "D": ["A", "B"], "E": ["C", "D"]}
    expected = max(durations["A"] + durations["C"], max(durations["A"], durations["B"]) + durations["D"]) + durations["E"]
    return Spec("heldout_dependency_schedule", f"作業時間(分)は{durations!r}です。開始前に終えておく作業は{deps!r}です。時刻0から開始し、条件を満たした作業は同時に進められます。作業Eが最も早く終了する時刻を求めてください。",
        [code(f"duration = {durations!r}\ndependencies = {deps!r}\norder = ['A', 'B', 'C', 'D', 'E']\nseen = set()\nfor task in order:\n    assert all(parent in seen for parent in dependencies[task])\n    seen.add(task)\nprint(len(order))"),
         code("finished = {}\nfor task in order:\n    start = max([finished[parent] for parent in dependencies[task]] + [0])\n    finished[task] = start + duration[task]\nprint(finished['E'])")],
        ["5\n", f"{expected}\n"],
        ["前に必要な作業が先に並んでいるか確認します。", "依存する作業の終了を待って開始する時刻を順に求めます。"], str(expected), "分", "独立した作業を同時に進められる条件を反映した終了時刻です。")


def spec_distance(r: random.Random) -> Spec:
    chars = "ABCDE"
    first = "".join(r.choice(chars) for _ in range(7))
    second = first[:2] + r.choice(chars) + first[3:5] + r.choice(chars) + first[5:]
    # Independent recursive reference with memoization, unlike row-wise DP below.
    from functools import lru_cache
    @lru_cache(None)
    def distance(i: int, j: int) -> int:
        if i == 0:
            return j
        if j == 0:
            return i
        return min(distance(i - 1, j) + 1, distance(i, j - 1) + 1,
                   distance(i - 1, j - 1) + (first[i - 1] != second[j - 1]))
    expected = distance(len(first), len(second))
    return Spec("heldout_text_edit_distance", f"商品コード{first!r}を{second!r}に直します。1文字の追加、削除、別の文字への置換をそれぞれ1回と数えます。順序の入れ替えは1回の操作にはできません。必要な操作の最小回数を求めてください。",
        [code(f"first = {first!r}\nsecond = {second!r}\nprint(len(first), len(second))"),
         code("previous = list(range(len(second) + 1))\nfor i, a in enumerate(first, 1):\n    current = [i]\n    for j, b in enumerate(second, 1):\n        current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))\n    previous = current\nprint(previous[-1])")],
        [f"{len(first)} {len(second)}\n", f"{expected}\n"],
        ["2つのコードの長さを確認します。", "追加、削除、置換に分けて最少の操作数を求めます。"], str(expected), "回", "文字の位置ごとに必要な最小操作数を比較した結果です。")


TRAIN_FAMILIES = [spec_invoice, spec_weighted, spec_inventory, spec_units, spec_duration,
                  spec_events, spec_refunds, spec_percentile, spec_sensor, spec_packages,
                  spec_repair_boundary, spec_repair_group, spec_repair_sort, spec_intervals,
                  spec_priority, spec_negative]
HELDOUT_FAMILIES = [spec_interval_union, spec_rolling, spec_dependency, spec_distance]


def normalized_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def fingerprint(text: str) -> str:
    return hashlib.sha256(normalized_prompt(text).encode("utf-8")).hexdigest()


def extract_prompts(obj):
    if isinstance(obj, dict):
        for key in ("prompt", "question", "input", "question_text"):
            if isinstance(obj.get(key), str):
                yield obj[key]
        for key, value in obj.items():
            if key not in ("prompt", "question", "input", "question_text", "full_text", "target"):
                if isinstance(value, (dict, list)):
                    yield from extract_prompts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from extract_prompts(value)


def heldout_fingerprints(paths: list[Path]) -> tuple[set[str], dict]:
    hashes = set()
    manifest = {}
    for path in paths:
        blob = path.read_bytes()
        if path.suffix.lower() == ".jsonl":
            objects = [json.loads(line) for line in blob.splitlines() if line.strip()]
        else:
            objects = [json.loads(blob)]
        values = [prompt for item in objects for prompt in extract_prompts(item)]
        hashes.update(fingerprint(prompt) for prompt in values)
        manifest[str(path)] = {"sha256": hashlib.sha256(blob).hexdigest(), "prompt_fields": len(values)}
    return hashes, manifest


def literal_normalizer(source: str):
    tree = ast.parse(source)
    nodes = [n.value for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "_BASE_NORM_DICT" for t in n.targets)]
    if len(nodes) != 1:
        raise ValueError("Original normalizer definition not found")
    mapping = ast.literal_eval(nodes[0])
    for number in range(0xFF01, 0xFF5F):
        char = chr(number)
        if char not in ("（", "）", "！", "？", "：", "～"):
            mapping[char] = chr(number - 0xFEE0)
    return str.maketrans(mapping)


def validate_row(row: dict, vocab: dict, normalizer) -> None:
    prompt, target, full = row["prompt"], row["target"], row["full_text"]
    if full != prompt + target or not target:
        raise ValueError(f"{row['id']}: prompt/target mismatch")
    missing = sorted(set(full) - set(vocab))
    if missing:
        raise ValueError(f"{row['id']}: vocabulary missing {missing!r}")
    if full.translate(normalizer) != full:
        raise ValueError(f"{row['id']}: runtime normalization changes text")
    if len(full) + 1 > MAX_CONTEXT:
        raise ValueError(f"{row['id']}: {len(full) + 1} characters incl EOS exceed {MAX_CONTEXT}")
    spans = row["loss_mask_spans"]
    if len(spans) != len(row["tool_steps"]):
        raise ValueError(f"{row['id']}: observation count mismatch")
    previous = len(prompt)
    for span, step in zip(spans, row["tool_steps"]):
        start, end = span
        if not previous <= start < end <= len(full):
            raise ValueError(f"{row['id']}: invalid mask boundaries")
        if full[start:end] != step["observation"]:
            raise ValueError(f"{row['id']}: observation/mask mismatch")
        previous = end
    # All target observation blocks, and only observations, must be masked.
    obs = list(re.finditer(r"\n```output\n.*?\n```\n", full[len(prompt):], re.S))
    observed = [[len(prompt) + m.start(), len(prompt) + m.end()] for m in obs]
    if observed != spans:
        raise ValueError(f"{row['id']}: target output fence not masked exactly")
    if not row["reference_final_answer"] or row["reference"]["kind"] != "numeric":
        raise ValueError(f"{row['id']}: missing final answer")
    Decimal(row["reference"]["value"])


def make_row(spec: Spec, split: str, scenario: int, heldout_family: bool,
             executor, observation_text, vocab: dict, normalizer) -> dict:
    prompt = "問題: " + spec.prompt
    context_calls = 0
    if spec.context_code is not None:
        executor.reset()
        result = executor.execute(spec.context_code)
        if result.get("exception_type") != spec.context_exception or result.get("success"):
            raise ValueError(f"{spec.family}: expected diagnostic failure not reproduced")
        prompt += "\n元のコードと実行結果:\n```python\n" + spec.context_code + "```"
        prompt += observation_text(result)
        context_calls += 1
    prompt += "\n道具: Python\n思考:"
    prompt = prompt.translate(normalizer)
    executor.reset()  # Code shown in the prompt is never implicitly executed.
    full = prompt
    spans = []
    steps = []
    for index, (source, expected, action) in enumerate(zip(spec.codes, spec.expected_stdout, spec.actions)):
        source = source.translate(normalizer)
        full += "\n" + action + "\n```python\n" + source + "```"
        result = executor.execute(source)
        if (not result.get("success") or result.get("stdout") != expected or
                result.get("stderr") or result.get("truncated")):
            raise ValueError(f"{spec.family}/{scenario} step {index}: reference and native result differ: {result!r} vs {expected!r}")
        observation = observation_text(result)
        start = len(full)
        full += observation
        spans.append([start, len(full)])
        steps.append({"index": index, "code": source, "success": True,
                      "stdout": result["stdout"], "observation": observation})
    answer = spec.answer + (" " + spec.unit if spec.unit else "")
    full += "答え: " + answer + "。" + spec.conclusion
    full = full.translate(normalizer)
    row_id = f"practical_v1_{spec.family}_{scenario:03d}"
    row = {
        "id": row_id, "source": "selfauthored:geocpu/practical-python/v1",
        "source_kind": "newly_authored_parameterized_practical_tasks",
        "license": "CC0-1.0", "family": spec.family,
        "group_id": f"scenario:{spec.family}:{scenario:03d}",
        "family_partition": "unseen_family" if heldout_family else "known_family",
        "split": split, "_v46_group": GROUP, "scenario_index": scenario,
        "prompt": prompt, "target": full[len(prompt):], "full_text": full,
        "loss_mask_spans": spans, "tool_steps": steps,
        "reference": {"kind": "numeric", "value": spec.answer, "unit": spec.unit},
        "reference_final_answer": answer,
        "final_answer_line": full[spans[-1][1]:],
        "verification": {"method": "exact_v47_cpu_executor_vs_independent_reference",
                         "target_python_calls": len(steps), "prompt_diagnostic_calls": context_calls,
                         "final_stdout_matches_reference": steps[-1]["stdout"].strip() == spec.answer,
                         "gpu_model_evaluations": 0},
    }
    validate_row(row, vocab, normalizer)
    return row


def read_r6(path: Path) -> bytes:
    blob = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        if "assets/r6_bundle.zip" in archive.namelist():
            blob = archive.read("assets/r6_bundle.zip")
    if hashlib.sha256(blob).hexdigest() != R6_SHA256:
        raise ValueError("The v47 original R6 runtime bundle hash differs")
    return blob


def build(r6_bundle: Path, output: Path, heldout_paths: list[Path] | None = None,
          train_per_family: int = 32, dev_per_family: int = 8, eval_per_family: int = 12,
          frozen_root: Path | None = None) -> dict:
    if min(train_per_family, dev_per_family, eval_per_family) < 1:
        raise ValueError("Every split needs at least one case per applicable family")
    names = {split: f"practical_{split}.jsonl" for split in ("train", "dev", "expanded_eval")}
    outputs = [output / name for name in names.values()] + [output / "practical_manifest.json"]
    if any(path.exists() for path in outputs):
        raise FileExistsError("Curriculum output already exists; choose a new output directory")
    r6_blob = read_r6(r6_bundle)
    external, heldout_manifest = heldout_fingerprints(heldout_paths or [])
    frozen_root = frozen_root or Path(__file__).resolve().parent / "frozen"
    if (frozen_root / "registry.json").is_file():
        import frozen_eval
        count = 0
        for record in frozen_eval.iter_heldout_prompts(frozen_root):
            external.add(fingerprint(record["prompt"]))
            count += 1
        heldout_manifest["frozen_evaluation_registry"] = {
            "sha256": hashlib.sha256((frozen_root / "registry.json").read_bytes()).hexdigest(),
            "prompt_fields": count, "includes_T8_profiles": True,
            "source_inventory_sha256": hashlib.sha256((frozen_root / "source_files.json").read_bytes()).hexdigest(),
        }
    # Original T1-T7 data are always checked directly from the pinned archive.
    with zipfile.ZipFile(io.BytesIO(r6_blob)) as original:
        vocab = json.loads(original.read("base_metadata.json"))["vocab"]
        normalizer = literal_normalizer(original.read("qwenputer_controller_v30.py").decode("utf-8"))
        original_files = [n for n in original.namelist() if n.startswith("data/") and
                          ("eval_" in n or "manifest" in n) and n.endswith(".jsonl")]
        for name in original_files:
            blob = original.read(name)
            fields = [prompt for line in blob.splitlines() if line.strip()
                      for prompt in extract_prompts(json.loads(line))]
            external.update(fingerprint(p) for p in fields)
            heldout_manifest[f"r6_bundle.zip:{name}"] = {
                "sha256": hashlib.sha256(blob).hexdigest(), "prompt_fields": len(fields)}
        with tempfile.TemporaryDirectory(prefix="geocpu_curriculum_runtime_") as temporary:
            temp = Path(temporary)
            for filename in ("tir_engine_v44r1.py", "executor_worker_v44r1.py"):
                (temp / filename).write_bytes(original.read(filename))
            module_spec = importlib.util.spec_from_file_location("curriculum_native_tir", temp / "tir_engine_v44r1.py")
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_spec.name] = module
            module_spec.loader.exec_module(module)
            rows = {split: [] for split in names}
            seen = set()
            with module.PythonExecutor(5.0, 1024, 32768, startup_timeout_seconds=180) as engine:
                for factory in TRAIN_FAMILIES + HELDOUT_FAMILIES:
                    family = factory(random.Random(1)).family
                    only_eval = factory in HELDOUT_FAMILIES
                    schedule = [("expanded_eval", i) for i in range(eval_per_family)] if only_eval else (
                        [("train", i) for i in range(train_per_family)] +
                        [("dev", i) for i in range(train_per_family, train_per_family + dev_per_family)] +
                        [("expanded_eval", i) for i in range(train_per_family + dev_per_family,
                         train_per_family + dev_per_family + eval_per_family)])
                    for split, scenario in schedule:
                        spec = factory(rng_for(family, scenario))
                        row = make_row(spec, split, scenario, only_eval, engine, module.observation_text, vocab, normalizer)
                        digest = fingerprint(row["prompt"])
                        if digest in external:
                            raise ValueError(f"Exact normalized overlap with frozen tests: {row['id']}")
                        if digest in seen:
                            raise ValueError(f"Duplicate prompt across generated cases: {row['id']}")
                        seen.add(digest)
                        rows[split].append(row)
                    print(f"CURRICULUM_VERIFIED family={family} rows={len(schedule)} model_updates=0", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    file_info = {}
    for split, filename in names.items():
        blob = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                       for row in rows[split]).encode("utf-8")
        path = output / filename
        with path.open("xb") as sink:
            sink.write(blob)
        file_info[split] = {"file": filename, "rows": len(rows[split]),
                           "sha256": hashlib.sha256(blob).hexdigest(), "size_bytes": len(blob)}
    all_rows = [row for split_rows in rows.values() for row in split_rows]
    manifest = {
        "version": VERSION, "seed": SEED, "runtime_r6_sha256": R6_SHA256,
        "vocab_size": len(vocab), "max_context_including_eos": MAX_CONTEXT,
        "files": file_info, "train_families": [f(random.Random(1)).family for f in TRAIN_FAMILIES],
        "unseen_eval_families": [f(random.Random(1)).family for f in HELDOUT_FAMILIES],
        "generated_rows": len(all_rows), "unique_normalized_prompts": len(seen),
        "target_python_calls_executed": sum(len(row["tool_steps"]) for row in all_rows),
        "prompt_diagnostic_calls_executed": sum(row["verification"]["prompt_diagnostic_calls"] for row in all_rows),
        "max_observed_context_including_eos": max(len(row["full_text"]) + 1 for row in all_rows),
        "generation_counts": {"train_per_family": train_per_family,
                              "dev_per_family": dev_per_family,
                              "eval_per_family": eval_per_family},
        "token_accounting": {
            split: {"context_tokens_including_eos": sum(len(row["full_text"]) + 1 for row in items),
                    "masked_observation_tokens": sum(b - a for row in items for a, b in row["loss_mask_spans"]),
                    "supervised_target_tokens_including_eos": sum(len(row["target"]) + 1 -
                        sum(b - a for a, b in row["loss_mask_spans"]) for row in items)}
            for split, items in rows.items()},
        "unknown_characters": [], "normalized_prompt_overlap_with_checked_heldouts": 0,
        "checked_heldouts": heldout_manifest, "split_unit": "scenario/input-table/test-case group",
        "family_counts": {split: dict(Counter(row["family"] for row in items)) for split, items in rows.items()},
        "eval_partition_counts": dict(Counter(row["family_partition"] for row in rows["expanded_eval"])),
        "model_training_updates": 0, "gpu_model_evaluations": 0,
        "limitations": [
            "Known-family dev/eval share the algorithm template with training; they are input-disjoint, not template-disjoint.",
            "Only the four unseen_eval_families are absent from this new training/development curriculum; previous training is not audited for those concepts.",
            "Parameterized table/code variants are not independent real-world tasks or independent knowledge facts.",
            "All observations and numeric references are CPU verified. This does not measure the model's answers or guarantee improvement.",
            "Exact normalized prompt checks do not prove absence of semantic overlap or paraphrases.",
            "No original T1-T8 items or scores are changed by this generator.",
        ],
    }
    with (output / "practical_manifest.json").open("x", encoding="utf-8") as sink:
        json.dump(manifest, sink, ensure_ascii=False, indent=2)
        sink.write("\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r6-bundle", type=Path, required=True,
                        help="Pinned r6_bundle.zip or the outer v47 base bundle")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; existing files are never replaced")
    parser.add_argument("--heldout", type=Path, action="append", default=[], help="Additional frozen JSON/JSONL to check for normalized prompt overlap")
    parser.add_argument("--frozen-root", type=Path, help="Frozen T1-T8/additional evaluator registry; defaults to this package's frozen folder")
    parser.add_argument("--train-per-family", type=int, default=32)
    parser.add_argument("--dev-per-family", type=int, default=8)
    parser.add_argument("--eval-per-family", type=int, default=12)
    args = parser.parse_args()
    manifest = build(args.r6_bundle, args.output, args.heldout,
                     args.train_per_family, args.dev_per_family, args.eval_per_family, args.frozen_root)
    print(json.dumps({"CURRICULUM_COMPLETE": manifest["files"],
                      "cpu_executed_target_calls": manifest["target_python_calls_executed"],
                      "model_training_updates": 0, "gpu_model_evaluations": 0}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
