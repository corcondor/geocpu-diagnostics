#!/usr/bin/env python3
"""Deterministic, executable v48 *candidate* curriculum. No training or Drive writes.

This produces train/dev/test rows; never append to an existing training file here.
Real-world knowledge must come from licensed sources via a separate ingestion pass.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import random
import re
from pathlib import Path

FAMILIES = ('csv_revenue', 'json_join', 'code_repair', 'log_window', 'evidence_chain')
CONFIG = {'train': 128, 'dev': 32, 'test': 32}  # per family: 640/160/160, 512 Python train


def evaluate(source):
    # All source strings below are constructed by this module, not supplied by a user.
    tree = ast.parse(source, mode='exec')
    if any(isinstance(x, (ast.ImportFrom, ast.Global, ast.Nonlocal)) for x in ast.walk(tree)):
        raise ValueError('unsafe generated trace')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        env = {'__builtins__': __builtins__}
        exec(compile(tree, '<generated-reference>', 'exec'), env, env)
    return buf.getvalue().strip()


def make_python_case(family, rng, case_id):
    if family == 'csv_revenue':
        region = rng.choice(['east', 'west', 'north'])
        records = [(rng.choice(['east', 'west', 'north']), rng.randint(1, 9), rng.randint(11, 49)*10,
                    rng.choice(['paid', 'paid', 'pending'])) for _ in range(7)]
        records[0] = (region, rng.randint(1, 9), rng.randint(11, 49)*10, 'paid')
        csv_data = 'region,units,price,status\\n' + '\\n'.join(','.join(map(str, r)) for r in records)
        # Make literal LF characters in the evidence text; Python code uses a repr.
        data = csv_data.replace('\\n', '\n')
        goal = sum(u*p for r, u, p, s in records if r == region and s == 'paid')
        prompt = f'業務CSVのうち、地域 {region} で決済済みの売上金額の合計を求め、円で説明してください。\nCSV:\n{data}\n道具: Python\n思考:'
        code = ('import csv, io\n'
                f'rows = list(csv.DictReader(io.StringIO({data!r})))\n'
                f'total = sum(int(r["units"]) * int(r["price"]) for r in rows if r["region"] == {region!r} and r["status"] == "paid")\n'
                'print(total)')
        answer = f'地域 {region} の決済済みレコードだけを抽出して集計しました。合計は {goal} 円です。'
    elif family == 'json_join':
        tiers = {f'u{i}': rng.choice(['gold', 'silver']) for i in range(1, 5)}
        orders = [{'user': rng.choice(list(tiers)), 'price': rng.randint(9, 70)*10,
                   'qty': rng.randint(1, 5)} for _ in range(7)]
        goal = sum(o['price']*o['qty']*(9 if tiers[o['user']] == 'gold' else 10)//10 for o in orders)
        prompt = ('顧客のランクと注文JSONを結合し、goldのみ10%割引した合計請求額を円で求めてください。'
                  f'\n顧客: {json.dumps(tiers, ensure_ascii=False)}'
                  f'\n注文: {json.dumps(orders, ensure_ascii=False)}\n道具: Python\n思考:')
        code = (f'customers = {tiers!r}\norders = {orders!r}\n'
                'amount = sum(o["price"] * o["qty"] * (9 if customers[o["user"]] == "gold" else 10) // 10 for o in orders)\n'
                'print(amount)')
        answer = f'顧客IDからランクを引き、注文を結合して割引後の金額を集計しました。合計は {goal} 円です。'
    elif family == 'code_repair':
        numbers = [rng.randint(0, 40) for _ in range(8)]
        cutoff = rng.randint(10, 25)
        goal = sum(x for x in numbers if x >= cutoff)
        buggy = 'def total(values, threshold):\n    return sum(x for x in values[:-1] if x > threshold)'
        prompt = (f'次の関数は末尾の値を落とし、境界値を除外する不具合があります。{cutoff}以上の値をすべて足す関数に修正し、'
                  f'{numbers}を入力した結果を示してください。\n誤ったコード:\n{buggy}\n道具: Python\n思考:')
        code = ('def total(values, threshold):\n    return sum(x for x in values if x >= threshold)\n'
                f'print(total({numbers!r}, {cutoff}))')
        answer = f'末尾の取りこぼしと比較条件を修正して検証しました。修正後の合計は {goal} です。'
    elif family == 'log_window':
        logs = sorted([(rng.randint(0, 59), rng.choice(['error', 'ok', 'ok'])) for _ in range(10)])
        start = rng.randint(5, 24); end = start + rng.randint(12, 25)
        selected = [s for m, s in logs if start <= m < end]
        err = selected.count('error'); total = len(selected)
        text = '\n'.join(f'10:{m:02d} {s}' for m, s in logs)
        prompt = (f'10:{start:02d}以上10:{end:02d}未満のログだけを対象に、error件数/全件数を求めてください。'
                  f'\nログ:\n{text}\n道具: Python\n思考:')
        code = (f'rows = {logs!r}\n'
                f'selected = [status for minute, status in rows if {start} <= minute < {end}]\n'
                'print(f"{selected.count(\'error\')}/{len(selected)}")')
        goal = f'{err}/{total}'
        answer = f'時刻範囲を半開区間として抽出しました。error件数/全件数は {goal} です。'
    else:
        raise ValueError(family)
    observed = evaluate(code)
    if observed != str(goal):
        raise AssertionError((case_id, observed, goal))
    target = f'\n```python\n{code}\n```\n```output\n{observed}\n```\n{answer}'
    return prompt, target, 'verified_python_trace'


def make_evidence_case(rng, case_id):
    # Fictional records; no claims about real-world general knowledge.
    customer = f'C{rng.randrange(1000,9999)}'
    team = f'T{rng.randrange(100,999)}'
    reviewer = f'R{rng.randrange(100,999)}'
    location = rng.choice(['札幌', '福岡', '名古屋', '仙台', '広島'])
    distractor = rng.choice(['東京', '大阪', '京都'])
    docs = [f'[資料A] 顧客 {customer} の担当チームは {team} です。',
            f'[資料B] チーム {team} の承認担当は {reviewer} です。',
            f'[資料C] 承認担当 {reviewer} の拠点は {location} です。',
            f'[資料D] 別の顧客の承認担当の拠点は {distractor} です。']
    rng.shuffle(docs)
    prompt = (f'以下の仮想業務資料だけを根拠に、顧客 {customer} の承認担当の拠点を答え、'
              '参照した3件を順に説明してください。\n' + '\n'.join(docs) + '\n思考:')
    target = (f'\n顧客 {customer} から資料Aでチーム {team} を特定し、資料Bで承認担当 {reviewer} を特定します。'
              f'資料Cで拠点が {location} と分かります。結論: {location}。')
    return prompt, target, 'fictional_evidence_reasoning'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def generate(folder: Path, seed: int = 4807):
    folder.mkdir(parents=True, exist_ok=True)
    report = {'seed': seed, 'counts': {}, 'source': 'procedurally generated fictional tasks', 'model_updates': 0,
              'note': 'Candidate only. No benchmark leakage scan or production admission performed.'}
    seen = set()
    for split, count in CONFIG.items():
        path = folder / f'v48_{split}.jsonl'
        with path.open('x', encoding='utf-8') as out:
            families = {}
            for j, family in enumerate(FAMILIES):
                for i in range(count):
                    # stable generation independent of iteration order
                    local_seed = int.from_bytes(hashlib.sha256(f'{seed}:{split}:{family}:{i}'.encode()).digest()[:8], 'big')
                    rng = random.Random(local_seed)
                    name = f'v48:{split}:{family}:{i:04d}'
                    prompt, target, verification = (make_evidence_case(rng, name) if family == 'evidence_chain'
                                                    else make_python_case(family, rng, name))
                    if prompt in seen:
                        raise AssertionError('cross split duplicate prompt')
                    seen.add(prompt)
                    full = prompt + target
                    if len(full) > 2048:
                        raise AssertionError('context budget exceeded')
                    mask = [[prompt.__len__() + m.start(), prompt.__len__() + m.end()]
                            for m in re.finditer(r'(?m)^```output[ \t]*\n.*?^```[ \t]*(?:\n|$)', target, re.S)]
                    if family != 'evidence_chain' and len(mask) != 1:
                        raise AssertionError('Python observation must be masked')
                    row = {'id': name, 'source_problem_id': name, 'source': 'geocpu_v48_procedural',
                           'license': 'CC0-1.0', 'domain': family, 'split': split,
                           'prompt': prompt, 'target': target, 'full_text': full,
                           'loss_mask_spans': mask, 'verification': verification}
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                    families[family] = families.get(family, 0) + 1
        report['counts'][split] = {'rows': count*len(FAMILIES), 'by_family': families, 'sha256': sha256(path)}
    (folder/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--seed', default=4807, type=int)
    args = ap.parse_args()
    print(json.dumps(generate(args.out, args.seed), ensure_ascii=False, indent=2))