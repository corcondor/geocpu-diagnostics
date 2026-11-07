#!/usr/bin/env python3
"""Strict, immutable-T1..T8 comparisons. Missing metrics stay missing, never 0%."""
import argparse
import json
from pathlib import Path

TIERS = tuple(f'T{i}' for i in range(1, 9))


def compare(base: dict, candidate: dict):
    if set(base['tiers']) != set(TIERS) or set(candidate['tiers']) != set(TIERS):
        raise ValueError('Both runs must declare T1 through T8 explicitly, even if unmeasured')
    output = []
    for tier in TIERS:
        a, b = base['tiers'][tier], candidate['tiers'][tier]
        names = set(a)|set(b)
        for name in sorted(names):
            x, y = a.get(name), b.get(name)
            if x is None or y is None:
                output.append({'tier': tier, 'metric': name, 'status': 'not_comparable:missing'})
                continue
            keys = ('correct', 'total', 'cohort_sha256', 'scorer_id', 'decoding_id')
            if any(k not in x or k not in y for k in keys):
                raise ValueError(f'incomplete scored metric {tier}/{name}')
            if any(x[k] != y[k] for k in ('total','cohort_sha256','scorer_id','decoding_id')):
                output.append({'tier': tier, 'metric': name, 'status': 'not_comparable:protocol_or_cohort_differs',
                               'base': x, 'candidate': y})
                continue
            if not 0 <= x['correct'] <= x['total'] or not 0 <= y['correct'] <= y['total']:
                raise ValueError(f'impossible score {tier}/{name}')
            n = x['total']
            output.append({'tier': tier, 'metric': name, 'status': 'comparable', 'base': x['correct'],
                           'candidate': y['correct'], 'total': n,
                           'delta_percentage_points': round((y['correct']-x['correct'])*100/n, 3) if n else None})
    return output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    b = json.loads(args.base.read_text(encoding='utf-8'))
    c = json.loads(args.candidate.read_text(encoding='utf-8'))
    result = {'base': b['run_id'], 'candidate': c['run_id'], 'comparison': compare(b, c)}
    if args.out.exists(): raise FileExistsError('refuse to overwrite a prior scorecard')
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()