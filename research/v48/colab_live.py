#!/usr/bin/env python3
"""Read-only Colab log follower or isolated command runner with durable live metrics.

Usage (for an existing Colab job):
 python -u colab_live.py --out /content/drive/MyDrive/GeoCPU_v48_logs --follow /content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106/v47_train_eval_suite_step_30500/suite_....log

For a NEW job only:
 python -u colab_live.py --out /content/drive/MyDrive/GeoCPU_v48_logs -- python -u /content/EXISTING_LAUNCHER.py
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

RE_PROGRESS = re.compile(r'(?i)\bProgress\s*:\s*(\d+)\s*/\s*(\d+)(?:\s*\|\s*Acc:\s*([0-9.]+)%)?')
RE_PYTHON = re.compile(r'PYTHON_TIR_STEP_(\d+)_PROGRESS\s+(\d+)/(\d+)')
RE_STAGE = re.compile(r'\b(preflight|training|tier_1_through_8|original_tier1_7|original_tier8|engineering_matplotlib_cad|python_120)\s+(START|EXIT|FINISH)\s*(\d+)?', re.I)
RE_TIER = re.compile(r'\b(?:Tier|TIER)\s*([1-8])\s*[:/]')
RE_STEP = re.compile(r'(?i)(?:\bStep\s+(\d+)\s*/\s*(\d+)|\bstep\s+(\d+)\s+loss\s*[:=]\s*([0-9.]+))')
RE_GRADED = re.compile(r'(?i)(?:Accuracy\s*:\s*(\d+)\s*/\s*(\d+)|Acc\s*:\s*[0-9.]+%\s*\((\d+)\s*/\s*(\d+)\))')
RE_COMPLETE = re.compile(r'PYTHON_TIR_STEP_(\d+)_COMPLETE\s+(\{.*\})')


def utc():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def extract(line: str) -> dict:
    d = {}
    if (m := RE_PROGRESS.search(line)):
        d['progress'] = {'done': int(m[1]), 'total': int(m[2])}
        if m[3] is not None:
            d['progress']['reported_accuracy_percent'] = float(m[3])
    if (m := RE_PYTHON.search(line)):
        d['python_progress'] = {'checkpoint_step': int(m[1]), 'done': int(m[2]), 'total': int(m[3])}
    if (m := RE_STAGE.search(line)):
        d['stage'] = {'name': m[1], 'state': m[2], 'exit_code': int(m[3]) if m[3] else None}
    if (m := RE_TIER.search(line)):
        d['tier'] = 'T' + m[1]
    if (m := RE_STEP.search(line)):
        d['training'] = {'step': int(m[1] or m[3])}
        if m[2]: d['training']['total'] = int(m[2])
        if m[4]: d['training']['loss'] = float(m[4])
    if (m := RE_GRADED.search(line)):
        a, b = (m[1], m[2]) if m[1] else (m[3], m[4])
        d['scored'] = {'correct': int(a), 'total': int(b), 'percent': round(100*int(a)/int(b), 3) if int(b) else None}
    if (m := RE_COMPLETE.search(line)):
        try:
            summary = json.loads(m[2])
            d['python_complete'] = {'checkpoint_step': int(m[1]), 'n': summary.get('n'),
                'all': summary.get('all', {})}
        except json.JSONDecodeError:
            d['parse_error'] = 'invalid_python_summary_json'
    if 'Traceback' in line or 'KeyError:' in line or 'RuntimeError:' in line:
        d['error'] = line.strip()[:500]
    return d


class StreamRecorder:
    def __init__(self, directory: Path, interval: float = 10):
        self.directory = directory
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f'nonempty output folder: {directory}; use a new run ID')
        directory.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self.state = {'started_utc': utc(), 'last_update_utc': None, 'lines': 0, 'stage_status': {},
                      'score_by_tier': {}, 'error_count': 0, 'status': 'running',
                      'rule': 'intermediate metrics; do not equate tool invocation with final accuracy'}
        self.events = (directory/'events.jsonl').open('x', encoding='utf-8', buffering=1)
        self.log = (directory/'console.log').open('x', encoding='utf-8', buffering=1)
        self.last_flush = 0.0
        self.persist(force=True)

    def ingest(self, line: str):
        parsed = extract(line)
        event = {'time_utc': utc(), 'line': line, **parsed}
        self.log.write(line+'\n')
        self.events.write(json.dumps(event, ensure_ascii=False) + '\n')
        self.state['lines'] += 1
        self.state['last_update_utc'] = utc()
        for name in ('progress', 'python_progress', 'python_complete', 'scored', 'training'):
            if name in parsed: self.state[name] = parsed[name]
        if 'stage' in parsed:
            self.state['stage_status'][parsed['stage']['name']] = parsed['stage']
        if 'tier' in parsed:
            self.state['current_tier'] = parsed['tier']
        if 'scored' in parsed and 'current_tier' in self.state:
            self.state['score_by_tier'][self.state['current_tier']] = parsed['scored']
        if 'error' in parsed:
            self.state['error_count'] += 1
            self.state['last_error'] = parsed['error']
        self.persist()

    def persist(self, force=False):
        if force or time.monotonic() - self.last_flush >= self.interval:
            temp = self.directory/'status.json.tmp'
            with temp.open('w', encoding='utf-8') as stream:
                json.dump(self.state, stream, ensure_ascii=False, indent=2)
                stream.write('\n')
            os.replace(temp, self.directory/'status.json')
            self.last_flush = time.monotonic()

    def close(self, code: int):
        self.state['status'] = 'complete' if code == 0 else ('stopped_by_user' if code == 130 else 'failed')
        self.state['returncode'] = code
        self.state['finished_utc'] = utc()
        self.persist(force=True)
        self.events.close(); self.log.close()


def follow(path, out, seconds):
    rec = StreamRecorder(out, seconds)
    exit_code = 0
    try:
        with path.open('r', encoding='utf-8', errors='replace') as src:
            while True:
                s = src.readline()
                if s:
                    print(s.rstrip(), flush=True)
                    rec.ingest(s.rstrip('\r\n'))
                else:
                    rec.persist()
                    time.sleep(min(seconds, 1))
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        exit_code = 1
        raise
    finally:
        rec.close(exit_code)


def execute(cmd, out, seconds):
    rec = StreamRecorder(out, seconds)
    q = queue.Queue()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            encoding='utf-8', errors='replace', text=True, bufsize=1)
    def reader():
        for s in proc.stdout: q.put(s.rstrip('\r\n'))
        q.put(None)
    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        while True:
            try: line = q.get(timeout=seconds)
            except queue.Empty:
                rec.persist(force=True)
                print(f'[heartbeat] {utc()} | {rec.state["lines"]} log lines', flush=True)
                continue
            if line is None: break
            print(line, flush=True)
            rec.ingest(line)
        code = proc.wait()
        proc.stdout.close()
        rec.close(code)
        return code
    except KeyboardInterrupt:
        proc.terminate()
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
        proc.stdout.close()
        rec.close(130)
        return 130


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True, help='new empty log folder on Drive or local fs')
    p.add_argument('--interval', type=float, default=10, help='heartbeat/flush seconds')
    p.add_argument('--follow', type=Path, help='follow an existing running log without starting a job')
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.interval <= 0: p.error('--interval must be positive')
    command = args.command
    if command[:1] == ['--']: command = command[1:]
    if bool(args.follow) == bool(command): p.error('choose exactly one of --follow or -- COMMAND')
    return follow(args.follow, args.out, args.interval) if args.follow else execute(command, args.out, args.interval)


if __name__ == '__main__':
    sys.exit(main())