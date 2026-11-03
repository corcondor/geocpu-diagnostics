"""Single-use private Kaggle entry with a process hard stop for raw evaluation.

The native v43 generator has no internal wall limit.  The child therefore runs
under this parent watchdog.  A killed child can leave flushed whole rows; the
parent seals only verified rows as PARTIAL, never COMPLETE or a score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

from eval_fixed_diagnostic import seal_interrupted_partial
from prelaunch_contract import verify_prelaunch_contract
from source_pools import file_sha


def launch(args) -> dict:
    if not Path("/kaggle").exists() or Path("/content/drive/MyDrive").exists():
        raise RuntimeError("Only a private disposable Kaggle GPU diagnostic is admitted")
    contract = verify_prelaunch_contract(args.prelaunch_contract,
        required_run_seconds=args.hard_stop_seconds,
        required_save_reserve_seconds=args.save_reserve_seconds)
    if args.checkpoint_kind == "parent" and contract["arm"] != "parent":
        raise ValueError("The parent diagnostic requires its own Root claim")
    if args.checkpoint_kind == "hybrid_250" and contract["arm"] not in ("control", "python"):
        raise ValueError("A hybrid diagnostic requires its own arm-specific Root claim")
    deadline = datetime.fromisoformat(contract["hard_stop_at_utc"].replace("Z", "+00:00")).astimezone(timezone.utc)
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    timeout = min(args.hard_stop_seconds - args.save_reserve_seconds,
                  remaining - args.save_reserve_seconds)
    if timeout < 600:
        raise ValueError("The new private evaluation claim leaves less than 600 seconds of work time")
    if args.output_dir.exists() != args.resume:
        raise ValueError("New output must be absent; resume output must already exist")
    command = [sys.executable, "-u", str(Path(__file__).with_name("eval_fixed_diagnostic.py")),
               "--v43-zip", str(args.v43_zip), "--overlay-source", str(args.overlay_source),
               "--checkpoint", str(args.checkpoint), "--checkpoint-kind", args.checkpoint_kind,
               "--output-dir", str(args.output_dir), "--prelaunch-contract", str(args.prelaunch_contract),
               "--hard-stop-seconds", str(args.hard_stop_seconds),
               "--save-reserve-seconds", str(args.save_reserve_seconds)]
    if args.resume:
        command.append("--resume")
    environment = dict(os.environ)
    environment["GEOCPU_EVAL_WATCHDOG_CHILD"] = "1"
    try:
        child = subprocess.run(command, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        # subprocess.run kills and waits for the child before it raises.  The
        # raw file can no longer be written while it is being sealed.
        args.output_dir.mkdir(parents=True, exist_ok=True)
        record = {"schema":"geocpu_hybrid_eval_watchdog_stop_v1",
                  "status":"HARD_STOP", "child_exited":True,
                  "checkpoint_sha256":file_sha(args.checkpoint),
                  "claim_sha256":file_sha(args.prelaunch_contract),
                  "evaluator_sha256":file_sha(Path(__file__).with_name("eval_fixed_diagnostic.py")),
                  "timeout_seconds":timeout, "training_updates":0,
                  "partial_sealed":False}
        try:
            partial = seal_interrupted_partial(output_dir=args.output_dir,
                checkpoint=args.checkpoint, arm=contract["arm"],
                overlay_source=args.overlay_source, v43_zip=args.v43_zip)
            record["partial_sealed"] = True
            record["partial_sha256"] = file_sha(args.output_dir / "PARTIAL.json")
            record["completed_generations"] = partial["completed_generations"]
        except Exception as seal_error:
            record["seal_error"] = type(seal_error).__name__ + ": " + str(seal_error)
        target = args.output_dir / "WATCHDOG_STOP.json"
        with target.open("x", encoding="utf-8") as sink:
            json.dump(record, sink, sort_keys=True, indent=2)
            sink.write("\n")
        return record
    if child.returncode != 0:
        raise RuntimeError("Diagnostic child exited with error; inspect its saved output. "
                           + child.stderr[-4000:])
    complete = args.output_dir / "COMPLETE.json"
    partial = args.output_dir / "PARTIAL.json"
    if complete.exists() == partial.exists():
        raise ValueError("The exited diagnostic child has neither or both terminal records")
    return {"status":"COMPLETE" if complete.exists() else "PARTIAL",
            "child_exited":True,
            "terminal_sha256":file_sha(complete if complete.exists() else partial),
            "stdout_tail":child.stdout[-1000:],
            "training_updates":0}


def main():
    parser = argparse.ArgumentParser()
    for name in ("v43_zip", "overlay_source", "checkpoint", "output_dir", "prelaunch_contract"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--checkpoint-kind", choices=("parent", "hybrid_250"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hard-stop-seconds", type=int, default=1800)
    parser.add_argument("--save-reserve-seconds", type=int, default=180)
    print(json.dumps(launch(parser.parse_args()), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
