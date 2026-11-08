#!/usr/bin/env python3
"""v48 one-shot evaluation: runs in the same Colab session right after training.

Order (cheap and decisive first, everything resumable, nothing is deleted or overwritten):
  0. CPU checks: no live training lock, both checkpoints and the original Tier stage pass their own
     pre-flight (`stage_official_tier_v47.py --preflight-only`) BEFORE any generation.
  1. v48 dev / unseen_eval (4 domains x 128 rows each) on step 61,500 and on the step 40,500 baseline.
  2. The unchanged original Tier 1-8 stage on step 61,500 (compared with the stored step 40,500 run).
  3. Reports: original-Tier comparison (compare_results) and the v48 before/after table.
  4. Optional (--aux): the v47 auxiliary suites through eval_candidate (hours).

Model-free parts (cases, scoring, orchestration order) are covered by test_eval_v48_oneshot.py.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DOMAINS = ("python", "logic", "evidence", "japanese")
SPLITS = ("dev", "unseen_eval")
MODE = "generate"
MAX_TOKENS = 512            # reference answers are < 300 characters; a longer budget only wastes GPU on rambling
EXPECTED_ROWS = 128
BASE_RUN, BASE_STEP = "practical_20261107_10k_multistep_40500", 40500
BASE_EVAL_DIR = "practical_20261107_10k_multistep_40500_eval_full_knowledge"
FINAL_STEP = 61500
CODE_ZIP = "GeoCPU_practical_training_logging_20261107/GeoCPU_practical_training_logging_20261107_10k.zip"
CODE_ZIP_SHA256 = "939e75d61aba5cd9f271b46e3d210d42af5a0c51904b80df101dbf40072d55d5"
FENCE = re.compile(r"(?ms)^```[^\n]*\n.*?^```[ \t]*(?:\n|$)")
ANSWER = re.compile(r"答え\s*[:：]\s*([^\n。]+)")


# ---------------------------------------------------------------- cases and scoring (model-free)
def suite_name(split: str, domain: str) -> str:
    return f"v48_{split}_{domain}"


ALL_SUITES = [suite_name(s, d) for s in SPLITS for d in DOMAINS]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_cases(split: str, domain: str, data_dir: Path = DATA) -> list[dict]:
    """The exact files the training run kept out of training (their SHA is in data/manifest.json)."""
    name = f"{domain}_{split}.jsonl"
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    info = manifest["files"][name]
    path = data_dir / name
    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"v48 evaluation file differs from its manifest: {name}")
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [r["id"] for r in rows]
    if (len(rows) != info["rows"] or len(rows) != EXPECTED_ROWS or len(set(ids)) != len(ids)
            or any(r.get("split") != split or r.get("training_category") != domain for r in rows)):
        raise ValueError(f"v48 evaluation rows are not the expected {EXPECTED_ROWS} unique {split}/{domain} rows")
    return rows


def final_answer(model_text: str):
    """Last '答え: X。' in the prose after the last closed code fence; never code or tool output."""
    closed = list(FENCE.finditer(model_text))
    prose = model_text[closed[-1].end():] if closed else model_text
    if re.search(r"(?m)^```", prose):
        return None                      # an unterminated fence cannot become an answer
    hits = ANSWER.findall(prose)
    return hits[-1].strip() if hits else None


def score_v48(case: dict, generation: dict) -> dict:
    got = final_answer(generation["model_generated_text"])
    expected = str(case["reference_final_answer"]).strip()
    exact = got == expected
    eos = bool(generation.get("ended_with_eos"))
    python_calls = int(generation.get("python_invocations", 0))
    return {"terminal_exact": exact and eos, "exact_final": exact, "format_valid": got is not None,
            "ended_with_eos": eos, "python_invocations": python_calls,
            "tool_grounded_exact": exact and eos and python_calls > 0,
            "final_answer": got, "expected_answer": expected}


def install_scoring(E) -> None:
    """Route the v48 suites through eval_candidate's own question loop, logger and raw-evidence binding."""
    E.SUPPLEMENTAL.update(ALL_SUITES)
    original = E.score_case

    def score_case(suite, case, generation, work):
        if suite in ALL_SUITES:
            return score_v48(case, generation), "terminal_exact", "scored"
        return original(suite, case, generation, work)

    E.score_case = score_case


def generation_settings() -> dict:
    return {"max_tokens": MAX_TOKENS, "enable_search": False, "enable_python": True,
            "loop_guard": False, "repetition_penalty": 1.0}


def run_sets(E, frozen, generator, output: Path, *, checkpoint_id, checkpoint_sha, runtime_sha, corpus_sha,
             data_dir: Path = DATA, limit=None, after_suite=None, only=None) -> dict:
    """Resumable: a suite that is already complete is verified and skipped by run_questions.

    `limit` exists only so CPU tests can use a few rows per suite; production always runs all 128."""
    install_scoring(E)
    settings = generation_settings()
    results = {}
    for split in SPLITS:
        for domain in DOMAINS:
            suite = suite_name(split, domain)
            if only is not None and suite not in only:
                continue
            cases = load_cases(split, domain, data_dir)
            if limit:
                cases = cases[:limit]
            identity = {"suite": suite, "mode": MODE, "checkpoint_id": checkpoint_id,
                        "checkpoint_sha256": checkpoint_sha, "dataset_sha256": E.hash_object(cases),
                        "scorer_sha256": E.hash_object({"wrapper": E.file_sha(Path(__file__)),
                                                         "frozen": frozen.SOURCE_FILES_SHA256}),
                        "runtime_sha256": runtime_sha,
                        "budget": dict(settings, retrieval_corpus_sha256=corpus_sha),
                        "seed": 0, "metric": "terminal_exact"}
            status = E.run_questions(suite, MODE, cases, generator, output / (E.safe_name(suite) + "__" + MODE),
                                     identity, resume=True)
            results[suite] = status
            if after_suite is not None:
                after_suite(suite)
            print(f"V48_SET {suite} {status.get('correct')}/{status.get('answered')} "
                  f"(state {status.get('state')})", flush=True)
    return results


def mirror_root(output: Path) -> Path:
    """Fixed local path per output folder: the logger stores raw-file paths, so a resumed run must see the same path."""
    base = Path("/content/_v48_eval_local") if Path("/content").is_dir() else Path(tempfile.gettempdir()) / "_v48_eval_local"
    output = Path(output)
    return base / (output.parent.name + "__" + output.name)


def run_sets_mirrored(E, frozen, generator, output: Path, **kw) -> dict:
    """The per-question durable log is fsync-heavy; on Drive it costs more than the generation itself.
    Log on the local disk, restore finished suites from Drive on resume, copy each suite to Drive when it completes."""
    import shutil
    output = Path(output)
    local = mirror_root(output)
    local.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    for done in sorted(output.glob("*__" + MODE)):
        if not (local / done.name).exists():
            shutil.copytree(done, local / done.name)

    def publish(suite):
        folder = E.safe_name(suite) + "__" + MODE
        shutil.copytree(local / folder, output / folder, dirs_exist_ok=True)

    return run_sets(E, frozen, generator, local, after_suite=publish, **kw)


def shard_suites(shards: int) -> list[list[str]]:
    """Split the 8 suites into 1, 2 or 4 workers by domain (each worker keeps a domain's dev and unseen_eval)."""
    groups = {1: [DOMAINS], 2: [DOMAINS[:2], DOMAINS[2:]], 4: [(d,) for d in DOMAINS]}
    if shards not in groups:
        raise ValueError("--shards must be 1, 2 or 4")
    return [[suite_name(s, d) for d in domains for s in SPLITS] for domains in groups[shards]]


def summarize_sets(before_dir: Path, after_dir: Path) -> dict:
    """Plain before/after table from each suite's durable status.json (no rounding of counts)."""
    def read(root, suite):
        p = Path(root) / (suite + "__" + MODE) / "status.json"
        if not p.is_file():
            return None
        r = json.loads(p.read_text(encoding="utf-8"))
        totals = r.get("metric_totals") or {}
        return {"correct": r.get("correct"), "answered": r.get("answered"), "total": r.get("total"),
                "accuracy": r.get("accuracy"), "state": r.get("state"),
                "detail": {k: totals.get(k) for k in ("exact_final", "format_valid", "ended_with_eos",
                                                      "tool_grounded_exact")}}
    rows = []
    for split in SPLITS:
        for domain in DOMAINS:
            suite = suite_name(split, domain)
            rows.append({"split": split, "domain": domain, "suite": suite,
                         "before": read(before_dir, suite), "after": read(after_dir, suite)})
    totals = {}
    for split in SPLITS:
        for side in ("before", "after"):
            parts = [r[side] for r in rows if r["split"] == split]
            complete = all(p and p["state"] == "complete" for p in parts)
            totals[f"{split}_{side}"] = ({"correct": sum(p["correct"] for p in parts),
                                         "answered": sum(p["answered"] for p in parts)} if complete else None)
    detail = {}
    for side in ("before", "after"):
        for key in ("exact_final", "format_valid", "ended_with_eos", "tool_grounded_exact"):
            parts = [r[side] for r in rows]
            ok = all(p and p["state"] == "complete" and (p.get("detail") or {}).get(key) is not None for p in parts)
            detail[f"{side}_{key}"] = sum(p["detail"][key] for p in parts) if ok else None
    return {"rows": rows, "totals": totals, "detail_totals": detail}


def sets_markdown(summary: dict) -> str:
    def cell(x):
        return "未評価" if not x or x.get("correct") is None else f"{x['correct']}/{x['answered']} ({100*x['correct']/x['answered']:.1f}%)"
    lines = ["# v48 評価セット（学習に入れていない問題）", "",
             "| 評価 | 領域 | 40,500 | 61,500 |", "|---|---|---:|---:|"]
    for r in summary["rows"]:
        lines.append(f"| {r['split']} | {r['domain']} | {cell(r['before'])} | {cell(r['after'])} |")
    for split in SPLITS:
        lines.append(f"| {split} | **合計** | {cell(summary['totals'][split+'_before'])} | "
                     f"{cell(summary['totals'][split+'_after'])} |")
    lines += ["", "正答は「答え: X。」の最終行が参照値と完全一致し、かつ EOS で終わったもの。", ""]
    d = summary.get("detail_totals") or {}
    if any(v is not None for v in d.values()):
        def n(v):
            return "未評価" if v is None else str(v)
        lines += ["## 内訳（全1,024問のうち）", "", "| 指標 | 40,500 | 61,500 |", "|---|---:|---:|"]
        for key, label in (("exact_final", "答えが参照値と一致（EOSを問わない）"),
                           ("format_valid", "「答え: X。」の形式で終わっている"),
                           ("ended_with_eos", "EOSで終了した"),
                           ("tool_grounded_exact", "Pythonを実行して正答（EOS終了）")):
            lines.append(f"| {label} | {n(d.get('before_'+key))} | {n(d.get('after_'+key))} |")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------- orchestration
def stamp_utc() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def find_final_run(study: Path, step: int = FINAL_STEP) -> str:
    """Newest v48 run folder that finished training at `step` (training_saved.json present)."""
    found = []
    for folder in sorted(Path(study).glob("v48_A100_from40500_*")):
        saved = folder / "training_saved.json"
        if saved.is_file() and json.loads(saved.read_text(encoding="utf-8")).get("step") == step:
            found.append(folder.name)
    if not found:
        raise FileNotFoundError(f"no completed v48 training run at step {step} under {study}")
    return found[-1]


def extract_code(zip_path: Path, sha: str, destination: Path) -> Path:
    if sha256_file(zip_path) != sha:
        raise RuntimeError("evaluation code ZIP SHA-256 differs from the pinned one")
    work = Path(tempfile.mkdtemp(prefix="geocpu_eval_v48_code_", dir=destination))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(work)
    return work


def bootstrap_modules(code_dir: Path):
    for name in ("live_logging", "frozen_eval", "train_candidate", "eval_candidate", "compare_results",
                 "train_from_40500"):
        sys.modules.pop(name, None)
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(code_dir))
    import eval_candidate as E
    import frozen_eval as frozen
    import train_candidate as OLD
    import train_from_40500 as T
    return E, frozen, OLD, T


def child_command(args, command: str, run: str, step: int, output: Path, only=None) -> list[str]:
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), command, "--study", str(args.study),
           "--code-dir", str(args.code_dir), "--run", run, "--step", str(step), "--output", str(output)]
    return cmd + (["--only", ",".join(only)] if only else [])


def oneshot(args, *, run_child, run_group, report_originals, E=None, frozen=None, OLD=None, T=None) -> dict:
    """The whole evaluation. `run_child(command, label)` -> returncode and `run_group([(label, command)])` ->
    {label: returncode} are injected so the order is testable on the CPU."""
    study = Path(args.study)
    if E is None:
        E, frozen, OLD, T = bootstrap_modules(Path(args.code_dir))
    T.refuse_active_work_v48(OLD, study)                       # never overwrites; ignores only dead launches
    run = find_final_run(study) if args.run == "auto" else args.run
    step = args.step
    # 0. fail on CPU before any generation
    new_ckpt, new_receipt = E.validate_checkpoint(study, run, step)
    base_ckpt, base_receipt = E.validate_checkpoint(study, BASE_RUN, BASE_STEP)
    root = study / f"v48_eval_{args.stamp}"
    after, before_sets, report_dir = root / f"step{step}", root / f"step{BASE_STEP}_v48sets", root / "reports"
    if root.exists() and not args.resume:
        raise FileExistsError(f"{root} exists; pass --resume to continue that evaluation")
    after.mkdir(parents=True, exist_ok=True)
    before_sets.mkdir(parents=True, exist_ok=True)
    state = {"run": run, "step": step, "stamp": args.stamp, "root": str(root),
             "checkpoint_sha256": new_receipt["checkpoint_sha256"],
             "baseline_checkpoint_sha256": base_receipt["checkpoint_sha256"], "stages": []}

    def save():
        (root / "oneshot_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def record(label, code):
        state["stages"].append({"stage": label, "returncode": code})
        save()

    tier_dir = after / "original_tiers"
    tier_done = (tier_dir / "complete.json").is_file()
    tier_command = frozen.build_official_tier_command(
        study=study, run=run, step=step, checkpoint_sha=new_receipt["checkpoint_sha256"], output=tier_dir)
    if not tier_done:
        if tier_dir.exists():
            raise RuntimeError(f"{tier_dir} exists but is not complete; inspect it (the original stage refuses overwrite)")
        probe = frozen.build_official_tier_command(
            study=study, run=run, step=step, checkpoint_sha=new_receipt["checkpoint_sha256"],
            output=after / "preflight_probe", preflight_only=True)
        code = run_child(probe, "original_stage_preflight")
        record("original_stage_preflight", code)
        if code:
            raise RuntimeError(f"stage original_stage_preflight stopped with return code {code}")
    # 1. one concurrent group: the original Tier stage and the v48 sets of both checkpoints.
    #    Batch-1 generation leaves the A100 mostly idle, so independent processes overlap almost for free.
    group = [] if tier_done else [("original_tier_1_8", tier_command)]
    for tag, (ck_run, ck_step, out) in (("new", (run, step, after)), ("base", (BASE_RUN, BASE_STEP, before_sets))):
        for index, suites in enumerate(shard_suites(args.shards)):
            group.append((f"v48_sets_{tag}_{index}", child_command(args, "sets", ck_run, ck_step, out, only=suites)))
    codes = run_group(group)
    for label, code in codes.items():
        record(label, code)
    summary = summarize_sets(before_sets, after)
    root.joinpath("v48_sets_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    root.joinpath("v48_sets_summary.md").write_text(sets_markdown(summary), encoding="utf-8")
    print(sets_markdown(summary), flush=True)
    failed = sorted(label for label, code in codes.items() if code)
    if failed:
        raise RuntimeError(f"stages stopped: {failed}; finished work is kept, rerun with --resume --stamp {args.stamp}")
    if not (tier_dir / "complete.json").is_file():
        raise RuntimeError("original Tier stage returned 0 but complete.json is missing")
    report_dir.mkdir(exist_ok=True)
    state["originals_report"] = report_originals(study / BASE_EVAL_DIR, after, report_dir / "original_tiers_vs_40500")
    if args.aux:
        code = run_child([sys.executable, "-u", str(Path(E.__file__)), "run", "--study", str(study),
                          "--run", run, "--step", str(step), "--output", str(after), "--resume", "--aux-only"]
                         + (["--full-knowledge"] if args.full_knowledge else []), "aux_suites")
        record("aux_suites", code)
        if code:
            raise RuntimeError(f"stage aux_suites stopped with return code {code}")
    state["finished"] = True
    save()
    return state


def default_report_originals(before, after, output):
    import compare_results
    try:
        payload = compare_results.write_reports(before, after, output)
    except Exception as error:                      # the comparison must never hide the measurements
        print(f"COMPARISON_NOT_WRITTEN {type(error).__name__}: {error}", flush=True)
        return {"written": False, "error": f"{type(error).__name__}: {error}"}
    wanted = {("T1", "final_answer_strict_match"), ("T1", "exact_step_termination"), ("T2", "end_to_end"),
              ("T3", "logit_argmax"), ("T3", "generative_exact"), ("T5", "logical_operator_accuracy"),
              ("T4", "SWE_Software"), ("T4", "JMMLU_Academic")}
    for row in payload["rows"]:
        if (row["suite"], row["metric"]) in wanted:
            f = lambda v: "—" if v is None else f"{100*v:.2f}%"
            print(f"COMPARE {row['suite']} {row['metric']}: {f(row['before'])} -> {f(row['after'])} "
                  f"(comparable={row['comparable']})", flush=True)
    print("RETENTION", json.dumps(payload["retention"], ensure_ascii=False), flush=True)
    return {"written": True, "output": str(output)}


def subprocess_runner(command, label):
    print(f"\n=== {label} ===", flush=True)
    return subprocess.run(command).returncode


def subprocess_group_runner(group, *, log_dir=None, every=40):
    """Run commands concurrently; stream each one prefixed with its label. Per-question logger lines are thinned
    (every `every`-th shown); everything is written to a per-stage log. Waits for all, never kills a sibling."""
    import threading
    log_dir = Path(log_dir) if log_dir else None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    procs, threads = {}, []

    def pump(label, proc):
        shown = 0
        log = (log_dir / (label + ".log")).open("a", encoding="utf-8") if log_dir else None
        try:
            for line in proc.stdout:
                if log:
                    log.write(line)
                    log.flush()
                if line.startswith("[20"):
                    shown += 1
                    if shown % every:
                        continue
                with lock:
                    print(f"[{label}] {line}", end="" if line.endswith("\n") else "\n", flush=True)
        finally:
            if log:
                log.close()

    for label, command in group:
        print(f"=== start {label} ===", flush=True)
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        procs[label] = proc
        thread = threading.Thread(target=pump, args=(label, proc), daemon=True)
        thread.start()
        threads.append(thread)
    codes = {}
    for label, proc in procs.items():
        codes[label] = proc.wait()
    for thread in threads:
        thread.join()
    return codes


def cmd_sets(args) -> None:
    """Load one saved checkpoint on the GPU and run the 8 v48 suites (subprocess: frees the GPU on exit)."""
    E, frozen, OLD, T = bootstrap_modules(Path(args.code_dir))
    checkpoint, receipt = E.validate_checkpoint(Path(args.study), args.run, args.step)
    with tempfile.TemporaryDirectory(prefix="geocpu_v48_eval_") as temp:
        source = E.prepare_source(Path(args.study), temp)
        model, vocab, inverse = E.load_model(source, checkpoint, receipt["checkpoint_sha256"])
        from local_search_engine_v42 import LocalSearchEngineV42
        corpus = frozen.source_path("public/retrieval/combined_dev_wikidata_first.jsonl")
        model.search_engine = LocalSearchEngineV42(corpus, geosearch_url="")
        settings = generation_settings()

        def generate(case):
            generation = model.generate_v44fix(case["prompt"], vocab, inverse, **settings)
            generation["prompt_oov_characters"] = sorted(set(case["prompt"]) - set(vocab))
            return generation

        runtime_sha = E.hash_object({p.name: E.file_sha(p) for p in sorted(source.glob("*.py"))})
        only = [x for x in args.only.split(",") if x] if args.only else None
        if only is not None and not set(only) <= set(ALL_SUITES):
            raise ValueError(f"unknown suites in --only: {sorted(set(only) - set(ALL_SUITES))}")
        run_sets_mirrored(E, frozen, generate, Path(args.output), only=only, checkpoint_id=f"{args.run}:{args.step}",
                 checkpoint_sha=receipt["checkpoint_sha256"], runtime_sha=runtime_sha,
                 corpus_sha=E.file_sha(corpus))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    one = sub.add_parser("oneshot")
    one.add_argument("--study", type=Path, required=True)
    one.add_argument("--run", default="auto")
    one.add_argument("--step", type=int, default=FINAL_STEP)
    one.add_argument("--code-zip", type=Path)
    one.add_argument("--code-zip-sha256", default=CODE_ZIP_SHA256)
    one.add_argument("--stamp", default=None)
    one.add_argument("--resume", action="store_true")
    one.add_argument("--aux", action="store_true", help="also run the v47 auxiliary suites (hours)")
    one.add_argument("--full-knowledge", action="store_true")
    one.add_argument("--shards", type=int, default=4, choices=(1, 2, 4),
                     help="v48-set workers per checkpoint (they run alongside the original Tier stage)")
    sets = sub.add_parser("sets")
    sets.add_argument("--only", default="", help="comma-separated suite names")
    for name in ("--study", "--code-dir", "--output"):
        sets.add_argument(name, type=Path, required=True)
    sets.add_argument("--run", required=True)
    sets.add_argument("--step", type=int, required=True)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    if args.command == "sets":
        cmd_sets(args)
        return
    study = Path(args.study)
    args.stamp = args.stamp or stamp_utc()
    zip_path = args.code_zip or (study / CODE_ZIP)
    args.code_dir = extract_code(zip_path, args.code_zip_sha256, Path(tempfile.gettempdir()))
    log_dir = study / f"v48_eval_{args.stamp}" / "logs"
    state = oneshot(args, run_child=subprocess_runner,
                    run_group=lambda group: subprocess_group_runner(group, log_dir=log_dir),
                    report_originals=default_report_originals)
    print("V48_EVAL_DONE", json.dumps({k: state[k] for k in ("run", "step", "root")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
