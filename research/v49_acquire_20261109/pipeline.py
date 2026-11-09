#!/usr/bin/env python3
"""pipeline.py - the steps of colab_acquire_check.ipynb as plain functions, so that they can be unit-tested locally on tiny fixtures.

Runs on a Google Colab CPU runtime (12 GB RAM, no GPU).  Glue only: the work is done by acquire.py (download + verify),
checks.py (OOV, length, duplicates, decontamination, sandbox) and convert.py (one converter per dataset).  Standard library only
(+ pyarrow / numpy that the Colab image has).  Nothing here needs a secret, a GPU or a connection other than the one acquire.py opens
to huggingface.co.

Every step function prints one line   PASS [step] detail   or   FAIL [step] reason   and raises StepFailed (an AssertionError) on FAIL,
so a notebook cell that calls it stops the run; the cells additionally `assert R["pass"]`.

Writing rules (the owner's): only NEW folders on Drive; a file that exists with a different sha256 is never overwritten (the step FAILs
and says to set RUN_SUFFIX so that a fresh output folder is used); nothing is ever deleted on Drive.  Everything heavy happens on the
Colab local disk (cfg.work, default /content/v49_work), which is scratch.  Every write to Drive goes to `<name>.part` first and is renamed
into place only when the target still does not exist; cfg.drive_sync (the notebook: flush_and_unmount + mount) runs after steps 5, 7b, 8
and 9, and step 9 re-reads every file it lists from the fresh mount.

Flow (the numbers are the notebook's cells):
  4 dry run -> 5 acquire -> 6a local mirror of the train files -> 6b vocabulary + eval index -> 6c raw checks (counts vs the cards,
  OOV, length, 100-row reading dumps) -> 7-0 sandbox self-test (Drive still mounted: proves the OS barrier) -> 7a converters on the first
  2,000 records -> 7b converters on everything (in waves; Drive is unmounted while untrusted code runs; coding and tengentoppa run in
  shards that are published one by one, so a disconnect loses at most one shard) -> 8 filters (exact + MinHash near duplicates,
  13-gram decontamination, old corpus, near siblings of train inside dev / heldout) -> 8b optional reviews with a time budget ->
  9 final rows, registry.json, REPORT_JA.md.
"""
from __future__ import annotations

import bisect
import collections
import contextlib
import dataclasses
import datetime
import functools
import hashlib
import inspect
import io
import itertools
import json
import math
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import acquire
import checks
import convert

PIPELINE_VERSION = "2"
OUT_NAME = "v49_acquire_20261109"
ROWS_NAME = "external_data_v49_rows"
STAGE_A_NAME = "external_data_v49_stageA"            # unfiltered converted rows; NOT tied to RUN_SUFFIX (a filter-only re-run must not reconvert)
HOLD_DIRNAME = "_HOLD_do_not_mix"                    # rows of datasets whose adoption is on hold live here, never next to the candidates
HOLD_KEYS = ("toolace", "wizard")
GROUP_INDEPENDENT = ("jaquad", "hotpot")             # datasets whose dev / heldout are split by a natural group
USER_EVAL_EXTRA = "v49_eval_extra"
SPLITS = ("train", "dev", "heldout")
KEYS = ("coding", "tengentoppa", "jaquad", "hotpot", "toolace", "wizard")                    # convert.py keys, report order
CONVERT_ORDER = ("coding", "tengentoppa", "hotpot", "wizard", "jaquad", "toolace")           # slowest first
ACQ_KEY = {"coding": "coding", "tengentoppa": "tengentoppa", "jaquad": "jaquad", "hotpot": "hotpotqa", "toolace": "toolace",
           "wizard": "wizardlm"}
EXPECTED_COUNTS = {"coding": 113796, "tengentoppa": 125000, "jaquad": 31748, "hotpot": 90447, "toolace": 11300, "wizard": 116422}
COUNT_SOURCE = {"coding": "dataset card: 113,796", "tengentoppa": "dataset card: 125,000", "jaquad": "owner's plan: train 31,748 (card: 39,696 in total)",
                "hotpot": "dataset card: distractor train 90,447", "toolace": "task text: 11,300", "wizard": "dataset card: 116,422"}
KNOWN_VOCAB_SHA = "8329446036ff708445cf08983267c2c599dae54b23efeb92be86129b153e8288"        # sha256 of the sorted vocabulary characters
V48_EVAL_NAMES = tuple(f"{fam}_unseen_eval.jsonl" for fam in ("evidence", "japanese", "logic", "python"))
R_EXACT, R_NEAR, R_EVAL, R_OLD_EXACT, R_OLD_NEAR, R_SPLIT_LEAK = 1, 2, 3, 4, 5, 6
REASON = {R_EXACT: "exact_duplicate_within_dataset", R_NEAR: "near_duplicate_within_dataset", R_EVAL: "eval_overlap_13gram",
          R_OLD_EXACT: "exact_duplicate_of_old_corpus", R_OLD_NEAR: "near_duplicate_of_old_corpus",
          R_SPLIT_LEAK: "dev_heldout_row_with_a_near_sibling_in_train"}
STAGE_A_FILES = ("train.jsonl", "dev.jsonl", "heldout.jsonl", "manifest.json", "sample_dump.txt", "rejects_sample.jsonl")
SHARD_FILES = ("train.jsonl", "dev.jsonl", "heldout.jsonl", "manifest.json", "rejects_sample.jsonl")
FUNNEL = ("converted", "after_exact_dedup", "after_near_dedup", "after_eval_decontam", "after_old_corpus_dedup", "after_split_hygiene")
BASE_NOT_COVERED = [
    "GSM8K / MATH test sets are not on Drive and were not downloaded (the owner approved exactly six datasets): "
    "they are NOT part of the decontamination.  Put them as .jsonl/.json under <Drive root>/" + USER_EVAL_EXTRA + "/ and re-run "
    "step 6b onwards (stage A is kept: a new RUN_SUFFIX only redoes the filters).",
    "Semantic (paraphrase) overlap is only looked at, not removed: the character-bigram look-alike search is lexical and not exhaustive.",
    "HotpotQA validation is indexed by its questions only (its passages are Wikipedia paragraphs shared with train by construction).",
    "Upstream model terms (Nemotron-4, Phi-3, Mixtral, calm3, Qwen2.5, WizardLM-2) were NOT checked: only the dataset licences were.",
    "Near-duplicate removal is a MinHash estimate: it reliably removes pairs of Jaccard >= 0.85, usually those near 0.8, and misses looser "
    "paraphrases (see the cut and the detection table in the report).",
]


class StepFailed(AssertionError):
    pass


def require(cond, msg: str) -> None:
    if not cond:
        raise StepFailed(msg)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


def not_covered(missing_optional: Optional[List[str]] = None) -> List[str]:
    out = list(BASE_NOT_COVERED)
    if missing_optional:
        out.append("These frozen eval suites of the owner's registry were NOT found on Drive and are NOT part of the decontamination: "
                   + "; ".join(missing_optional))
    return out


# ====================================================================================================== configuration
@dataclasses.dataclass
class Cfg:
    root: str = acquire.DEFAULT_ROOT
    suffix: str = ""                                    # RUN_SUFFIX: a fresh OUTPUT folder after a refusal (never overwrite); stage A is kept
    raw_suffix: str = ""                                # RAW_SUFFIX: a fresh folder for the downloaded files (only after a refused raw file)
    work: str = "/content/v49_work"                     # local scratch (NOT Drive)
    only: Optional[tuple] = None                        # convert.py keys; None = all six
    smoke_limit: int = 2000
    stop_on_count_mismatch: bool = False
    expected_counts: Optional[dict] = dataclasses.field(default_factory=lambda: dict(EXPECTED_COUNTS))
    expected_old_rows: Optional[int] = checks.OLD_CORPUS_ROWS
    old_corpus: Optional[str] = None
    max_chars: int = convert.DEFAULT_CAP
    near_threshold: float = 0.8                         # the Jaccard we want to remove ...
    near_margin: float = 0.05                           # ... the MinHash estimate is cut this much lower (see checks.MinHasher)
    split_hygiene_threshold: float = 0.5                # a dev / heldout row whose estimated Jaccard with a train row is >= this is dropped
    run_split_hygiene: bool = True
    ngram: int = 13
    boilerplate_eval_df: int = 20                       # a shingle shared by this many EVAL items is template boilerplate (never the train-side count)
    min_shared: int = 1
    wizard_max_error_rate: float = 0.01                 # shown in the report next to the measured rate; it no longer decides anything
    wizard_min_statements: int = 300
    run_sandbox: bool = True
    require_os_containment: Optional[bool] = None       # None = yes on Colab (the child must run as an unprivileged user), no elsewhere
    python_allow_unverified: bool = False               # True also keeps python rows without a stored output (default: dropped and counted)
    parallel: int = 2
    shard_keys: tuple = ("coding", "tengentoppa")       # converted in shards of shard_records records, each published when it is done
    shard_records: int = 20000
    max_unsharded_minutes: float = 120.0                # 7b refuses to start an unsharded dataset estimated above this
    poll_s: float = 15.0
    heartbeat_s: float = 120.0
    vocab_override: Optional[str] = None                # tests only: a base_metadata.json on the local disk
    allow_vocab_mismatch: bool = False                  # True: a vocabulary whose character set differs from the one on the owner's PC is accepted
    eval_extra_dir: Optional[str] = None                # the notebook writes the embedded v48 unseen_eval texts here
    require_v48: bool = True
    run_semantic: bool = True
    semantic_max_queries: int = 500
    semantic_min_cos: float = 0.5
    semantic_min_shared_blocks: int = 5
    semantic_max_block_df: int = 5
    semantic_budget_s: float = 900.0
    run_cross: bool = True
    cross_budget_s: float = 900.0
    check_disk: bool = True
    allow_local: bool = False                           # tests only (acquire refuses real downloads off Colab)
    stage_dir: Optional[str] = None
    drive_detach: Optional[Callable] = None             # the notebook: flush_and_unmount -> Drive is not visible while untrusted code runs
    drive_attach: Optional[Callable] = None             # the notebook: mount again
    drive_sync: Optional[Callable] = None               # the notebook: flush_and_unmount + mount (writes really reach Drive)
    log: Callable = print
    progress: Optional[Callable] = None
    state: dict = dataclasses.field(default_factory=dict)
    timings: list = dataclasses.field(default_factory=list)

    def __post_init__(self):
        self.root = str(self.root)
        if self.only is not None:
            bad = [k for k in self.only if k not in KEYS]
            if bad:
                raise ValueError(f"unknown dataset key(s) {bad}; known: {list(KEYS)}")
            self.only = tuple(self.only)
        if (self.drive_detach is None) != (self.drive_attach is None):
            raise ValueError("drive_detach and drive_attach go together")

    # --- which datasets
    @property
    def keys(self) -> tuple:
        return KEYS if self.only is None else tuple(k for k in KEYS if k in self.only)

    def acq_only(self):
        return None if self.only is None else [ACQ_KEY[k] for k in self.keys]

    @property
    def near_cut(self) -> float:
        return round(self.near_threshold - self.near_margin, 6)

    @property
    def containment_required(self) -> bool:
        return acquire._in_colab() if self.require_os_containment is None else bool(self.require_os_containment)

    # --- places
    @property
    def ext_dirname(self) -> str:
        return acquire.EXTERNAL_DIRNAME + self.raw_suffix

    @property
    def ext(self) -> Path:
        return Path(self.root) / self.ext_dirname

    @property
    def out(self) -> Path:
        return Path(self.root) / (OUT_NAME + self.suffix)

    @property
    def rows(self) -> Path:
        return Path(self.root) / (ROWS_NAME + self.suffix)

    @property
    def old_corpus_path(self) -> Path:
        return Path(self.old_corpus) if self.old_corpus else Path(self.root) / checks.OLD_CORPUS_REL

    @property
    def mirror(self) -> Path:
        return Path(self.work) / "mirror"

    @property
    def vocab_file(self) -> Path:
        return Path(self.work) / "vocab_chars.json"

    def code_hashes(self) -> dict:
        return {Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                for m in (acquire, checks, convert, sys.modules[__name__])}

    def code_tag(self) -> str:
        h = self.code_hashes()
        return sha256_text("".join(h[k] for k in sorted(h)))[:8]

    def converter_hashes(self) -> dict:
        """Hashes of everything that decides what a converted row looks like: convert.py, the sandbox (source of the static rules and of the
        child, how it is launched, the executor profile), the runtime symbol normaliser and the vocabulary.  pipeline.py, acquire.py and the
        filter code of checks.py are NOT in it: changing them must not throw the (hours-long) converted rows away."""
        def norm(t: str) -> str:
            return t.replace("\r\n", "\n")
        sb = hashlib.sha256()
        for part in (checks._VALIDATE_SRC, checks._CHILD_SRC, repr(sorted(checks.SANDBOX_PROFILES.items())), inspect.getsource(checks._spawn),
                     inspect.getsource(checks.code_sandbox_run), inspect.getsource(checks._resolve_run_as), inspect.getsource(checks.static_check),
                     checks.SANDBOX_BACKEND, repr(sorted(checks.RUNTIME_NORMALIZE.items()))):
            sb.update(norm(part).encode("utf-8"))
            sb.update(b"\0")
        vocab = "none"
        if self.vocab_file.is_file():
            try:
                vocab = hashlib.sha256("".join(sorted(json.loads(self.vocab_file.read_text(encoding="utf-8"))["vocab"])).encode("utf-8")).hexdigest()
            except (ValueError, KeyError, OSError):
                vocab = "unreadable"
        return {"convert.py": hashlib.sha256(Path(convert.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                "checks.sandbox_and_normaliser": sb.hexdigest(), "vocabulary_chars": vocab}

    def converter_tag(self) -> str:
        h = self.converter_hashes()
        return sha256_text("".join(h[k] for k in sorted(h)))[:10]

    def conv_dir(self, key: str, smoke: bool = False) -> Path:
        return Path(self.work) / (("smoke_" if smoke else "converted_") + self.converter_tag()) / key

    def shard_name(self, a: int, b: Optional[int]) -> str:
        return f"{a:09d}_{'end' if b is None else format(b, '09d')}"

    def shard_root(self, key: str, a: int, b: Optional[int]) -> Path:
        """--out-dir of one shard job (the rows are in <shard_root>/<key>/)."""
        return Path(self.work) / ("converted_" + self.converter_tag()) / "_shards" / f"{key}_{self.shard_name(a, b)}"

    def stage_a_drive(self, key: str) -> Path:
        return Path(self.root) / STAGE_A_NAME / self.converter_tag() / key

    def shard_drive(self, key: str, a: int, b: Optional[int]) -> Path:
        return Path(self.root) / STAGE_A_NAME / self.converter_tag() / f"{key}__shards" / self.shard_name(a, b)

    def mirror_dir(self, key: str) -> Path:
        return self.mirror / acquire.folder_of(_ds(key)["repo"])


def make_cfg(**kw) -> Cfg:
    """Cfg with a progress printer that writes to stdout (visible in the notebook)."""
    kw.setdefault("progress", checks.make_printer(30.0, stream=sys.stdout))
    return Cfg(**kw)


def _ds(key: str) -> dict:
    akey = ACQ_KEY[key]
    return next(d for d in acquire.DATASETS if d["key"] == akey)


# ====================================================================================================== step plumbing
def _flush_after_failure(cfg, name: str) -> None:
    """A step that writes to Drive and then fails still leaves what it wrote; make sure it really reaches Drive (never hides the failure)."""
    if cfg.drive_sync is None:
        return
    try:
        cfg.log(f"  flushing Google Drive writes after the failure of [{name}] ...")
        cfg.drive_sync()
    except Exception as e:                                                          # noqa: BLE001
        cfg.log(f"  WARNING: could not flush Drive after the failure ({type(e).__name__}: {e})")


def step(name: str, flush_on_failure: bool = False):
    """Decorator: prints the header and one PASS/FAIL line; fn(cfg, ...) returns (detail, result dict).  flush_on_failure: the steps that write
    to Drive flush it (cfg.drive_sync) also when they fail."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(cfg, *a, **kw):
            t0 = time.time()
            cfg.log(f"=== {name} ===")
            try:
                detail, res = fn(cfg, *a, **kw)
            except StepFailed as e:
                cfg.log(f"FAIL [{name}] {e}")
                if flush_on_failure:
                    _flush_after_failure(cfg, name)
                raise
            except Exception as e:                                                  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()[-6:]
                cfg.log(f"FAIL [{name}] {type(e).__name__}: {e}\n    " + "\n    ".join(tb))
                if flush_on_failure:
                    _flush_after_failure(cfg, name)
                raise StepFailed(f"{name}: {type(e).__name__}: {e}") from e
            out = dict(res or {})
            out.update({"pass": True, "step": name, "detail": detail, "seconds": round(time.time() - t0, 1)})
            cfg.timings.append({"step": name, "seconds": out["seconds"]})
            cfg.log(f"PASS [{name}] {detail}  ({out['seconds']} s)")
            return out
        wrapper.step_name = name
        return wrapper
    return deco


def write_once(path, text: str) -> str:
    """Create a text file on Drive (via <path>.part); identical content is fine, different content is refused."""
    try:
        return checks.write_new_file(str(path), text)
    except FileExistsError as e:
        raise StepFailed(f"{e}.  Nothing was overwritten.  To produce new output, set RUN_SUFFIX (e.g. '_2') in the notebook and run again.") from e


def publish_file(src, dst) -> str:
    """Copy a local file to Drive: never over an existing file (re-checked right before the rename); verified by sha256; a partial copy of an
    earlier run is continued when its bytes are exactly the start of the source (otherwise moved aside, never deleted)."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = checks.sha256_file(str(src))
    if os.path.lexists(dst):
        if checks.sha256_file(str(dst)) == want:
            return "identical"
        raise StepFailed(f"refusing to overwrite {dst}: it exists with a different sha256.  To produce new output, set RUN_SUFFIX (e.g. '_2') "
                         f"and run again.")
    part = Path(str(dst) + ".part")
    n, sha, _resumed = acquire._copy_resume(str(src), str(part))
    require(sha == want and n == src.stat().st_size, f"copy of {src.name} to Drive does not match the source (kept as {part})")
    if os.path.lexists(dst):
        raise StepFailed(f"{dst} appeared while it was being copied; it was NOT overwritten (the finished copy stays at {part})")
    os.rename(part, dst)
    return "created"


def drive_sync(cfg: Cfg, why: str) -> None:
    """Make Drive writes really durable: the notebook's hook flushes and remounts (Drive/FUSE writes are asynchronous)."""
    if cfg.drive_sync is not None:
        cfg.log(f"  flushing Google Drive writes ({why}) ...")
        cfg.drive_sync()


@contextlib.contextmanager
def untrusted_code_window(cfg: Cfg):
    """Drive is NOT mounted while code from the datasets runs: the owner's whole Drive (model weights, optimiser, the old corpus) is not
    reachable from the VM during that time, whatever the sandbox does.  The converters only touch cfg.work."""
    if cfg.drive_detach is None:
        yield False
        return
    cfg.log("  Google Drive is unmounted while code from the datasets runs ...")
    cfg.drive_detach()
    try:
        yield True
    finally:
        cfg.drive_attach()
        cfg.log("  Google Drive is mounted again.")


def save_state(cfg: Cfg, name: str, obj) -> None:
    cfg.state[name] = obj
    write_once(cfg.out / "state" / f"{name}.json", jdump(obj))


def load_state(cfg: Cfg, name: str):
    if name in cfg.state:
        return cfg.state[name]
    p = cfg.out / "state" / f"{name}.json"
    if p.is_file():
        cfg.state[name] = json.loads(p.read_text(encoding="utf-8"))
        return cfg.state[name]
    return None


_CACHE: Dict[str, Any] = {}


# ====================================================================================================== 3: environment self-check
@step("3 modules")
def step_modules(cfg: Cfg):
    require(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} is too old (need 3.10+)")
    acquire.validate_table(acquire.DATASETS)
    require(set(KEYS) == set(convert.DATASETS), "pipeline keys differ from convert.DATASETS")
    require(checks.RUNTIME_NORMALIZE == convert.NORMALIZE, "checks.py and convert.py disagree on the runtime symbol normaliser")
    require(checks.SANDBOX_BACKEND == "checks.code_sandbox_check(profile=executor)", "unexpected sandbox backend name")
    h = cfg.code_hashes()
    return "modules import; table valid; " + ", ".join(f"{k} {v[:8]}" for k, v in sorted(h.items())), {"code_sha256": h}


# ====================================================================================================== 4: dry run
def stale_files(cfg: Cfg) -> List[Path]:
    """Leftover *.stale-N copies of earlier dead runs (moved aside by the never-delete rule); the owner can remove them by hand."""
    out: List[Path] = []
    for d in (cfg.ext, cfg.out, cfg.rows, Path(cfg.root) / STAGE_A_NAME):
        if d.is_dir():
            out += [p for p in d.rglob("*") if p.is_file() and ".stale-" in p.name]
    return sorted(out)


@step("4 dry run")
def step_dry_run(cfg: Cfg):
    plan = acquire.acquire(cfg.root, cfg.acq_only(), dry_run=True, log=cfg.log, ext_dirname=cfg.ext_dirname)
    sel = acquire.select(list(acquire.DATASETS), cfg.acq_only())
    expect = acquire.total_bytes(sel)
    require(plan["total_bytes"] == expect and plan["n_files"] == sum(len(d["files"]) for d in sel),
            f"plan says {plan['total_bytes']:,} bytes / {plan['n_files']} files, the table says {expect:,}")
    biggest = max(f["size"] for d in sel for f in d["files"])
    train = acquire.total_bytes(sel, ("train",))
    notes = []
    need_drive = expect + 2 * train + (1 << 30)               # raw files + stage A + final rows + dumps/state (7a measures the real size)
    if cfg.check_disk:
        Path(cfg.work).mkdir(parents=True, exist_ok=True)
        need_local = biggest + train + (2 << 30)
        free_local = shutil.disk_usage(cfg.work).free
        require(free_local >= need_local, f"local disk has {free_local:,} B free, need about {need_local:,} B (staging + mirror + rows)")
        notes.append(f"local free {free_local / 2**30:.1f} GiB")
        try:
            free_drive = shutil.disk_usage(cfg.root).free
        except OSError:
            free_drive = None
        if free_drive:
            require(free_drive >= need_drive, f"Drive reports {free_drive:,} B free, need about {need_drive:,} B (raw files + converted rows + final rows)")
            notes.append(f"Drive free {free_drive / 2**30:.1f} GiB (need about {need_drive / 2**30:.1f})")
        else:
            notes.append("Drive free space unknown")
    stale = stale_files(cfg)
    if stale:
        total = sum(p.stat().st_size for p in stale)
        cfg.log(f"  note: {len(stale)} leftover '.stale-N' file(s) from earlier interrupted runs take {total / 2**20:,.0f} MiB on Drive; "
                f"they are never deleted by this notebook - remove them by hand if you need the space: " + ", ".join(str(p) for p in stale[:5]))
        notes.append(f"{len(stale)} stale files ({total / 2**20:,.0f} MiB)")
    return (f"{plan['n_files']} files, expected total {expect:,} bytes = {expect / 1e9:.2f} GB "
            f"(train {train:,}; dev/validation {plan['do_not_train_bytes']:,})" + "".join("; " + x for x in notes)), \
        {"expected_total_bytes": expect, "n_files": plan["n_files"], "need_drive_bytes": need_drive, "stale_files": len(stale)}


# ====================================================================================================== 5: acquire
def verify_acquired(cfg: Cfg) -> dict:
    """After acquire.acquire(): every selected dataset has provenance, the licence the plan expects, and all its files on Drive."""
    man = acquire.load_manifest(cfg.root, ext_dirname=cfg.ext_dirname)
    out = {}
    for key in cfg.keys:
        ds = _ds(key)
        prov = man.get(ds["repo"])
        require(prov is not None, f"{ds['repo']}: provenance.json is missing - run step 5 first")
        require(set(prov["license"]["ids_from_card"]) == set(ds["licenses"]), f"{ds['repo']}: licence on the card {prov['license']['ids_from_card']}")
        files = []
        for f in prov["files"]:
            p = cfg.ext / f["dest_rel"]
            require(p.is_file() and p.stat().st_size == f["size"], f"{f['dest_rel']}: missing or wrong size on Drive")
            files.append({"dest_rel": f["dest_rel"], "role": f["role"], "size": f["size"], "sha256": f["sha256"]})
        require(acquire.train_files(cfg.root, ds["key"], ext_dirname=cfg.ext_dirname), f"{ds['repo']}: no train files")
        out[key] = {"repo": ds["repo"], "commit_sha": prov["commit_sha"], "retrieved_utc": prov["retrieved_utc"],
                    "license_name": prov["license"]["name"], "license_ids": prov["license"]["ids_from_card"],
                    "license_url": prov["license"]["url"], "share_alike": prov["license"]["share_alike"],
                    "license_lines": prov["license"].get("card_licence_lines") or [],
                    "attribution": prov["attribution"]["lines"], "row_defaults": prov["row_defaults"],
                    "adoption_at_acquire": prov["adoption"], "language": prov["language"], "rows_per_card": prov["rows_per_card"],
                    "generator_notes": prov["generator_notes"], "quality_notes": prov["quality_notes"],
                    "files": files, "train_bytes": sum(f["size"] for f in files if f["role"] == "train"),
                    "do_not_train_bytes": sum(f["size"] for f in files if f["role"] in acquire.NO_TRAIN_ROLES)}
    return out


@step("5 acquire", flush_on_failure=True)
def step_acquire(cfg: Cfg):
    res = acquire.acquire(cfg.root, cfg.acq_only(), log=cfg.log, allow_local=cfg.allow_local, stage_dir=cfg.stage_dir, ext_dirname=cfg.ext_dirname)
    drive_sync(cfg, "step 5")
    info = verify_acquired(cfg)
    save_state(cfg, "step5_acquire", info)
    return (f"{len(info)} datasets, {res['total_bytes']:,} bytes verified against the Hub (sha256 / git blob id), "
            f"{res['downloaded_bytes']:,} downloaded in this run; licences match the plan"), {"datasets": list(info)}


# ====================================================================================================== 6a: local mirror
def mirror_train_files(cfg: Cfg, key: str) -> List[Path]:
    """Train files of one dataset inside the local mirror, named by the provenance record (dev/validation can never be listed)."""
    d = cfg.mirror_dir(key)
    prov = json.loads((d / "provenance.json").read_text(encoding="utf-8"))
    ext_folder = cfg.mirror
    return [ext_folder / f["dest_rel"] for f in prov["files"] if f["role"] == "train" and f.get("may_train")]


@step("6a local mirror")
def step_mirror(cfg: Cfg):
    man = acquire.load_manifest(cfg.root, ext_dirname=cfg.ext_dirname)
    total = copied = 0
    for key in cfg.keys:
        ds = _ds(key)
        prov = man.get(ds["repo"])
        require(prov is not None, f"{ds['repo']}: not acquired")
        exp = {f["dest_rel"]: f for f in prov["files"]}
        folder = acquire.folder_of(ds["repo"])
        (cfg.mirror / folder).mkdir(parents=True, exist_ok=True)
        for small in ("provenance.json", "README.md"):
            if (cfg.ext / folder / small).is_file():
                shutil.copyfile(cfg.ext / folder / small, cfg.mirror / folder / small)
        for src in acquire.train_files(cfg.root, ds["key"], ext_dirname=cfg.ext_dirname):
            rel = src.relative_to(cfg.ext).as_posix()
            e = exp[rel]
            dst = cfg.mirror / rel
            total += e["size"]
            if dst.is_file() and dst.stat().st_size == e["size"] and checks.sha256_file(str(dst)) == e["sha256"]:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            part = Path(str(dst) + f".part{os.getpid()}")
            if part.exists():                                                        # local scratch copy of our own: safe to replace
                part.unlink()
            n, sha = acquire._copy_hashing(str(src), str(part))
            require(n == e["size"] and sha == e["sha256"], f"{rel}: the local copy differs from the verified file on Drive")
            os.replace(part, dst)
            copied += n
    return f"{total:,} bytes of train files are on the local disk and match the recorded sha256 ({copied:,} copied now)", {"bytes": total}


# ====================================================================================================== 6b: vocabulary + eval index
def _controller_source(cfg: Cfg) -> Optional[str]:
    name = "qwenputer_controller_v30.py"
    try:
        for zp in checks._bundle_zip_paths(cfg.root):
            with zipfile.ZipFile(zp) as z:
                names = z.namelist()
                for n in names:
                    if n.endswith(name):
                        return z.read(n).decode("utf-8")
                if checks.VOCAB_NESTED_ZIP in names:
                    with zipfile.ZipFile(io.BytesIO(z.read(checks.VOCAB_NESTED_ZIP))) as inner:
                        for n in inner.namelist():
                            if n.endswith(name):
                                return inner.read(n).decode("utf-8")
    except (OSError, zipfile.BadZipFile, UnicodeDecodeError):
        return None
    return None


def _jaquad_dev_texts(paths):
    seen = set()
    for p in paths:
        for r in checks.iter_rows(str(p)):
            c = r.get("context")
            if isinstance(c, str) and c not in seen:
                seen.add(c)
                yield c
            q = r.get("question")
            if isinstance(q, str):
                yield q


def _hotpot_validation_texts(paths):
    for p in paths:
        for r in checks.iter_rows(str(p), columns=["question"]):
            q = r.get("question")
            if isinstance(q, str):
                yield q


def decontam_sources(cfg: Cfg, ef) -> list:
    """Everything the training rows must not overlap: the eval manifests/code found by checks.find_eval_files, the v48 unseen_eval
    texts, the registry's other frozen suites (python120, engineering64, visual10, the new_public dev sets) where they exist on Drive, the
    dev/validation files that acquire.py keeps under do_not_train_*, and anything the owner drops into v49_eval_extra/."""
    sources = list(ef.sources)
    dnt = acquire.do_not_train_files(cfg.root, ext_dirname=cfg.ext_dirname)
    for repo, paths in sorted(dnt.items()):
        if repo == "SkelterLabsInc/JaQuAD":
            sources.append(("jaquad_dev", _jaquad_dev_texts(paths)))
        elif repo == "hotpotqa/hotpot_qa":
            sources.append(("hotpot_validation", _hotpot_validation_texts(paths)))
    extra = Path(cfg.root) / USER_EVAL_EXTRA
    if extra.is_dir():
        for p in sorted(extra.iterdir()):
            if p.is_file() and p.suffix.lower() in (".jsonl", ".ndjson", ".json"):
                kind = "jsonl" if p.suffix.lower() in (".jsonl", ".ndjson") else "json"
                sources.append(checks.EvalSource(label="extra:" + p.name, chain=(str(p),), kind=kind, required=False, description="owner's extra eval file"))
    return sources


def eval_sources(cfg: Cfg):
    """(EvalFiles, all decontamination sources); cached for 8b after a kernel restart is rebuilt from the files."""
    try:
        ef = checks.find_eval_files(cfg.root, extra_dirs=[cfg.eval_extra_dir] if cfg.eval_extra_dir else [], strict=True)
    except checks.EvalFilesMissing as e:
        raise StepFailed(f"required eval files are missing, so decontamination would be incomplete: {e}")
    return ef, decontam_sources(cfg, ef)


def tier_table(per_source: dict) -> List[dict]:
    """T1..T8 -> the files that stand for them and how many items / shingles each contributed to the eval index."""
    out = []
    for tier, names in checks.EVAL_TIER_FILES.items():
        out.append({"tier": tier, "files": {n: {"items": (per_source.get(n) or {}).get("items", 0), "shingles": (per_source.get(n) or {}).get("shingles", 0)} for n in names}})
    return out


@step("6b vocabulary and eval files")
def step_vocab_eval(cfg: Cfg):
    notes: List[str] = []
    # ---- the model vocabulary: never the census derived from a corpus
    if cfg.vocab_override:
        V = checks.vocab_from_json_bytes(Path(cfg.vocab_override).read_bytes(), str(cfg.vocab_override))
    else:
        V = checks.load_vocab(cfg.root, out_dir=str(cfg.out / "state"), allow_derive=False, progress=cfg.progress)
    require(not V.derived, "the vocabulary was derived from a corpus census, not read from the model's own file")
    require(len(V.chars) == checks.EXPECTED_SINGLE_CHARS and V.n_entries == checks.EXPECTED_VOCAB_SIZE,
            f"vocabulary has {len(V.chars)} single characters / {V.n_entries} entries, expected {checks.EXPECTED_SINGLE_CHARS} / {checks.EXPECTED_VOCAB_SIZE}")
    vocab_sha_match = V.sha256_chars == KNOWN_VOCAB_SHA
    if not vocab_sha_match:
        cfg.log("!" * 20 + f" MISMATCH: vocabulary character set sha256 {V.sha256_chars[:16]}... differs from the one seen on the owner's PC ({KNOWN_VOCAB_SHA[:16]}...)")
        require(cfg.allow_vocab_mismatch,
                f"the vocabulary ({V.source}) has a different character set than the model's vocabulary on the owner's PC: a stale file would "
                f"silently decide which rows are dropped as out-of-vocabulary.  Fix the file, or set ALLOW_VOCAB_MISMATCH = True in the notebook "
                f"if the model's vocabulary really changed.")
        notes.append("vocabulary sha256 MISMATCH vs the owner's PC copy (accepted: ALLOW_VOCAB_MISMATCH)")
    Path(cfg.work).mkdir(parents=True, exist_ok=True)
    cfg.vocab_file.write_text(json.dumps({"vocab": {c: i for i, c in enumerate(sorted(V.chars))}, "source": V.source}, ensure_ascii=False), encoding="utf-8")
    _CACHE["vocab"] = V
    # ---- the runtime symbol normaliser: checks.py == convert.py == the controller source (when reachable)
    require(checks.RUNTIME_NORMALIZE == convert.NORMALIZE, "checks.py and convert.py disagree on the symbol normaliser")
    ctrl = _controller_source(cfg)
    if ctrl is not None:
        ok, why = checks.verify_runtime_normalizer(ctrl)
        require(ok, f"the runtime normaliser differs from qwenputer_controller_v30.py: {why}")
        notes.append("normaliser identical to the controller source")
    else:
        notes.append("controller source not found in the bundle (normaliser not compared with it)")
    # ---- eval files
    ef, sources = eval_sources(cfg)
    cfg.log(ef.report())
    labels = {s.label for s in ef.sources}
    if cfg.require_v48:
        missing = [n for n in V48_EVAL_NAMES if "v48:" + n not in labels]
        require(not missing, f"v48 unseen_eval texts not found: {missing}")
    index = checks.build_eval_index(sources, ngram=cfg.ngram, progress=lambda label, n: cfg.log(f"  {label}: {n:,} shingles so far"))
    st = index.stats()
    required = {s.label for s in ef.sources if s.required}
    empty = [lab for lab in required if st["per_source"].get(lab, {}).get("items", 0) == 0]
    require(not empty, f"eval sources without any item: {empty}")
    no_shingles = [lab for lab in sorted(required) if st["per_source"].get(lab, {}).get("shingles", 0) == 0]
    require(not no_shingles, f"eval sources that contributed no {cfg.ngram}-gram at all (a manifest of ids / metadata only would index nothing and "
                             f"decontaminate nothing): {no_shingles}")
    require(st["distinct_shingles"] > 0, "the eval index is empty")
    for key in cfg.keys:
        want = {"jaquad": "jaquad_dev", "hotpot": "hotpot_validation"}.get(key)
        if want:
            require(st["per_source"].get(want, {}).get("items", 0) > 0, f"{want} was not indexed (dev/validation files for decontamination)")
    _CACHE["eval_index"], _CACHE["eval_sources"], _CACHE["eval_files"] = index, sources, ef
    vd = V.as_dict()
    vd["source"] = vd["source"][len("cache:"):] if vd["source"].startswith("cache:") else vd["source"]          # same text on a re-run that reads the cache
    tiers = tier_table(st["per_source"])
    cfg.log("  T1-T8 coverage: " + "; ".join(f"{t['tier'].split(' ')[0]} " + "+".join(f"{n}({v['shingles']:,})" for n, v in t["files"].items()) for t in tiers))
    info = {"vocab": dict(vd, sha_matches_owner_pc_copy=vocab_sha_match),
            "eval": {"sources": [{"label": s.label, "where": s.display, "required": s.required} for s in ef.sources],
                     "extra_sources": [s.label if isinstance(s, checks.EvalSource) else s[0] for s in sources[len(ef.sources):]],
                     "missing_optional": ef.missing_optional, "notes": ef.notes, "tiers": tiers,
                     "index": {"ngram": st["ngram"], "distinct_shingles": st["distinct_shingles"], "per_source": st["per_source"],
                               "shingles_in_several_eval_items": st["shingles_in_several_eval_items"],
                               "max_eval_items_per_shingle": st["max_eval_items_per_shingle"]}},
            "notes": notes}
    save_state(cfg, "step6b_vocab_eval", info)
    return (f"vocabulary {len(V.chars)} chars from {V.source}; {len(ef.sources)} eval sources + {len(sources) - len(ef.sources)} dev/validation/extra sources, "
            f"{st['distinct_shingles']:,} distinct {cfg.ngram}-gram shingles; {len(ef.missing_optional)} optional suites not found; " + "; ".join(notes)), info


# ====================================================================================================== 6c: raw checks
def _squad_text(r):
    return "\n".join([str(r.get("title", "")), str(r.get("context", "")), str(r.get("question", "")), " ".join(str(x) for x in r.get("answer_texts") or [])])


def _tengentoppa_text(r):
    return str(r.get("instruction", "")) + "\n" + str(r.get("refined_answer", ""))


def _toolace_text(r):
    return str(r.get("system", "")) + "\n" + "\n".join(str(m.get("value", "")) for m in (r.get("conversations") or []) if isinstance(m, dict))


RAW_TEXT = {"jaquad": _squad_text, "tengentoppa": _tengentoppa_text, "toolace": _toolace_text, "coding": checks.default_text,
            "wizard": checks.default_text, "hotpot": checks.default_text}


def raw_rows(cfg: Cfg, key: str, stats: Optional[dict] = None):
    """Rows of the mirrored train files.  Lenient like the converters (an undecodable byte or a damaged line does not abort a 1 GB file; it is
    counted in `stats` and reported loudly): the strict reader is kept for the old corpus."""
    for p in mirror_train_files(cfg, key):
        yield from checks.iter_rows(str(p), strict=False, stats=stats)


def mismatch_banner(cfg: Cfg, key: str, n: int, exp: int) -> None:
    cfg.log("!" * 24 + f" MISMATCH {key}: counted {n:,} records, {COUNT_SOURCE[key]} ({n - exp:+,}) " + "!" * 24)


@step("6c raw checks")
def step_raw_checks(cfg: Cfg):
    V = _CACHE.get("vocab")
    require(V is not None, "run step 6b first (vocabulary)")
    out, mism = {}, []
    for key in cfg.keys:
        tf = RAW_TEXT[key]
        prefixes: collections.Counter = collections.Counter()
        bad: dict = {}

        def rows():
            for r in raw_rows(cfg, key, bad):
                if key == "jaquad":
                    prefixes[str(r.get("id"))[:3]] += 1
                yield r
        t0 = time.time()
        oov = checks.oov_stats(rows(), V, text_fn=tf, progress=cfg.progress)
        ln = checks.length_stats(raw_rows(cfg, key), text_fn=tf, progress=cfg.progress)["ALL"]
        n = oov["n_rows"]
        bad_lines, repl = bad.get("bad_lines", 0), bad.get("decode_replacements", 0)
        if bad_lines or repl:
            cfg.log("!" * 24 + f" BAD INPUT {key}: {bad_lines:,} damaged JSON line(s) skipped by this check (the converter counts them as rejects), "
                    f"{repl:,} line(s) with undecodable bytes (replaced by U+FFFD; such rows fail the vocabulary filter) " + "!" * 24)
        require(n > 0 and ln["n"] == n, f"{key}: no records or the two passes disagree ({n} vs {ln['n']})")
        dump = cfg.out / "reading" / f"raw_{key}_100.txt"
        checks.sample_for_reading(raw_rows(cfg, key), n=100, seed=7, out_path=str(dump), text_fn=tf,
                                  title=f"{key} RAW records (before conversion)", vocab=V, progress=cfg.progress)
        total = n + bad_lines
        exp = (cfg.expected_counts or {}).get(key)
        match = None if exp is None else (total == exp)
        if match is False:
            mismatch_banner(cfg, key, total, exp)
            mism.append({"dataset": key, "counted": total, "expected": exp, "source": COUNT_SOURCE[key]})
        if key == "jaquad":
            badp = {p: c for p, c in prefixes.items() if p in ("de-", "te-")}
            require(not badp, f"jaquad train files contain dev/test ids: {badp}")
        out[key] = {"n_raw": n, "bad_lines": bad_lines, "decode_replacements": repl, "n_total": total, "expected": exp, "count_match": match,
                    "count_source": COUNT_SOURCE[key],
                    "oov": {k: oov[k] for k in ("n_chars", "oov_frac", "rows_with_any_oov", "rows_over_threshold", "share_rows_over_threshold", "n_distinct_oov_chars")},
                    "top_oov": oov["top_oov"][:8],
                    "length": {k: ln[k] for k in ("mean", "p50", "p90", "p99", "max", "over_2048", "share_over_2048", "over_1900", "share_over_1900")},
                    "id_prefixes": dict(prefixes) if key == "jaquad" else None, "reading_file": str(dump)}
        cfg.log(f"  {key}: {total:,} records, OOV chars {oov['oov_frac']:.4%}, rows with >1% OOV {oov['share_rows_over_threshold']:.2%}, "
                f"rows over 2,048 chars {ln['share_over_2048']:.2%}  ({time.time() - t0:.0f} s)")
    if mism and cfg.stop_on_count_mismatch:
        raise StepFailed("record counts differ from the cards for: " + ", ".join(m["dataset"] for m in mism) + " (STOP_ON_COUNT_MISMATCH is on)")
    save_state(cfg, "step6c_raw_checks", {"datasets": out, "mismatches": mism})
    tail = "all counts match the cards" if not mism else f"MISMATCH in {len(mism)}: " + ", ".join(f"{m['dataset']} {m['counted']:,} vs {m['expected']:,}" for m in mism)
    return f"{len(out)} datasets read end to end (bad lines are counted, not fatal); {tail}", {"datasets": out, "mismatches": mism}


def n_records(raw: Optional[dict], key: str) -> Optional[int]:
    """Records the converter will read for `key`: parseable rows + damaged lines (it counts those as rejects)."""
    d = ((raw or {}).get("datasets", {}).get(key) or {})
    if d.get("n_raw") is None:
        return None
    return d["n_raw"] + d.get("bad_lines", 0)


# ====================================================================================================== 7-0: the sandbox must prove itself on THIS machine
@step("7-0 sandbox self-test")
def step_sandbox_selftest(cfg: Cfg):
    """Runs before any dataset code does, with Drive still mounted: the child must run as an unprivileged user, must not be able to list or
    write the mounted Drive, must refuse the escape payloads of the review, import every allowed module under the memory limit (RLIMIT_AS, -S),
    stop an endless loop and cap its output.  No weaker fallback exists: if this fails, nothing is converted."""
    r = checks.sandbox_selftest(require_os_containment=cfg.containment_required, drive_unmounted=cfg.drive_detach is not None)
    failed = [c for c in r["checks"] if not c["ok"]]
    for c in r["checks"]:
        cfg.log(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['name']}" + ("" if c["ok"] else f": {c['detail']}"))
    require(r["ok"], f"the sandbox self-test failed ({len(failed)} of {len(r['checks'])} checks): " + "; ".join(f"{c['name']}: {c['detail']}" for c in failed))
    require(r["backend"] == checks.SANDBOX_BACKEND, "unexpected sandbox backend")
    if cfg.containment_required:
        require(r["run_as"] not in (None, "root"), "the sandbox child does not run as an unprivileged user")
    rec = {"backend": r["backend"], "run_as": r["run_as"], "n_checks": len(r["checks"]), "containment_required": cfg.containment_required,
           "drive_unmounted_during_code_runs": cfg.drive_detach is not None, "checks": [{"name": c["name"], "ok": c["ok"]} for c in r["checks"]]}
    _CACHE["sandbox"] = rec                                    # valid for THIS process only: a new runtime proves itself again
    text = jdump(rec)
    write_once(cfg.out / "state" / f"sandbox_selftest_{sha256_text(text)[:8]}.json", text)          # a record, named by its content: never a conflict
    return (f"{len(r['checks'])} checks passed; backend {r['backend']}; child runs as {r['run_as'] or 'the notebook user (no OS barrier: local test only)'}"), \
        {"selftest": r}


def ensure_sandbox_ok(cfg: Cfg) -> dict:
    """The self-test result of THIS process (7a / 7b never start code without it); a fresh runtime runs it again, a result saved on Drive by an
    earlier runtime is never trusted (the test is about the machine that is about to run the code)."""
    st = _CACHE.get("sandbox")
    if st is None or (cfg.containment_required and st.get("run_as") in (None, "root")):
        step_sandbox_selftest(cfg)
        st = _CACHE["sandbox"]
    require(st["backend"] == checks.SANDBOX_BACKEND, "the sandbox self-test belongs to another backend: run step 7-0 again")
    if cfg.containment_required:
        require(st.get("run_as") not in (None, "root"), "the sandbox self-test shows no OS barrier: run step 7-0 again on the Colab runtime")
    return st


def last_sandbox_record(cfg: Cfg) -> Optional[dict]:
    """What registry.json says about the sandbox: this process's self-test, else the newest record on Drive."""
    if _CACHE.get("sandbox") is not None:
        return _CACHE["sandbox"]
    files = sorted((cfg.out / "state").glob("sandbox_selftest_*.json"), key=lambda q: q.stat().st_mtime) if (cfg.out / "state").is_dir() else []
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


# ====================================================================================================== 7: converters
def new_job(key: str, start: int = 0, stop: Optional[int] = None, out_root: Optional[Path] = None, shard: bool = False) -> dict:
    ident = key if not shard else f"{key}@{start}-{'end' if stop is None else stop}"
    return {"id": ident, "key": key, "start": start, "stop": stop, "out_root": out_root, "shard": shard}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)                      # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259                        # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate(job: dict, grace_s: float = 10.0) -> None:
    """Stop a converter and the sandbox children it started (they live in the converter's session / their own sessions)."""
    p = job["proc"]
    pid = getattr(p, "pid", None)
    try:
        if os.name == "posix" and pid:
            os.killpg(pid, signal.SIGTERM)
        elif hasattr(p, "terminate"):
            p.terminate()
    except (OSError, AttributeError):
        pass
    t0 = time.time()
    while time.time() - t0 < grace_s and p.poll() is None:
        time.sleep(0.2)
    if p.poll() is None:
        try:
            if os.name == "posix" and pid:
                os.killpg(pid, signal.SIGKILL)
            elif hasattr(p, "kill"):
                p.kill()
        except (OSError, AttributeError):
            pass


def _converter_cmd(cfg: Cfg, job: dict, limit: Optional[int]) -> List[str]:
    key = job["key"]
    cmd = [sys.executable, "-X", "utf8", str(Path(convert.__file__)), "--dataset", key, "--raw-dir", str(cfg.mirror_dir(key)),
           "--out-dir", str(job["out_root"]), "--vocab", str(cfg.vocab_file), "--max-chars", str(cfg.max_chars),
           "--sample-dump", "0" if job["shard"] else "100"]
    if job["start"]:
        cmd += ["--start", str(job["start"])]
    stop = job["stop"] if job["stop"] is not None else limit
    if stop:
        cmd += ["--limit", str(stop)]
    if not cfg.run_sandbox:
        cmd += ["--no-sandbox"]
    elif cfg.containment_required:
        cmd += ["--require-os-containment"]
    if cfg.python_allow_unverified:
        cmd += ["--allow-unverified-python"]
    return cmd


def _run_window(cfg: Cfg, jobs: List[dict], limit: Optional[int], smoke: bool) -> dict:
    """Run converter subprocesses (at most cfg.parallel at a time, in the given order) until all have finished.  An interrupt or any other
    exception terminates every converter (and the sandbox children in their sessions) before it propagates: no orphan keeps running
    behind a second 'Run all'.  A lock file per job refuses a second converter for the same job while the first one is alive."""
    logdir, lockdir = Path(cfg.work) / "logs", Path(cfg.work) / "locks"
    logdir.mkdir(parents=True, exist_ok=True)
    lockdir.mkdir(parents=True, exist_ok=True)
    pending, running, rcs = list(jobs), [], {}
    t0, last_beat = time.time(), time.time()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0")
    try:
        while pending or running:
            while pending and len(running) < max(1, cfg.parallel):
                job = pending.pop(0)
                job["out_root"].mkdir(parents=True, exist_ok=True)
                lock = lockdir / f"convert_{job['id'].replace('@', '_')}_{'smoke' if smoke else 'full'}.lock"
                if lock.is_file():
                    try:
                        other = int(lock.read_text().strip() or 0)
                    except ValueError:
                        other = 0
                    require(not _pid_alive(other), f"a converter for {job['id']} is still running (pid {other}); stop it (Runtime > Interrupt execution) or wait "
                                                   f"for it - starting a second one would fight over the same files")
                cmd = _converter_cmd(cfg, job, limit)
                lp = logdir / f"convert_{'smoke' if smoke else 'full'}_{cfg.converter_tag()}_{job['id'].replace('@', '_')}.log"
                fh = open(lp, "w", encoding="utf-8")
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(cfg.work), start_new_session=(os.name == "posix"))
                lock.write_text(str(getattr(proc, "pid", 0) or 0), encoding="utf-8")
                running.append(dict(job, proc=proc, fh=fh, log=lp, t0=time.time(), lock=lock, out=job["out_root"] / job["key"]))
                cfg.log(f"  started convert.py --dataset {job['key']}" + (f" --start {job['start']}" if job["start"] else "")
                        + (f" --limit {job['stop'] if job['stop'] is not None else limit}" if (job["stop"] is not None or limit) else "") + f"  (log {lp.name})")
            time.sleep(cfg.poll_s)
            for job in list(running):
                rc = job["proc"].poll()
                if rc is not None:
                    job["fh"].close()
                    try:
                        job["lock"].unlink()
                    except OSError:
                        pass
                    rcs[job["id"]] = {"rc": rc, "seconds": round(time.time() - job["t0"], 1), "log": str(job["log"])}
                    running.remove(job)
                    cfg.log(f"  finished {job['id']}: rc={rc} after {rcs[job['id']]['seconds']} s")
            if running and time.time() - last_beat >= cfg.heartbeat_s:
                last_beat = time.time()
                for job in running:
                    done = sum(p.stat().st_size for p in job["out"].glob("*.part-*")) if job["out"].is_dir() else 0
                    cfg.log(f"  [{(time.time() - t0) / 60:5.1f} min] {job['id']} still running, {done / 1e6:.1f} MB of rows written so far")
    except BaseException:
        for job in running:
            cfg.log(f"  stopping converter {job['id']} ...")
            _terminate(job)
            try:
                job["fh"].close()
                job["lock"].unlink()
            except OSError:
                pass
        raise
    return rcs


def _run_converters(cfg: Cfg, jobs: List[dict], smoke: bool, limit: Optional[int] = None, wave_size: Optional[int] = None,
                    on_wave: Optional[Callable] = None) -> dict:
    """convert.py as one subprocess per job.  wave_size=None: one window for all jobs (Drive stays unmounted for the whole window).
    wave_size=n: the jobs run in waves of n; after each wave Drive is mounted again and on_wave(finished_ok_jobs, results) can publish what
    is done, so a disconnect loses at most one wave.  Failed jobs raise StepFailed after the finished ones have been handed to on_wave."""
    chunks = [list(jobs)] if not wave_size else [list(jobs[i:i + wave_size]) for i in range(0, len(jobs), wave_size)]
    results: dict = {}
    for chunk in chunks:
        if not chunk:
            continue
        with untrusted_code_window(cfg):
            res = _run_window(cfg, chunk, limit, smoke)
        results.update(res)
        ok_jobs = [j for j in chunk if res.get(j["id"], {}).get("rc") == 0]
        if on_wave and ok_jobs:
            on_wave(ok_jobs, res)
        bad = {k: v for k, v in res.items() if v["rc"] != 0}
        if bad:
            tails = []
            for k, v in bad.items():
                tails.append(f"--- {k} (rc {v['rc']}) log tail ---\n" + "\n".join(Path(v["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
            raise StepFailed("converter(s) failed: " + ", ".join(f"{k} rc={v['rc']}" for k, v in bad.items()) + "\n" + "\n".join(tails))
    return results


def read_manifest(d: Path) -> dict:
    p = Path(d) / "manifest.json"
    require(p.is_file(), f"{p} is missing")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_manifest(cfg: Cfg, key: str, d: Path, limit: Optional[int], n_raw: Optional[int], start: int = 0, allow_empty: bool = False) -> dict:
    """limit = absolute stop index of the run (None = to the end), start = first index; n_raw = records of the whole dataset (or None)."""
    man = read_manifest(d)
    require(man.get("balanced") is True, f"{key}: converter bookkeeping is not balanced (n_in != n_out + rejects)")
    require(man["n_out"] > 0 or allow_empty, f"{key}: the converter emitted no row from {man['n_in']} records")
    if limit:
        want = (limit if n_raw is None else min(limit, n_raw)) - start
    else:
        want = None if n_raw is None else n_raw - start
    if want is not None:
        require(man["n_in"] == want, f"{key}: converter read {man['n_in']:,} records, expected {want:,}")
    n_files = sum(e["rows"] for e in man["files"].values())
    require(n_files == man["n_out"], f"{key}: split files hold {n_files} rows, manifest says {man['n_out']}")
    for sp in SPLITS:
        require(man["files"][f"{sp}.jsonl"]["rows"] == man["splits"].get(sp, 0), f"{key}: split counts disagree for {sp}")
    return man


def iter_stage_a(cfg: Cfg, key: str):
    """(split, raw JSON line) of the unfiltered converted rows, train then dev then heldout; the position in this order is the row's index."""
    d = cfg.conv_dir(key)
    for sp in SPLITS:
        with open(d / f"{sp}.jsonl", encoding="utf-8", newline="\n") as f:
            for line in f:
                if line.strip():
                    yield sp, line


def validate_rows(cfg: Cfg, key: str, ck: "convert.Checks") -> dict:
    """Independent pass over a dataset's converted files: every row re-read from JSON must satisfy convert.validate_row, ids are unique,
    a group never spans two splits, no dev/test id of JaQuAD, spans cover the injected observations."""
    ids, group_split, n, bad = set(), {}, 0, collections.Counter()
    for sp, line in iter_stage_a(cfg, key):
        row = json.loads(line)
        n += 1
        why = convert.validate_row(row, ck)
        if why:
            bad[why] += 1
        require(row["split"] == sp, f"{key}: row {row['id']} says split {row['split']} but sits in {sp}.jsonl")
        require(row["id"] not in ids, f"{key}: duplicate row id {row['id']}")
        ids.add(row["id"])
        g = row.get("group_key")
        if g is not None and key in GROUP_INDEPENDENT:                # rows of the other datasets are their own group: nothing to check
            require(group_split.setdefault(g, sp) == sp, f"{key}: group {g!r} appears in two splits ({group_split[g]} and {sp})")
        if key == "jaquad":
            require(not row["id"].startswith(("v49acq_jaquad_de-", "v49acq_jaquad_te-")), f"{key}: dev/test id in the rows: {row['id']}")
    require(not bad, f"{key}: rows that fail convert.validate_row after the JSON round trip: {dict(bad)}")
    return {"rows": n, "groups": len(group_split)}


def _converted_checks(cfg: Cfg, key: str, V) -> dict:
    paths = [str(cfg.conv_dir(key) / f"{sp}.jsonl") for sp in SPLITS]

    def rows():
        for p in paths:
            yield from checks.iter_rows(p)
    oov = checks.oov_stats(rows(), V, progress=cfg.progress)
    ln = checks.length_stats(rows(), progress=cfg.progress)["ALL"]
    require(oov["rows_over_threshold"] == 0, f"{key}: {oov['rows_over_threshold']} converted rows have more than 1% characters outside the vocabulary")
    require(ln["over_2048"] == 0 and ln["chars_over_2000"] == 0, f"{key}: converted rows longer than the cap (over 2,048: {ln['over_2048']}, over 2,000 chars: {ln['chars_over_2000']})")
    require(ln["full_text_mismatch"] == 0, f"{key}: {ln['full_text_mismatch']} rows with full_text != prompt + target")
    return {"oov_frac": oov["oov_frac"], "rows_with_any_oov": oov["rows_with_any_oov"], "rows_over_threshold": oov["rows_over_threshold"],
            "normalizer_changed_rows": oov["rows_changed_by_normalizer"],
            "length": {k: ln[k] for k in ("n", "mean", "p50", "p90", "p99", "max", "over_1900", "over_2048", "chars_over_2000", "full_text_mismatch")}}


def _convert_ck(cfg: Cfg):
    return convert.Checks(vocab_path=str(cfg.vocab_file), max_chars=cfg.max_chars, run_code=False, use_external=False)


def check_sandbox_backend(key: str, man: dict, cfg: Cfg) -> None:
    """A dataset that executes code must have been converted with the executor-profile sandbox, as an unprivileged user on Colab."""
    if key != "coding" or not cfg.run_sandbox:
        return
    b = man.get("backends") or {}
    require(b.get("sandbox") == checks.SANDBOX_BACKEND, f"{key}: converted with sandbox backend {b.get('sandbox')!r}, not {checks.SANDBOX_BACKEND!r}")
    if cfg.containment_required:
        require(b.get("sandbox_run_as") not in (None, "root"), f"{key}: the sandbox child did not run as an unprivileged user ({b.get('sandbox_run_as')!r})")


def _dir_bytes(d: Path) -> int:
    return sum(p.stat().st_size for p in Path(d).glob("*.jsonl")) if Path(d).is_dir() else 0


@step("7a convert smoke")
def step_convert_smoke(cfg: Cfg):
    require(cfg.vocab_file.is_file(), "run step 6b first (vocabulary file)")
    if cfg.run_sandbox:
        ensure_sandbox_ok(cfg)
    raw = load_state(cfg, "step6c_raw_checks")
    t0 = time.time()
    jobs = [new_job(k, stop=cfg.smoke_limit, out_root=cfg.conv_dir(k, smoke=True).parent) for k in CONVERT_ORDER if k in cfg.keys]
    rcs = _run_converters(cfg, jobs, smoke=True, limit=cfg.smoke_limit)
    ck = _convert_ck(cfg)
    res, est, est_bytes = {}, {}, 0
    for key in cfg.keys:
        d = cfg.conv_dir(key, smoke=True)
        n_raw = n_records(raw, key)
        man = _check_manifest(cfg, key, d, cfg.smoke_limit, n_raw)
        check_sandbox_backend(key, man, cfg)
        ids = set()
        n = 0
        for sp in SPLITS:
            for r in checks.iter_rows(str(d / f"{sp}.jsonl")):
                why = convert.validate_row(r, ck)
                require(why is None, f"{key}: smoke row {r.get('id')} invalid: {why}")
                require(r["id"] not in ids and r["split"] == sp, f"{key}: duplicate id or wrong split file in the smoke rows")
                ids.add(r["id"])
                n += 1
        require(n == man["n_out"], f"{key}: smoke files hold {n} rows, manifest says {man['n_out']}")
        res[key] = {"n_in": man["n_in"], "n_out": man["n_out"], "keep_rate": man["keep_rate"], "seconds": rcs[key]["seconds"],
                    "top_rejects": dict(list(man["rejects"].items())[:5]), "backends": man.get("backends")}
        if n_raw and man["n_in"]:
            est[key] = round(rcs[key]["seconds"] * n_raw / man["n_in"] / 60, 1)
            est_bytes += int(_dir_bytes(d) / man["n_in"] * n_raw)
        cfg.log(f"  {key}: {man['n_in']:,} in -> {man['n_out']:,} rows ({man['keep_rate']:.1%}), {rcs[key]['seconds']:.0f} s, sandbox: {(man.get('backends') or {}).get('sandbox')}")
    if est:
        cfg.log("  rough full-run estimate (minutes, from this smoke run; Colab may differ; coding / tengentoppa run in shards): "
                + ", ".join(f"{k} ~{v:g}" for k, v in est.items()))
        cfg.log("  keep the browser tab open during 7b (or use Colab's background execution): a closed tab can stop a running cell.  Finished "
                "shards and datasets are already on Drive, so a re-run continues where it stopped.")
    est_drive = int(est_bytes * 2.5) + (256 << 20)            # stage A + final rows + dumps (the smoke rows are the first records: a rough measure)
    out = {"datasets": res, "estimate_minutes": est, "est_converted_bytes": est_bytes, "est_drive_bytes": est_drive}
    save_state(cfg, "step7a_smoke", {"estimate_minutes": est, "est_converted_bytes": est_bytes, "est_drive_bytes": est_drive})
    return (f"{len(res)} converters ran on the first {cfg.smoke_limit:,} records each; every row passed validate_row again after reading it back "
            f"({time.time() - t0:.0f} s); about {est_drive / 2**30:.1f} GiB of Drive space is needed for the rows"), out


def restore_stage_a(cfg: Cfg, key: str) -> bool:
    """Copy the unfiltered rows of an earlier run from Drive to the local disk (resume).  False if there is none.  The Drive folder is keyed by
    the converter tag (convert.py + sandbox + normaliser + vocabulary), so rows made by other converter code are simply not found."""
    src = cfg.stage_a_drive(key)
    meta_p = src / "stageA_meta.json"
    if not meta_p.is_file():
        return False
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    require(meta.get("converter") == cfg.converter_hashes(),
            f"{src} was made by different converter code or another vocabulary than this notebook has.  Nothing was overwritten or deleted.")
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dst = cfg.conv_dir(key)
    dst.mkdir(parents=True, exist_ok=True)
    for name in STAGE_A_FILES:
        s = src / name
        require(s.is_file(), f"{s} is missing from an earlier run")
        d = dst / name
        want = man["files"][name]["sha256"] if name in man["files"] else None
        if d.is_file() and (want is None or checks.sha256_file(str(d)) == want):
            continue
        shutil.copyfile(s, d)
        if want is not None:
            require(checks.sha256_file(str(d)) == want, f"{name} restored from Drive does not match the manifest sha256")
    return True


def publish_stage_a(cfg: Cfg, key: str, V) -> None:
    src, dst = cfg.conv_dir(key), cfg.stage_a_drive(key)
    for name in STAGE_A_FILES:
        publish_file(src / name, dst / name)
    meta = {"schema": "v49_stage_a_meta/2", "converter": cfg.converter_hashes(),                    # the filter code is not part of what stage A depends on
            "dataset_commit_sha": (read_manifest(src).get("provenance") or {}).get("commit_sha"),
            "vocab_sha256_chars": V.sha256_chars, "limit": None}
    write_once(dst / "stageA_meta.json", jdump(meta))


def ensure_stage_a(cfg: Cfg, key: str) -> None:
    if (cfg.conv_dir(key) / "manifest.json").is_file():
        return
    require(restore_stage_a(cfg, key), f"{key}: no converted rows locally or on Drive - run step 7b first")


def shard_ranges(n_records_total: int, size: int) -> List[tuple]:
    """Contiguous record ranges [(start, stop)], the last with stop None (to the end of the file whatever it holds)."""
    k = max(1, math.ceil(n_records_total / max(1, size)))
    return [(i * size, (i + 1) * size if i < k - 1 else None) for i in range(k)]


def _shard_done_local(cfg: Cfg, key: str, a: int, b: Optional[int]) -> bool:
    return (cfg.shard_root(key, a, b) / key / "manifest.json").is_file()


def restore_shard(cfg: Cfg, key: str, a: int, b: Optional[int]) -> bool:
    src = cfg.shard_drive(key, a, b)
    if not (src / "shard_meta.json").is_file():
        return False
    meta = json.loads((src / "shard_meta.json").read_text(encoding="utf-8"))
    require(meta.get("converter") == cfg.converter_hashes() and meta.get("range") == [a, b],
            f"{src} was made by different converter code or for another range.  Nothing was overwritten or deleted.")
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    dst = cfg.shard_root(key, a, b) / key
    dst.mkdir(parents=True, exist_ok=True)
    for name in SHARD_FILES:
        s = src / name
        require(s.is_file(), f"{s} is missing from an earlier run")
        d = dst / name
        want = man["files"][name]["sha256"] if name in man["files"] else None
        if d.is_file() and (want is None or checks.sha256_file(str(d)) == want):
            continue
        shutil.copyfile(s, d)
        if want is not None:
            require(checks.sha256_file(str(d)) == want, f"{name} restored from Drive does not match the shard manifest sha256")
    return True


def publish_shard(cfg: Cfg, key: str, a: int, b: Optional[int]) -> None:
    src, dst = cfg.shard_root(key, a, b) / key, cfg.shard_drive(key, a, b)
    man = read_manifest(src)
    require(man.get("balanced") is True, f"{key} shard {a}-{b}: converter bookkeeping is not balanced")
    require((man.get("range") or {}).get("start") == a and (man.get("range") or {}).get("stop") == b, f"{key} shard {a}-{b}: manifest range differs")
    for name in SHARD_FILES:
        publish_file(src / name, dst / name)
    write_once(dst / "shard_meta.json", jdump({"schema": "v49_shard_meta/1", "key": key, "range": [a, b], "converter": cfg.converter_hashes(),
                                               "n_in": man["n_in"], "n_out": man["n_out"]}))


def _order_jobs(jobs: List[dict]) -> List[dict]:
    """Slowest dataset first; the shards of different datasets alternate so that a wave of two pairs a coding shard with a tengentoppa shard."""
    by_key: Dict[str, List[dict]] = collections.defaultdict(list)
    for j in jobs:
        by_key[j["key"]].append(j)
    ordered, keys = [], [k for k in CONVERT_ORDER if k in by_key]
    sharded = [k for k in keys if any(j["shard"] for j in by_key[k])]
    ordered += [j for group in itertools.zip_longest(*[by_key[k] for k in sharded]) for j in group if j is not None]
    for k in keys:
        if k not in sharded:
            ordered += by_key[k]
    return ordered


def preflight_drive(cfg: Cfg, smoke: Optional[dict]) -> None:
    need = (smoke or {}).get("est_drive_bytes")
    if not need or not cfg.check_disk:
        return
    try:
        free = shutil.disk_usage(cfg.root).free
    except OSError:
        return
    require(free >= need, f"Drive reports {free:,} B free, the 7a smoke run extrapolates to about {need:,} B for the converted and final rows: not enough room")


@step("7b convert full", flush_on_failure=True)
def step_convert_full(cfg: Cfg):
    V = _CACHE.get("vocab")
    require(V is not None and cfg.vocab_file.is_file(), "run step 6b first (vocabulary)")
    if cfg.run_sandbox:
        ensure_sandbox_ok(cfg)
    raw = load_state(cfg, "step6c_raw_checks")
    smoke = load_state(cfg, "step7a_smoke")
    preflight_drive(cfg, smoke)
    jobs, resumed, plans = [], [], {}
    for key in cfg.keys:
        if (cfg.conv_dir(key) / "manifest.json").is_file() or restore_stage_a(cfg, key):          # local rows or the Drive copy of an earlier run
            resumed.append(key)
            continue
        total = n_records(raw, key)
        if key in cfg.shard_keys and total is not None and total > cfg.shard_records:
            plans[key] = shard_ranges(total, cfg.shard_records)
            for a, b in plans[key]:
                if _shard_done_local(cfg, key, a, b) or restore_shard(cfg, key, a, b):
                    continue
                jobs.append(new_job(key, a, b, cfg.shard_root(key, a, b), shard=True))
        else:
            est = ((smoke or {}).get("estimate_minutes") or {}).get(key)
            require(est is None or est <= cfg.max_unsharded_minutes,
                    f"{key}: the smoke run estimates ~{est:g} min for an unsharded conversion (limit {cfg.max_unsharded_minutes:g}); a disconnect would lose all of it.  "
                    f"Put '{key}' into the shard list (Cfg.shard_keys) or raise max_unsharded_minutes.")
            jobs.append(new_job(key, out_root=cfg.conv_dir(key).parent))
    if resumed:
        cfg.log("  resumed from earlier converted rows: " + ", ".join(resumed))
    jobs = _order_jobs(jobs)
    if jobs:
        cfg.log(f"  {len(jobs)} conversion job(s) in waves of {max(1, cfg.parallel)}; each finished job is published to Drive at once")
    todo_keys = sorted({j["key"] for j in jobs} | set(plans))

    def finish(ok_jobs, _res):
        for j in ok_jobs:
            if j["shard"]:
                _check_manifest(cfg, j["key"], j["out_root"] / j["key"], j["stop"], n_records(raw, j["key"]), start=j["start"], allow_empty=True)
                publish_shard(cfg, j["key"], j["start"], j["stop"])
                cfg.log(f"  shard {j['id']} published to Drive")
            else:
                man = _check_manifest(cfg, j["key"], cfg.conv_dir(j["key"]), None, n_records(raw, j["key"]))
                check_sandbox_backend(j["key"], man, cfg)
                publish_stage_a(cfg, j["key"], V)
                cfg.log(f"  {j['key']} published to Drive")
    if jobs:
        _run_converters(cfg, jobs, smoke=False, wave_size=max(1, cfg.parallel), on_wave=finish)
    for key, ranges in plans.items():
        if (cfg.conv_dir(key) / "manifest.json").is_file():
            continue
        roots = [cfg.shard_root(key, a, b) for a, b in ranges]
        for (a, b), r in zip(ranges, roots):
            require((r / key / "manifest.json").is_file(), f"{key}: shard {a}-{b} has no rows (run 7b again)")
        merged = convert.merge_shards(key, roots, cfg.conv_dir(key).parent)
        require(not merged["refused_overwrite"], f"{key}: merged rows differ from an earlier merge in {cfg.conv_dir(key)}")
        cfg.log(f"  {key}: {len(ranges)} shards merged into {merged['n_out']:,} rows")
    ck = _convert_ck(cfg)
    res = {}
    for key in cfg.keys:
        d = cfg.conv_dir(key)
        n_raw = n_records(raw, key)
        man = _check_manifest(cfg, key, d, None, n_raw)
        require(not man.get("limit"), f"{key}: these rows were made with --limit {man.get('limit')}, not a full run")
        check_sandbox_backend(key, man, cfg)
        vr = validate_rows(cfg, key, ck)
        require(vr["rows"] == man["n_out"], f"{key}: {vr['rows']} rows read back, manifest says {man['n_out']}")
        cc = _converted_checks(cfg, key, V)
        publish_stage_a(cfg, key, V)                                                        # identical files are skipped, nothing is overwritten
        res[key] = {"manifest": man, "validated": vr, "checks": cc}
        cfg.log(f"  {key}: {man['n_in']:,} records -> {man['n_out']:,} rows; splits {man['splits']}; top rejects "
                + ", ".join(f"{k}={v:,}" for k, v in list(man["rejects"].items())[:3]))
    drive_sync(cfg, "step 7b")
    return (f"{len(res)} datasets converted ({len(todo_keys)} now, {len(resumed)} resumed); rows re-validated, no id/group/split problem, "
            f"no row over the cap or over 1% OOV; unfiltered rows are kept on Drive under {STAGE_A_NAME}/{cfg.converter_tag()}/"), {"datasets": res}


# ====================================================================================================== 8: filters
def dedupe_text(row: dict) -> str:
    """Text for WITHIN-dataset duplicate detection.  One passage legitimately carries several questions (JaQuAD, HotpotQA), so for those the
    question + evidence + answer part is compared, not the passage; everything else is compared whole."""
    if row.get("family") in ("jaquad_human", "hotpot_en_short"):
        p = row["prompt"]
        i = p.rfind("\n質問: ")
        return (p[i:] if i >= 0 else p) + row["target"]
    return row["full_text"]


def full_text(row: dict) -> str:
    return row["full_text"]


def rows_at(cfg: Cfg, key: str, alive: bytearray):
    for i, (_, line) in enumerate(iter_stage_a(cfg, key)):
        if alive[i]:
            yield json.loads(line)


class Combined:
    """The surviving rows of several datasets as ONE stream (so that 'boilerplate' is judged over all rows); locate() maps back."""

    def __init__(self, cfg: Cfg, keys, alive: Dict[str, bytearray]):
        self.cfg, self.keys, self.alive = cfg, list(keys), alive
        self.maps, self.starts, tot = [], [], 0
        for k in self.keys:
            idx = [i for i, a in enumerate(alive[k]) if a]
            self.maps.append(idx)
            self.starts.append(tot)
            tot += len(idx)
        self.total = tot

    def rows(self):
        for k in self.keys:
            yield from rows_at(self.cfg, k, self.alive[k])

    def rows_of(self, key: str):
        return rows_at(self.cfg, key, self.alive[key])

    def locate(self, g: int):
        j = bisect.bisect_right(self.starts, g) - 1
        return self.keys[j], self.maps[j][g - self.starts[j]]


def _drop(alive: Dict[str, bytearray], reason: Dict[str, bytearray], comb: Combined, positions, code: int) -> Dict[str, int]:
    per: collections.Counter = collections.Counter()
    for g in positions:
        k, i = comb.locate(g)
        if alive[k][i]:
            alive[k][i] = 0
            reason[k][i] = code
            per[k] += 1
    return dict(per)


def _eval_queries(cfg: Cfg, sources) -> List[str]:
    qs, seen = [], set()
    for s in sources:
        if not isinstance(s, checks.EvalSource):
            continue
        try:
            for strings in checks.iter_eval_items(s):
                q = strings[0].strip() if strings else ""
                if len(q) >= 20 and q not in seen:
                    seen.add(q)
                    qs.append(q)
        except Exception:                                                           # noqa: BLE001 - the semantic review is optional
            continue
    if len(qs) > cfg.semantic_max_queries:
        qs = sorted(qs, key=sha256_text)[:cfg.semantic_max_queries]
    return qs


def _examples(items, k=5):
    return items[:k]


def load_alive(cfg: Cfg, keys) -> Dict[str, bytearray]:
    """Survivor masks rebuilt from the drop lists that step 8 wrote (so 8b and 9 do not depend on anything in memory)."""
    alive = {}
    for k in keys:
        n = sum(1 for _ in iter_stage_a(cfg, k))
        a = bytearray(b"\x01") * n
        p = cfg.out / "filters" / f"{k}_dropped.tsv"
        require(p.is_file(), f"{p} is missing: run step 8 first")
        for line in p.read_text(encoding="utf-8").splitlines()[1:]:
            a[int(line.split("\t", 1)[0])] = 0
        alive[k] = a
    return alive


@step("8 filters", flush_on_failure=True)
def step_filters(cfg: Cfg):
    """Exact + MinHash duplicates inside a dataset, 13-gram decontamination, the old corpus, near siblings of train inside dev / heldout.  The
    drop lists and the state are written as soon as these are done; the optional reviews are step 8b."""
    keys = cfg.keys
    for k in keys:
        ensure_stage_a(cfg, k)
    index = _CACHE.get("eval_index")
    require(index is not None, "run step 6b first (eval index)")
    raw = load_state(cfg, "step6c_raw_checks") or {}
    stats: Dict[str, Any] = {k: {"counts": {}, "examples": {}} for k in keys}
    alive: Dict[str, bytearray] = {}
    reason: Dict[str, bytearray] = {}
    for k in keys:
        n = sum(1 for _ in iter_stage_a(cfg, k))
        require(n == read_manifest(cfg.conv_dir(k))["n_out"], f"{k}: converted files and manifest disagree")
        alive[k], reason[k] = bytearray(b"\x01") * n, bytearray(n)
        stats[k]["counts"]["converted"] = n
    cut = cfg.near_cut
    # ---- 1. exact duplicates and 2. MinHash near duplicates, inside one dataset (keep first)
    for k in keys:
        comb = Combined(cfg, [k], alive)
        ex = checks.exact_dups(comb.rows(), text_fn=dedupe_text, progress=cfg.progress)
        _drop(alive, reason, comb, ex["within_dup_positions"], R_EXACT)
        stats[k]["counts"]["after_exact_dedup"] = sum(alive[k])
        stats[k]["examples"]["exact"] = _examples(ex["examples"])
        comb = Combined(cfg, [k], alive)
        nd = checks.near_dups_within(comb.rows(), text_fn=dedupe_text, threshold=cut, progress=cfg.progress)
        _drop(alive, reason, comb, nd["dup_positions"], R_NEAR)
        stats[k]["counts"]["after_near_dedup"] = sum(alive[k])
        stats[k]["examples"]["near"] = _examples(nd["examples"])
        stats[k]["near"] = {"clusters": nd["n_clusters_gt1"], "largest_cluster": nd["largest_cluster"], "cluster_size_hist": nd["cluster_size_hist"],
                            "backend": nd["hasher_backend"], "k": nd["k"], "bands": nd["bands"], "estimate_cut": cut, "target_jaccard": cfg.near_threshold}
        cfg.log(f"  {k}: converted {stats[k]['counts']['converted']:,} -> exact {stats[k]['counts']['after_exact_dedup']:,} -> near {stats[k]['counts']['after_near_dedup']:,}"
                f" (clusters {nd['n_clusters_gt1']:,}, largest {nd['largest_cluster']:,})")
    # ---- 3. decontamination against the eval texts: one pass over all datasets
    comb = Combined(cfg, keys, alive)
    dc = checks.decontaminate(comb.rows(), index, text_fn=full_text, boilerplate_eval_df=cfg.boilerplate_eval_df, min_shared=cfg.min_shared,
                              max_examples=10, progress=cfg.progress)
    per = _drop(alive, reason, comb, dc["flagged_positions"], R_EVAL)
    for k in keys:
        stats[k]["counts"]["after_eval_decontam"] = sum(alive[k])
        stats[k]["examples"]["eval_overlap"] = []
    glob: Dict[str, Any] = {"decontam": {"n_rows": dc["n_rows"], "rows_flagged": dc["rows_flagged"], "rows_boilerplate_only": dc["rows_boilerplate_only"],
                                         "rows_with_any_shared_shingle": dc["rows_with_any_shared_shingle"], "shared_count_hist": dc["shared_count_hist"],
                                         "boilerplate": dc["boilerplate"], "real_shingles_top": dc["real_shingles_top"], "examples": dc["examples"],
                                         "flagged_by_dataset": per, "ngram": cfg.ngram, "boilerplate_eval_df": cfg.boilerplate_eval_df,
                                         "min_shared": cfg.min_shared, "limits": dc["limits"]}}
    cfg.log(f"  decontamination: {dc['rows_flagged']:,} of {dc['n_rows']:,} rows share a {cfg.ngram}-gram with an eval text and are dropped "
            f"({dc['rows_boilerplate_only']:,} rows shared only phrases that {cfg.boilerplate_eval_df}+ eval items share and are kept); by dataset {per}")
    # ---- 4. the old corpus (exact, then MinHash): two streaming passes over the old rows, one index of all new rows
    oc = cfg.old_corpus_path
    require(oc.is_file(), f"old corpus not found: {oc}")
    box = [0]

    def counted(it):
        for r in it:
            box[0] += 1
            yield r
    t0 = time.time()
    old_exact = checks.build_exact_index(counted(checks.iter_rows(str(oc))), text_fn=checks.default_text, progress=cfg.progress)
    n_old = box[0]
    cfg.log(f"  old corpus: {n_old:,} rows found in {oc.name} ({oc.stat().st_size:,} bytes)  ({time.time() - t0:.0f} s)")
    if cfg.expected_old_rows is not None:
        require(n_old == cfg.expected_old_rows, f"the old corpus has {n_old:,} rows, the plan says {cfg.expected_old_rows:,}: wrong or incomplete file, "
                                                f"so the dedupe against it would be incomplete")
    comb = Combined(cfg, keys, alive)
    ex = checks.exact_dups(comb.rows(), text_fn=full_text, against=old_exact, progress=cfg.progress)
    per_ex = _drop(alive, reason, comb, ex.get("against_dup_positions", []), R_OLD_EXACT)
    del old_exact
    comb = Combined(cfg, keys, alive)
    cr = checks.cross_near_dups(comb.rows(), checks.iter_rows(str(oc)), text_fn=full_text, probe_text_fn=checks.default_text,
                                threshold=cut, progress=cfg.progress)
    per_nr = _drop(alive, reason, comb, cr["matched_index_positions"], R_OLD_NEAR)
    glob["old_corpus"] = {"path": str(oc), "rows_found": n_old, "rows_expected": cfg.expected_old_rows, "bytes": oc.stat().st_size,
                          "exact_dropped_by_dataset": per_ex, "near_dropped_by_dataset": per_nr, "probe_rows_scanned": cr["probe_rows_scanned"],
                          "probe_rows_with_match": cr["probe_rows_with_match"], "examples": cr["examples"], "threshold": cfg.near_threshold,
                          "estimate_cut": cut}
    for k in keys:
        stats[k]["counts"]["after_old_corpus_dedup"] = sum(alive[k])
    cfg.log(f"  old corpus dedupe: exact {per_ex}, near {per_nr}")
    # ---- 5. dev / heldout hygiene.  Only JaQuAD (by article) and HotpotQA (by shared paragraph) have a natural group; the four synthetic
    # datasets split by ROW, so siblings of one seed can sit in train and in dev / heldout.  A dev / heldout row with a near sibling in train
    # (estimated Jaccard >= split_hygiene_threshold) is dropped; paraphrase-level siblings remain, which is why these files are labelled
    # 'not group-independent' everywhere.
    hyg: Dict[str, Any] = {}
    for k in keys:
        stats[k]["split_hygiene"] = None
        if not cfg.run_split_hygiene or k in GROUP_INDEPENDENT:
            continue
        comb = Combined(cfg, [k], alive)
        positions, texts = [], []
        for g, row in enumerate(comb.rows()):
            if row["split"] != "train":
                positions.append(g)
                texts.append({"id": row["id"], "full_text": dedupe_text(row)})
        info: Dict[str, Any] = {"dev_heldout_rows": len(positions), "dropped": 0, "threshold_estimate": cfg.split_hygiene_threshold}
        if texts:
            probe = ({"id": r["id"], "full_text": dedupe_text(r)} for r in comb.rows() if r["split"] == "train")
            sib = checks.cross_near_dups(texts, probe, threshold=cfg.split_hygiene_threshold, progress=cfg.progress)
            info.update({"dropped": len(sib["matched_index_positions"]), "train_rows_scanned": sib["probe_rows_scanned"], "examples": sib["examples"][:3]})
            _drop(alive, reason, comb, [positions[i] for i in sib["matched_index_positions"]], R_SPLIT_LEAK)
        hyg[k] = info
        stats[k]["split_hygiene"] = info
        if info["dropped"]:
            cfg.log(f"  {k}: {info['dropped']:,} of {info['dev_heldout_rows']:,} dev / heldout rows had a near sibling in train and were dropped")
    glob["split_hygiene"] = {"datasets": hyg, "threshold_estimate": cfg.split_hygiene_threshold, "group_independent": list(GROUP_INDEPENDENT)}
    for k in keys:
        stats[k]["counts"]["after_split_hygiene"] = sum(alive[k])
        stats[k]["counts"]["final_rows"] = sum(alive[k])
    # ---- the drop lists (one file per dataset) and the statistics: written NOW, before anything optional
    for k in keys:
        c = stats[k]["counts"]
        seq = [c[x] for x in FUNNEL]
        require(all(a >= b for a, b in zip(seq, seq[1:])), f"{k}: the funnel is not monotone: {seq}")
        require(c["final_rows"] == sum(alive[k]), f"{k}: final count disagrees with the survivor mask")
        c["n_raw"] = (raw.get("datasets", {}).get(k) or {}).get("n_total")
        lines = ["index\tid\treason\n"]
        for i, (_, line) in enumerate(iter_stage_a(cfg, k)):
            if not alive[k][i]:
                lines.append(f"{i}\t{json.loads(line)['id']}\t{REASON[reason[k][i]]}\n")
        write_once(cfg.out / "filters" / f"{k}_dropped.tsv", "".join(lines))
        stats[k]["dropped_by_reason"] = dict(collections.Counter(l.rstrip("\n").split("\t")[2] for l in lines[1:]))
    out = {"datasets": stats, "global": glob}
    save_state(cfg, "step8_filters", out)
    drive_sync(cfg, "step 8")
    fin = {k: stats[k]["counts"]["final_rows"] for k in keys}
    return (f"funnel done for {len(keys)} datasets; old corpus {n_old:,} rows found; decontamination dropped {dc['rows_flagged']:,}; "
            f"dev/heldout near siblings dropped {sum(v['dropped'] for v in hyg.values()):,}; rows kept per dataset: {fin}"), out


@step("8b optional reviews", flush_on_failure=True)
def step_reviews(cfg: Cfg):
    """Semantic look-alikes of the eval questions (for human reading) and the dataset x dataset duplicate matrix.  Neither removes a row; each
    has a wall-clock budget and says so when it was cut short.  Needs only the drop lists of step 8; result saved once (a re-run returns it)."""
    keys = cfg.keys
    for k in keys:
        ensure_stage_a(cfg, k)
    require(load_state(cfg, "step8_filters") is not None, "run step 8 first")
    done = load_state(cfg, "step8b_reviews")
    if done is not None:
        return "already done (state found; set RUN_SUFFIX to redo it)", {"reviews": done}
    alive = load_alive(cfg, keys)
    final = Combined(cfg, keys, alive)
    out: Dict[str, Any] = {"semantic": None, "cross_dataset": None, "rows": final.total}
    if cfg.run_semantic:
        info: dict = {}
        try:
            sources = _CACHE.get("eval_sources") or eval_sources(cfg)[1]
            qs = _eval_queries(cfg, sources)
            res = checks.semantic_neighbors(qs, final.rows(), text_fn=full_text, id_fn=lambda r, p: r.get("id"), top_n=3, min_cos=cfg.semantic_min_cos,
                                            min_shared_blocks=cfg.semantic_min_shared_blocks, max_block_df=cfg.semantic_max_block_df,
                                            progress=cfg.progress, deadline=time.time() + cfg.semantic_budget_s, info=info) if qs else []
            hit = [r for r in res if r["matches"]]
            hit.sort(key=lambda r: -r["matches"][0]["cosine"])
            text = checks.format_semantic_report(hit[:300])
            fname = f"semantic_review_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}.txt"
            write_once(cfg.out / "filters" / fname, text)
            best = [r["matches"][0]["cosine"] for r in hit]
            out["semantic"] = {"queries": len(qs), "queries_with_a_look_alike": len(hit), "best_cosine_ge_0.7": sum(1 for c in best if c >= 0.7),
                               "best_cosine_ge_0.9": sum(1 for c in best if c >= 0.9), "min_cos": cfg.semantic_min_cos,
                               "file": str(cfg.out / "filters" / fname), "note": "for reading only; no row was removed",
                               "truncated": bool(info.get("truncated")), "rows_scanned": info.get("rows_scanned"), "rows_total": final.total,
                               "budget_s": cfg.semantic_budget_s, "queries_sampled_from": "sha-sorted sample of the eval questions"}
            cfg.log(f"  semantic look-alikes: {len(hit)} of {len(qs)} eval questions have a row with cosine >= {cfg.semantic_min_cos}"
                    + (f"  [CUT SHORT by the {cfg.semantic_budget_s:g} s budget after {info.get('rows_scanned'):,} of {final.total:,} rows]" if info.get("truncated") else ""))
        except StepFailed:
            raise
        except Exception as e:                                                      # noqa: BLE001
            out["semantic"] = {"error": f"{type(e).__name__}: {e}"}
            cfg.log(f"!!! WARNING: the optional semantic review failed ({type(e).__name__}: {e}); recorded in the report, the run continues")
    if cfg.run_cross:
        info = {}
        try:
            cd = checks.cross_dataset_near_dups({k: final.rows_of(k) for k in keys}, text_fn=full_text, threshold=cfg.near_cut, progress=cfg.progress,
                                                deadline=time.time() + cfg.cross_budget_s, info=info)
            out["cross_dataset"] = {"rows": cd["rows"], "rows_with_match_in": cd["rows_with_match_in"], "examples": cd["examples"], "threshold": cfg.near_threshold,
                                    "estimate_cut": cfg.near_cut, "truncated": info.get("truncated"), "budget_s": cfg.cross_budget_s,
                                    "note": "report only; rows stay in their own dataset folder"}
            if info.get("truncated"):
                cfg.log(f"  cross-dataset duplicate matrix CUT SHORT by the {cfg.cross_budget_s:g} s budget ({info['truncated']} phase)")
        except Exception as e:                                                      # noqa: BLE001
            out["cross_dataset"] = {"error": f"{type(e).__name__}: {e}"}
            cfg.log(f"!!! WARNING: the optional cross-dataset duplicate matrix failed ({type(e).__name__}: {e}); recorded, the run continues")
    save_state(cfg, "step8b_reviews", out)
    drive_sync(cfg, "step 8b")
    sem, cd = out["semantic"] or {}, out["cross_dataset"] or {}
    return (f"semantic review: {'error' if 'error' in sem else ('cut short' if sem.get('truncated') else ('done' if sem else 'off'))}; "
            f"duplicate matrix: {'error' if 'error' in cd else ('cut short' if cd.get('truncated') else ('done' if cd else 'off'))}"), {"reviews": out}


# ====================================================================================================== 9: final rows, registry, report
def adoption_verdict(cfg: Cfg, key: str, final_rows: int, conv: dict, final_verification: Optional[dict] = None) -> dict:
    """A recommendation from the checks, never a fixed number.  The owner decides; every verdict is provisional until the 100 rows
    per dataset have been read by eye.  A dataset on hold (toolace, wizard) is NEVER promoted automatically."""
    why: List[str] = []
    if final_rows == 0:
        return {"code": "reject", "ja": "不採用（採用できる行が0件）", "why": ["no row survived the checks"]}
    if key == "toolace":
        return {"code": "hold", "ja": "保留（生成モデルが未確認: カードにも論文にも明記なし）", "why": ["adoption is gated on the generator model being verified"]}
    if key == "wizard":
        a = conv.get("arith_error_rate") or {}
        rate, n_st = a.get("statement_error_rate"), a.get("statements_checked") or 0
        shown = "測定不能" if rate is None else f"{rate:.2%}（{n_st:,}式中）"
        return {"code": "hold", "ja": f"保留（計画の「数学の明らかな誤りがあるものは避ける」に該当。測定できた単純な算術式の誤り率は {shown}で、これは誤り率の下限です。"
                                      f"自動では採用候補に上げません。100行を読んで、あなたが決めてください）",
                "why": [f"measured statement error rate {rate} over {n_st} statements (a lower bound); the owner's plan lists this dataset under 'avoid'"]}
    if key == "tengentoppa":
        return {"code": "conditional", "ja": "条件付き候補（Qwen2.5-32B生成・人手確認なし。目視100行を読んでから判断）", "why": ["generated by Qwen2.5-32B-Instruct, not checked by humans"]}
    if key == "coding":
        v = final_verification if final_verification is not None else {x[len("verification_"):]: n for x, n in (conv.get("extra") or {}).items() if x.startswith("verification_")}
        n_eq = v.get("stored_output_equals_sandbox", 0)
        n_un = v.get("unverified_no_execution", 0)
        return {"code": "candidate", "ja": f"採用候補（実行して答え自身の保存出力と一致したpython行 {n_eq:,}行のみ。他言語のcode_text行 {n_un:,}行は未実行・未検証）",
                "why": ["python rows: kept only when the answer's stored output equals the sandbox output", "code_text rows: not executed"]}
    if key == "jaquad":
        return {"code": "candidate", "ja": "採用候補（人手作成。CC BY-SA 3.0: 継承条件あり）", "why": ["human annotated"]}
    if key == "hotpot":
        return {"code": "candidate", "ja": "採用候補（人手作成・英語の短い行のみ。CC BY-SA 4.0: 継承条件あり）", "why": ["human annotated; English"]}
    return {"code": "candidate", "ja": "採用候補", "why": why}


def _final_stats(rows_iter) -> dict:
    n, target_chars, masked, fam, ver = 0, 0, 0, collections.Counter(), collections.Counter()
    tool_rows = 0
    en_rows = en_chars = 0
    for r in rows_iter:
        n += 1
        target_chars += len(r["target"])
        m = sum(b - a for a, b in r.get("loss_mask_spans") or [])
        masked += m
        fam[r["family"]] += 1
        if r.get("verification"):
            ver[r["verification"]] += 1
        tool_rows += bool(r.get("tool_steps"))
        if r.get("user_lang") == "en" or r.get("answer_lang") == "en" or str(r.get("language", "")).startswith("en"):
            en_rows += 1
            en_chars += len(r["target"]) - m
    return {"rows": n, "target_chars": target_chars, "masked_chars": masked, "supervised_chars": target_chars - masked, "families": dict(fam),
            "verification": dict(ver), "rows_with_tool_steps": tool_rows, "english_rows": en_rows, "english_supervised_chars": en_chars}


_SA_URL = {"CC-BY-SA-3.0": "https://creativecommons.org/licenses/by-sa/3.0/", "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/"}


def find_apache_license_text() -> Optional[str]:
    """The standard Apache-2.0 text (from `Apache License` to `END OF TERMS AND CONDITIONS`), taken from the licence file of an installed
    Apache-2.0 project (huggingface_hub is installed by cell 2), because Apache-2.0 section 4(a) asks that recipients get a copy of the
    licence and no other source is reachable from the Colab runtime.  None when it cannot be found (the notice then says so)."""
    from importlib import metadata
    for name in ("huggingface_hub", "requests", "pyarrow"):
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        for f in dist.files or []:
            if "LICENSE" not in f.name.upper():
                continue
            try:
                text = f.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            i, j = text.find("Apache License"), text.find("END OF TERMS AND CONDITIONS")
            if 0 <= i < j and "Version 2.0, January 2004" in text[i:i + 300]:
                return text[i:j + len("END OF TERMS AND CONDITIONS")].strip() + "\n"
    return None


def license_notice(info: dict, key: str, apache_text_available: bool) -> str:
    L = [f"LICENCE NOTICE for the rows of {info['repo']} (pinned commit {info['commit_sha']})", "",
         f"Licence: {info['license_name']}  <{info['license_url']}>", f"SPDX-style ids on the dataset card: {', '.join(info['license_ids'])}", ""]
    lines = info.get("license_lines") or []
    if lines:
        L += ["Lines of the dataset card that mention the licence:"] + [f"  | {x}" for x in lines] + [""]
    if any(i.lower().startswith("apache") for i in info["license_ids"]):
        L += ["Apache License 2.0, section 4: you must give recipients of the data or of derivative works a copy of the licence, keep the",
              "attribution notices, and state that the files were changed (these rows are a converted, filtered adaptation).",
              "The licence text is in LICENSE-Apache-2.0.txt next to this file." if apache_text_available else
              "The licence text could NOT be copied automatically (no installed Apache-2.0 package carried it); get it from https://www.apache.org/licenses/LICENSE-2.0.txt "
              "and keep it with these rows."]
    else:
        L += [f"Creative Commons {info['license_name']}: attribution + ShareAlike.  The full legal text is at {info['license_url']} (the licence deed link above).",
              "These rows are an ADAPTATION of the source data (converted, shortened, filtered).  They may be shared only under CC BY-SA 3.0 / 4.0 "
              "(or a licence CC lists as compatible), with the attribution in ATTRIBUTION.md.",
              "DO NOT merge ShareAlike rows into one redistributed single-licence file: JaQuAD (CC BY-SA 3.0) and HotpotQA (CC BY-SA 4.0) both embed "
              "Wikipedia text, and BY-SA 3.0 and 4.0 rows must stay separately labelled (each row also carries its own `license` and `source`)."]
    L += ["", "Upstream model terms of the generating models (Nemotron-4, Phi-3, Mixtral, calm3, Qwen2.5, WizardLM-2) were NOT checked."]
    return "\n".join(L) + "\n"


def _attribution_md(info: dict, key: str, cfg: Cfg) -> str:
    L = ["# Attribution and licence (keep this file with the rows)", "", f"Dataset: {info['repo']} (pinned commit `{info['commit_sha']}`)",
         f"Licence: {info['license_name']} <{info['license_url']}>", "Rows in this folder: converted to JSONL rows and filtered by v49_acquire (changes were made).", ""]
    L += [f"- {x}" for x in info["attribution"]]
    if info["share_alike"]:
        L += ["", "ShareAlike: these rows are an adaptation of the source data and must be shared under CC BY-SA (or a compatible licence) with the same attribution.",
              "Do not merge ShareAlike rows into one redistributed single-licence file: JaQuAD (BY-SA 3.0) and HotpotQA (BY-SA 4.0) both embed Wikipedia text and "
              "must stay separately labelled.  See LICENSE_NOTICE.txt."]
    L += ["", "Each row's `source` and `license` fields carry the same information.  Upstream model terms were not checked.", "", "Origin and quality notes:"]
    L += [f"- {x}" for x in info["generator_notes"] + info["quality_notes"]]
    return "\n".join(L) + "\n"


_RUN_INFO_COUNTER = itertools.count(1)


def verify_registry_on_drive(cfg: Cfg, reg: dict) -> int:
    """Re-hash every final file that registry.json lists (after drive_sync: from the fresh mount, not from a local cache)."""
    n = 0
    for k, d in reg["datasets"].items():
        for name, e in d["final"]["files"].items():
            p = Path(d["paths"]["rows"]) / name
            require(p.is_file() and checks.sha256_file(str(p)) == e["sha256"], f"{k}: {p} on Drive does not match the sha256 in registry.json")
            n += 1
    return n


@step("9 final rows and report", flush_on_failure=True)
def step_finalize(cfg: Cfg):
    V = _CACHE.get("vocab")
    info5, info6b, raw, st8 = (load_state(cfg, n) for n in ("step5_acquire", "step6b_vocab_eval", "step6c_raw_checks", "step8_filters"))
    require(all(x is not None for x in (info5, info6b, raw, st8)), "earlier steps have not run (state missing): run cells 5, 6b, 6c and 8 first")
    reviews = load_state(cfg, "step8b_reviews") or {}
    smoke7 = last_sandbox_record(cfg)
    for k in cfg.keys:
        ensure_stage_a(cfg, k)
    ck = _convert_ck(cfg)
    reg_ds: Dict[str, Any] = {}
    final_dir = Path(cfg.work) / f"final_{cfg.code_tag()}"
    apache = find_apache_license_text()
    for k in cfg.keys:
        man = read_manifest(cfg.conv_dir(k))
        dropped = set()
        for line in (cfg.out / "filters" / f"{k}_dropped.tsv").read_text(encoding="utf-8").splitlines()[1:]:
            dropped.add(int(line.split("\t", 1)[0]))
        d = final_dir / k
        d.mkdir(parents=True, exist_ok=True)
        fh = {sp: open(d / f"{sp}.jsonl", "w", encoding="utf-8", newline="\n") for sp in SPLITS}
        n_by = collections.Counter()
        try:
            for i, (sp, line) in enumerate(iter_stage_a(cfg, k)):
                if i in dropped:
                    continue
                fh[sp].write(line if line.endswith("\n") else line + "\n")
                n_by[sp] += 1
        finally:
            for f in fh.values():
                f.close()
        want = st8["datasets"][k]["counts"]["final_rows"]
        require(sum(n_by.values()) == want, f"{k}: {sum(n_by.values())} rows written, the filter step kept {want}")
        files = {}
        for sp in SPLITS:
            p = d / f"{sp}.jsonl"
            files[f"{sp}.jsonl"] = {"rows": n_by[sp], "bytes": p.stat().st_size, "sha256": checks.sha256_file(str(p))}

        def final_rows(sps=SPLITS):
            for sp in sps:
                yield from checks.iter_rows(str(d / f"{sp}.jsonl"))
        s_all, s_train = _final_stats(final_rows()), _final_stats(final_rows(("train",)))
        require(s_all["rows"] == want, f"{k}: re-read {s_all['rows']} rows, expected {want}")
        # every final row is still a valid row
        bad = collections.Counter()
        for r in final_rows():
            why = convert.validate_row(r, ck)
            if why:
                bad[why] += 1
        require(not bad, f"{k}: final rows that fail validate_row: {dict(bad)}")
        verdict = adoption_verdict(cfg, k, want, man, s_all["verification"])
        hold = k in HOLD_KEYS
        rd = cfg.rows / HOLD_DIRNAME / k if hold else cfg.rows / k
        for name in files:
            publish_file(d / name, rd / name)
        reading = cfg.out / "reading" / f"final_{k}_100.txt"
        checks.sample_for_reading(final_rows(), n=100, seed=7, out_path=str(reading), title=f"{k} FINAL rows (after all filters)", vocab=V, progress=cfg.progress)
        a5 = info5[k]
        write_once(rd / "ATTRIBUTION.md", _attribution_md(a5, k, cfg))
        write_once(rd / "LICENSE_NOTICE.txt", license_notice(a5, k, apache is not None))
        if apache is not None and any(i.lower().startswith("apache") for i in a5["license_ids"]):
            write_once(rd / "LICENSE-Apache-2.0.txt", apache)
        if hold:
            write_once(rd / "DO_NOT_TRAIN_UNTIL_APPROVED.txt",
                       f"{k}: adoption is ON HOLD ({verdict['ja']}).  These rows were written so that the checks can be read; do not concatenate this folder into a "
                       f"training set before the owner has approved it.  Every row also says adoption = hold.\n")
        c = st8["datasets"][k]["counts"]
        manifest = {"schema": "v49_rows_manifest/2", "dataset": k, "repo": a5["repo"], "commit_sha": a5["commit_sha"], "license": a5["license_name"],
                    "row_license": a5["row_defaults"]["license"], "row_source": a5["row_defaults"]["source"], "counts": c, "files": files,
                    "supervised_chars": {"train": s_train["supervised_chars"], "all": s_all["supervised_chars"]}, "families": s_all["families"],
                    "english": {"rows": s_all["english_rows"], "supervised_chars": s_all["english_supervised_chars"]},
                    "verification": s_all["verification"], "held": hold,
                    "split_unit": man.get("split_unit"), "group_independent_dev_heldout": man.get("group_independent_dev_heldout"),
                    "adoption_at_acquire": a5["adoption_at_acquire"], "adoption_verdict": verdict, "code_sha256": cfg.code_hashes(),
                    "converter_sha256": cfg.converter_hashes(),
                    "note": "adoption is provisional until the 100 reading rows have been read by eye"}
        write_once(rd / "manifest.json", jdump(manifest))
        # read back what is on Drive
        for name, e in files.items():
            require(checks.sha256_file(str(rd / name)) == e["sha256"], f"{k}: {name} on Drive differs from the file that was written")
        reg_ds[k] = {"key": k, "repo": a5["repo"], "commit_sha": a5["commit_sha"], "retrieved_utc": a5["retrieved_utc"], "held": hold,
                     "license": {"name": a5["license_name"], "ids": a5["license_ids"], "url": a5["license_url"], "share_alike": a5["share_alike"]},
                     "row_license": a5["row_defaults"]["license"], "row_source": a5["row_defaults"]["source"], "language": a5["language"],
                     "attribution": a5["attribution"], "generator_notes": a5["generator_notes"], "quality_notes": a5["quality_notes"],
                     "counts": c, "count_check": {"expected": raw["datasets"][k]["expected"], "counted": raw["datasets"][k].get("n_total", raw["datasets"][k]["n_raw"]),
                                                  "match": raw["datasets"][k]["count_match"], "source": raw["datasets"][k]["count_source"]},
                     "raw": raw["datasets"][k], "convert": {x: man.get(x) for x in ("n_in", "n_out", "keep_rate", "rejects", "families", "splits", "extra", "backends",
                                                                                      "arith_error_rate", "language", "fit", "verification", "python", "question_types",
                                                                                      "supervised_chars", "english", "split_unit", "group_independent_dev_heldout")},
                     "filters": st8["datasets"][k], "supervised_chars": {"train": s_train["supervised_chars"], "all": s_all["supervised_chars"]},
                     "english": {"rows": s_all["english_rows"], "supervised_chars": s_all["english_supervised_chars"]},
                     "final": {"by_split": dict(n_by), "families": s_all["families"], "verification": s_all["verification"], "rows_with_tool_steps": s_all["rows_with_tool_steps"],
                               "files": files}, "adoption": {"at_acquire": a5["adoption_at_acquire"], "verdict": verdict},
                     "paths": {"rows": str(rd), "reading_raw": raw["datasets"][k]["reading_file"], "reading_final": str(reading),
                               "converted_unfiltered": str(cfg.stage_a_drive(k)), "dropped_ids": str(cfg.out / "filters" / f"{k}_dropped.tsv")}}
        cfg.log(f"  {k}: {c['converted']:,} converted -> {want:,} final rows ({dict(n_by)}); verdict {verdict['code']}" + ("  [HOLD folder]" if hold else ""))
    mism = raw["mismatches"]
    missing_optional = info6b["eval"].get("missing_optional") or []
    reg = {"schema": "v49_acquire_registry/2", "pipeline_version": PIPELINE_VERSION, "owner_decision": acquire.OWNER_DECISION, "code_sha256": cfg.code_hashes(),
           "converter_sha256": cfg.converter_hashes(),
           "paths": {"out": str(cfg.out), "rows": str(cfg.rows), "hold_rows": str(cfg.rows / HOLD_DIRNAME), "external_data": str(cfg.ext), "stage_a": str(Path(cfg.root) / STAGE_A_NAME / cfg.converter_tag())},
           "trainable": [k for k in cfg.keys if k not in HOLD_KEYS], "held": [k for k in cfg.keys if k in HOLD_KEYS],
           "params": {"max_chars": cfg.max_chars, "oov_max_frac": convert.OOV_MAX_FRAC, "near_target_jaccard": cfg.near_threshold, "near_estimate_cut": cfg.near_cut,
                      "minhash_k": checks.MinHasher().k, "minhash_bands": checks.MinHashIndex().bands, "ngram": cfg.ngram,
                      "minhash_detection_probability": {str(j): round(p, 3) for j, p in detection_probs(checks.MinHasher().k, cfg.near_cut).items()},
                      "boilerplate_eval_df": cfg.boilerplate_eval_df, "min_shared": cfg.min_shared, "split_pct_groups": convert.SPLIT_PCT,
                      "split_hygiene_threshold_estimate": cfg.split_hygiene_threshold, "python_requires_stored_output": not cfg.python_allow_unverified,
                      "en_row_max": convert.EN_ROW_MAX, "wizard_max_error_rate_shown": cfg.wizard_max_error_rate, "wizard_min_statements": cfg.wizard_min_statements},
           "sandbox": smoke7, "vocab": info6b["vocab"], "eval": info6b["eval"], "old_corpus": st8["global"]["old_corpus"], "decontam": st8["global"]["decontam"],
           "split_hygiene": st8["global"].get("split_hygiene"), "semantic_review": reviews.get("semantic"), "cross_dataset_duplicates": reviews.get("cross_dataset"),
           "count_mismatches": mism, "not_covered": not_covered(missing_optional), "datasets": reg_ds}
    write_once(cfg.out / "registry.json", jdump(reg))
    write_once(cfg.out / "REPORT_JA.md", render_report(reg, cfg))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    try:
        import numpy as _np
        np_ver = _np.__version__
    except Exception:                                                               # noqa: BLE001
        np_ver = None
    write_once(cfg.out / f"run_info_{now}_{os.getpid()}_{next(_RUN_INFO_COUNTER)}.json",
               jdump({"utc": now, "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np_ver,
                      "minhash_backend": checks.MinHasher().backend, "cpu_count": os.cpu_count(), "timings": cfg.timings}))
    drive_sync(cfg, "step 9")
    n_checked = verify_registry_on_drive(cfg, reg)
    tot = sum(v["counts"]["final_rows"] for v in reg_ds.values())
    return (f"{tot:,} rows in {len(reg_ds)} datasets written to {cfg.rows.name}/ ({len(reg['held'])} on hold under {HOLD_DIRNAME}/); {n_checked} files re-hashed "
            f"after a fresh Drive mount; registry.json and REPORT_JA.md written to {cfg.out.name}/; count mismatches: {len(mism)}"), {"registry": reg}


# ====================================================================================================== the Japanese report
def detection_probs(k: int, cut: float, js=(0.9, 0.85, 0.8, 0.75, 0.7)) -> dict:
    """P(estimated Jaccard >= cut) for a pair of TRUE Jaccard j, with a k-permutation MinHash (binomial): how sharp the near-duplicate cut is."""
    kmin = math.ceil(cut * k - 1e-9)
    return {j: sum(math.comb(k, i) * j ** i * (1 - j) ** (k - i) for i in range(kmin, k + 1)) for j in js}


def _n(x) -> str:
    return "-" if x is None else f"{x:,}"


def _pct(x) -> str:
    return "-" if x is None else f"{x:.2%}"


def render_report(reg: dict, cfg: Cfg) -> str:
    ds = reg["datasets"]
    pr = reg["params"]
    L = ["# v49 外部データ 取得・検査レポート（自動生成）", "",
         "このファイルは colab_acquire_check.ipynb が作りました。数字はすべて今回の実行の実測値で、事前に決めた件数はありません。",
         "採用判定は検査結果からの暫定の提案です。各データセット100行の目視確認が済むまで確定しません。", "",
         "## 1. 結果の一覧", "",
         "| データセット | ライセンス | 元の件数 | 変換後 | 完全重複除去後 | 近似重複除去後 | 評価データ除外後 | 旧コーパス重複除去後 | dev/heldoutの近い兄弟除去後（採用候補の行数） | 採用判定 | 教師あり文字数（train） | 備考 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---|"]
    for k in KEYS:
        d = ds.get(k)
        if not d:
            continue
        c = d["counts"]
        note = []
        if d["count_check"]["match"] is False:
            note.append(f"件数がカードと不一致（数えた {_n(d['count_check']['counted'])} / 期待 {_n(d['count_check']['expected'])}）")
        if d["license"]["share_alike"]:
            note.append("継承条件あり")
        if d["held"]:
            note.append(f"**保留**: {HOLD_DIRNAME}/ に置いてあります（学習に混ぜないでください）")
        if d["raw"].get("bad_lines"):
            note.append(f"壊れたJSON行 {_n(d['raw']['bad_lines'])}")
        L.append(f"| {k} | {d['license']['name']} | {_n(c.get('n_raw'))} | {_n(c['converted'])} | {_n(c['after_exact_dedup'])} | {_n(c['after_near_dedup'])} | "
                 f"{_n(c['after_eval_decontam'])} | {_n(c['after_old_corpus_dedup'])} | {_n(c['after_split_hygiene'])} | {d['adoption']['verdict']['ja']} | "
                 f"{_n(d['supervised_chars']['train'])} | {'；'.join(note) or '-'} |")
    oc, dc = reg["old_corpus"], reg["decontam"]
    L += ["", "「元の件数」は取得したファイルを端から端まで数えた件数です。「変換後」は形式・長さ（1,900文字以内）・語彙外文字（1%超は除外）・コードの実行検証などを通った行です。",
          "各段階は前の段階に残った行だけを対象にした順番の絞り込みです。", "",
          "## 2. 全体の確認結果", "",
          f"- 旧コーパス: {_n(oc['rows_found'])} 行を確認（計画の値 {_n(oc['rows_expected'])}）。ファイル `{oc['path']}`。",
          f"- 語彙: {reg['vocab']['n_single_chars']} 文字（{reg['vocab']['source']}）。コーパスからの推定ではなくモデル自身の語彙ファイルです。"
          + ("" if reg["vocab"].get("sha_matches_owner_pc_copy") else "（注意: PC上の語彙とハッシュが一致しません。ALLOW_VOCAB_MISMATCH で許可された実行です）"),
          f"- 評価データ: {len(reg['eval']['sources'])} 件の評価ファイルに加え、{', '.join(reg['eval']['extra_sources']) or 'なし'} を照合に使用。"
          f"13文字の断片（数字と空白を除いたもの）の種類数 {_n(reg['eval']['index']['distinct_shingles'])}。",
          f"- 評価データとの一致で除外した行: {_n(dc['rows_flagged'])} 行（評価問題 {dc['boilerplate_eval_df']} 件以上が共有する定型句だけが一致した {_n(dc['rows_boilerplate_only'])} 行は残しました）。"
          f"データセット別: {dc['flagged_by_dataset']}。一致した断片の数の分布: {dc['shared_count_hist']}（1個だけの一致は、よくある言い回しの偶然の一致のことがあります。例は registry.json の decontam.examples）。"
          f"定型句かどうかは「評価問題側で何件が共有しているか」で決めています（学習行側の数ではありません）。1件の評価問題にしかない文は、何行がそれを写していても除外されます。",
          f"- 旧コーパスとの重複で除外した行: 完全一致 {oc['exact_dropped_by_dataset']}、近い重複（推定Jaccard {oc['estimate_cut']} 以上）{oc['near_dropped_by_dataset']}。"]
    sh = reg.get("split_hygiene") or {}
    if sh.get("datasets"):
        L.append("- dev / heldout の近い兄弟の除去（学習行と推定Jaccard {0} 以上で似ている dev / heldout 行を落とした数）: ".format(sh["threshold_estimate"])
                 + "、".join(f"{k} {_n(v['dropped'])}/{_n(v['dev_heldout_rows'])}" for k, v in sh["datasets"].items()) + "。")
    sem, cd = reg.get("semantic_review"), reg.get("cross_dataset_duplicates")
    if sem and "error" not in sem:
        cut = f"（時間切れ: {_n(sem.get('rows_scanned'))} / {_n(sem.get('rows_total'))} 行まで）" if sem.get("truncated") else ""
        L.append(f"- 意味的な類似の目視用リスト: 評価問題 {_n(sem['queries'])} 問（sha順の標本）のうち {_n(sem['queries_with_a_look_alike'])} 問に似た行があります（`{sem['file']}`）。行は除外していません。{cut}")
    elif sem:
        L.append(f"- 意味的な類似の確認は失敗しました（{sem['error']}）。")
    else:
        L.append("- 意味的な類似の確認（セル8b）はまだ実行されていません。")
    if cd and "error" not in cd:
        L.append(f"- データセット間の近似重複（推定Jaccard {cd['estimate_cut']} 以上）: {cd['rows_with_match_in']}" + ("（時間切れで途中まで）" if cd.get("truncated") else ""))
    elif cd:
        L.append(f"- データセット間の近似重複の確認は失敗しました（{cd['error']}）。")
    L += ["", "### T1〜T8 の評価データの照合状況", "", "| 区分 | ファイル（13文字の断片の数） |", "|---|---|"]
    for t in reg["eval"].get("tiers", []):
        L.append(f"| {t['tier']} | " + "、".join(f"{n}（{_n(v['shingles'])}）" for n, v in t["files"].items()) + " |")
    L += ["", "## 3. 件数の不一致（MISMATCH）", ""]
    if reg["count_mismatches"]:
        for m in reg["count_mismatches"]:
            L.append(f"- **MISMATCH** {m['dataset']}: 数えた件数 {_n(m['counted'])} に対し、期待 {_n(m['expected'])}（{m['source']}）。カードや計画の数字が実際と違う可能性があります。")
    else:
        L.append("- なし（6つともカード・計画の件数と一致）。")
    L += ["", "## 4. データセットごとの詳細", ""]
    for k in KEYS:
        d = ds.get(k)
        if not d:
            continue
        c, cv, rw = d["counts"], d["convert"], d["raw"]
        L += [f"### {k}（{d['repo']}）", "",
              f"- 取得したコミット: `{d['commit_sha']}`（取得日時 {d['retrieved_utc']}）。言語: {d['language']}。",
              f"- 採用判定: {d['adoption']['verdict']['ja']}",
              f"- 変換での主な除外理由: " + (", ".join(f"{r}={v:,}" for r, v in list((cv.get('rejects') or {}).items())[:6]) or "なし"),
              f"- 元データの語彙外文字の割合: {_pct(rw['oov']['oov_frac'])}（1%超の行 {_pct(rw['oov']['share_rows_over_threshold'])}）。2,048文字超の行: {_pct(rw['length']['share_over_2048'])}。",
              f"- 最終行数 {_n(c['final_rows'])}（分割 {d['final']['by_split']}）。ファミリー別: {d['final']['families']}。教師あり文字数 train {_n(d['supervised_chars']['train'])} / 全体 {_n(d['supervised_chars']['all'])}。",
              f"- 英語の行: 最終 {_n(d['english']['rows'])} 行（教師あり {_n(d['english']['supervised_chars'])} 文字）。長い英語の行は使わず、{convert.EN_ROW_MAX} 文字までの短いものだけを残しています（HotpotQA は英語の段落を日本語の枠に入れた形）。"
              f"変換直後のファミリー別: 英語 {cv.get('english')}、教師あり文字数 {cv.get('supervised_chars')}。",
              f"- 分割の単位: {cv.get('split_unit')}。dev / heldout が学習行から独立しているか: " + ("はい（グループ単位）" if cv.get("group_independent_dev_heldout") else "**いいえ**（行単位。dev / heldout をグループ独立の評価として使わないでください）") + "。",
              f"- 絞り込みで落ちた行の内訳: {d['filters'].get('dropped_by_reason')}。"]
        near = d["filters"].get("near")
        if near:
            L.append(f"- 近似重複: MinHash {near['k']}置換、推定Jaccard {near['estimate_cut']} 以上を同じ群として最初の1行だけ残す（目標は真のJaccard {near['target_jaccard']}）。"
                     f"群が2行以上のもの {_n(near['clusters'])} 群、最大 {_n(near['largest_cluster'])} 行、群の大きさの分布 {near['cluster_size_hist']}。"
                     "鎖状につながった群では、残す行と直接は似ていない行も落ちることがあります。")
        if d["final"]["verification"]:
            L.append(f"- 検証の内訳（最終行）: {d['final']['verification']}。")
        py = cv.get("python")
        if py:
            L.append(f"- python行: 実行して保存出力と一致した（検証済みの）行は {_n(py['rows_kept_stored_output_equals_sandbox'])} 行だけです。実行はできたが比べる保存出力が無かった行 "
                     f"{_n(py['ran_in_sandbox_but_had_no_stored_output'])} 行は「検証済み」として残さず、数だけ記録しました。")
        if cv.get("question_types"):
            L.append(f"- JaQuAD の質問タイプ（変換に届いた行）: {cv['question_types']}。「Multiple sentence reasoning」と「Ambiguous」は証拠が1文に収まらない・答えが定まらないので除外しています。")
        if cv.get("arith_error_rate"):
            a = cv["arith_error_rate"]
            L.append(f"- 算術式の検算: {_n(a['statements_checked'])} 式中 {_n(a['statements_wrong'])} 式が誤り（{_pct(a['statement_error_rate'])}）。行単位では {_pct(a['row_error_rate'])}。"
                     "検算できたのは単純な「a 演算 b = c」だけなので、実際の誤り率の下限です。")
        L += [f"- 目視用: 元データ100行 `{d['paths']['reading_raw']}`、最終行100行 `{d['paths']['reading_final']}`。",
              f"- 出力: `{d['paths']['rows']}`（train / dev / heldout の jsonl、manifest.json、ATTRIBUTION.md、LICENSE_NOTICE.txt）。", ""]
    L += ["## 5. 検査の範囲と限界", ""] + [f"- {x}" for x in reg["not_covered"]]
    probs = detection_probs(pr["minhash_k"], pr["near_estimate_cut"])
    L += ["- 重複の判定: 同一データセット内は完全一致→MinHash（{k}置換・5文字断片。推定Jaccard {cut} 以上を重複とみなす。目標の真のJaccard {t} に対し、見つかる確率は計算上 ".format(
        k=pr["minhash_k"], cut=pr["near_estimate_cut"], t=pr["near_target_jaccard"]) + "、".join(f"真のJaccard {j} の組で約{p:.0%}" for j, p in probs.items())
          + "）。JaQuAD と HotpotQA は1つの文章に複数の質問が付くため、質問＋根拠＋答えの部分で比べています。旧コーパスとは行全体で比べています。",
          "- 評価データとの照合は数字と空白を除いた13文字の完全一致です。言い換えや翻訳は検出できません。",
          "- 件数は実行ごとに決まります。同じコードと同じデータなら同じ結果になります（乱数は固定）。", ""]
    L += ["## 6. ライセンスと表示の義務", ""]
    for k in KEYS:
        d = ds.get(k)
        if d:
            L.append(f"- {k}: {d['license']['name']}" + ("。継承条件（ShareAlike）あり: 変換した行も同じライセンス系統で、出典を付けて共有する必要があります。"
                                                          "JaQuAD（BY-SA 3.0）と HotpotQA（BY-SA 4.0）はどちらも Wikipedia の文章を含むので、1つの単一ライセンスのファイルに混ぜて再配布しないでください。"
                                                          if d["license"]["share_alike"] else "。ライセンス表示と変更点の明記が必要です（ライセンス本文は LICENSE-Apache-2.0.txt）。")
                     + f" 出典の文言は `{d['paths']['rows']}/ATTRIBUTION.md` と `LICENSE_NOTICE.txt` にあります。")
    L += ["- 生成モデル（Nemotron-4、Phi-3、Mixtral、calm3、Qwen2.5、WizardLM-2）の利用条件は確認していません。", ""]
    L += ["## 7. 出力の場所", "", f"- 行データ（採用候補）: `{reg['paths']['rows']}/<データセット>/`", f"- 保留のデータセット（{', '.join(reg['held']) or 'なし'}）: `{reg['paths']['hold_rows']}/<データセット>/`（学習に混ぜない）",
          f"- 絞り込み前の変換結果（再実行ではここから戻す）: `{reg['paths']['stage_a']}/`", f"- 目視ファイル・状態・この報告: `{reg['paths']['out']}/`",
          f"- 取得した元ファイルと出所の記録: `{reg['paths']['external_data']}/`", ""]
    return "\n".join(L)
