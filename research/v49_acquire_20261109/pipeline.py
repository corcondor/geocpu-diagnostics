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
Colab local disk (cfg.work, default /content/v49_work), which is scratch.

Flow (the numbers are the notebook's cells):
  4 dry run -> 5 acquire -> 6a local mirror of the train files -> 6b vocabulary + eval index -> 6c raw checks (counts vs the cards,
  OOV, length, 100-row reading dumps) -> 7a converters on the first 2,000 records -> 7b converters on everything (resumable: the
  unfiltered rows are kept on Drive) -> 8 filters (exact + MinHash near duplicates, 13-gram decontamination, old corpus) ->
  9 final rows, registry.json, REPORT_JA.md.
"""
from __future__ import annotations

import bisect
import collections
import dataclasses
import datetime
import functools
import hashlib
import io
import json
import os
import platform
import shutil
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

PIPELINE_VERSION = "1"
OUT_NAME = "v49_acquire_20261109"
ROWS_NAME = "external_data_v49_rows"
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
R_EXACT, R_NEAR, R_EVAL, R_OLD_EXACT, R_OLD_NEAR = 1, 2, 3, 4, 5
REASON = {R_EXACT: "exact_duplicate_within_dataset", R_NEAR: "near_duplicate_within_dataset", R_EVAL: "eval_overlap_13gram",
          R_OLD_EXACT: "exact_duplicate_of_old_corpus", R_OLD_NEAR: "near_duplicate_of_old_corpus"}
STAGE_A_FILES = ("train.jsonl", "dev.jsonl", "heldout.jsonl", "manifest.json", "sample_dump.txt", "rejects_sample.jsonl")
NOT_COVERED = ["GSM8K / MATH test sets are not on Drive and were not downloaded (the owner approved exactly six datasets): "
               "they are NOT part of the decontamination.  Put them as .jsonl/.json under <Drive root>/" + USER_EVAL_EXTRA + "/ and re-run "
               "with a new RUN_SUFFIX to include them.",
               "Semantic (paraphrase) overlap is only looked at, not removed: the character-bigram look-alike search is lexical and not exhaustive.",
               "HotpotQA validation is indexed by its questions only (its passages are Wikipedia paragraphs shared with train by construction)."]


class StepFailed(AssertionError):
    pass


def require(cond, msg: str) -> None:
    if not cond:
        raise StepFailed(msg)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n"


# ====================================================================================================== configuration
@dataclasses.dataclass
class Cfg:
    root: str = acquire.DEFAULT_ROOT
    suffix: str = ""                                    # RUN_SUFFIX: a fresh output folder after a refusal (never overwrite)
    work: str = "/content/v49_work"                     # local scratch (NOT Drive)
    only: Optional[tuple] = None                        # convert.py keys; None = all six
    smoke_limit: int = 2000
    stop_on_count_mismatch: bool = False
    expected_counts: Optional[dict] = dataclasses.field(default_factory=lambda: dict(EXPECTED_COUNTS))
    expected_old_rows: Optional[int] = checks.OLD_CORPUS_ROWS
    old_corpus: Optional[str] = None
    max_chars: int = convert.DEFAULT_CAP
    near_threshold: float = 0.8
    ngram: int = 13
    boilerplate_df: int = 500
    min_shared: int = 1
    wizard_max_error_rate: float = 0.01                 # my default for "the error rate is small"; the owner decides
    wizard_min_statements: int = 300
    run_sandbox: bool = True
    parallel: int = 2
    poll_s: float = 15.0
    heartbeat_s: float = 120.0
    vocab_override: Optional[str] = None                # tests only: a base_metadata.json on the local disk
    eval_extra_dir: Optional[str] = None                # the notebook writes the embedded v48 unseen_eval texts here
    require_v48: bool = True
    run_semantic: bool = True
    semantic_max_queries: int = 10000
    semantic_min_cos: float = 0.5
    run_cross: bool = True
    check_disk: bool = True
    allow_local: bool = False                           # tests only (acquire refuses real downloads off Colab)
    stage_dir: Optional[str] = None
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

    # --- which datasets
    @property
    def keys(self) -> tuple:
        return KEYS if self.only is None else tuple(k for k in KEYS if k in self.only)

    def acq_only(self):
        return None if self.only is None else [ACQ_KEY[k] for k in self.keys]

    # --- places
    @property
    def ext(self) -> Path:
        return Path(self.root) / acquire.EXTERNAL_DIRNAME

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

    def conv_dir(self, key: str, smoke: bool = False) -> Path:
        return Path(self.work) / (("smoke_" if smoke else "converted_") + self.code_tag()) / key

    def stage_a_drive(self, key: str) -> Path:
        return self.out / "converted_unfiltered" / key

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
def step(name: str):
    """Decorator: prints the header and one PASS/FAIL line; fn(cfg, ...) returns (detail, result dict)."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(cfg, *a, **kw):
            t0 = time.time()
            cfg.log(f"=== {name} ===")
            try:
                detail, res = fn(cfg, *a, **kw)
            except StepFailed as e:
                cfg.log(f"FAIL [{name}] {e}")
                raise
            except Exception as e:                                                  # noqa: BLE001
                tb = traceback.format_exc().strip().splitlines()[-6:]
                cfg.log(f"FAIL [{name}] {type(e).__name__}: {e}\n    " + "\n    ".join(tb))
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
    """Create a text file on Drive; identical content is fine, different content is refused."""
    try:
        return checks.write_new_file(str(path), text)
    except FileExistsError as e:
        raise StepFailed(f"{e}.  Nothing was overwritten.  To produce new output, set RUN_SUFFIX (e.g. '_2') in the notebook and run again.") from e


def publish_file(src, dst) -> str:
    """Copy a local file to Drive: never over an existing file; verified by sha256; a stale .part is moved aside, not deleted."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = checks.sha256_file(str(src))
    if dst.exists():
        if checks.sha256_file(str(dst)) == want:
            return "identical"
        raise StepFailed(f"refusing to overwrite {dst}: it exists with a different sha256.  To produce new output, set RUN_SUFFIX (e.g. '_2') "
                         f"and run again.")
    part = Path(str(dst) + ".part")
    if os.path.lexists(part):
        acquire._quarantine(str(part))
    n, sha = acquire._copy_hashing(str(src), str(part))
    require(sha == want and n == src.stat().st_size, f"copy of {src.name} to Drive does not match the source (kept as {part})")
    os.rename(part, dst)
    return "created"


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
    h = cfg.code_hashes()
    return "modules import; table valid; " + ", ".join(f"{k} {v[:8]}" for k, v in sorted(h.items())), {"code_sha256": h}


# ====================================================================================================== 4: dry run
@step("4 dry run")
def step_dry_run(cfg: Cfg):
    plan = acquire.acquire(cfg.root, cfg.acq_only(), dry_run=True, log=cfg.log)
    sel = acquire.select(list(acquire.DATASETS), cfg.acq_only())
    expect = acquire.total_bytes(sel)
    require(plan["total_bytes"] == expect and plan["n_files"] == sum(len(d["files"]) for d in sel),
            f"plan says {plan['total_bytes']:,} bytes / {plan['n_files']} files, the table says {expect:,}")
    biggest = max(f["size"] for d in sel for f in d["files"])
    train = acquire.total_bytes(sel, ("train",))
    notes = []
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
        need_drive = expect + (3 << 29)
        if free_drive:
            require(free_drive >= need_drive, f"Drive reports {free_drive:,} B free, need about {need_drive:,} B")
            notes.append(f"Drive free {free_drive / 2**30:.1f} GiB")
        else:
            notes.append("Drive free space unknown")
    return (f"{plan['n_files']} files, expected total {expect:,} bytes = {expect / 1e9:.2f} GB "
            f"(train {train:,}; dev/validation {plan['do_not_train_bytes']:,})" + "".join("; " + x for x in notes)), {"expected_total_bytes": expect, "n_files": plan["n_files"]}


# ====================================================================================================== 5: acquire
def verify_acquired(cfg: Cfg) -> dict:
    """After acquire.acquire(): every selected dataset has provenance, the licence the plan expects, and all its files on Drive."""
    man = acquire.load_manifest(cfg.root)
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
        require(acquire.train_files(cfg.root, ds["key"]), f"{ds['repo']}: no train files")
        out[key] = {"repo": ds["repo"], "commit_sha": prov["commit_sha"], "retrieved_utc": prov["retrieved_utc"],
                    "license_name": prov["license"]["name"], "license_ids": prov["license"]["ids_from_card"],
                    "license_url": prov["license"]["url"], "share_alike": prov["license"]["share_alike"],
                    "attribution": prov["attribution"]["lines"], "row_defaults": prov["row_defaults"],
                    "adoption_at_acquire": prov["adoption"], "language": prov["language"], "rows_per_card": prov["rows_per_card"],
                    "generator_notes": prov["generator_notes"], "quality_notes": prov["quality_notes"],
                    "files": files, "train_bytes": sum(f["size"] for f in files if f["role"] == "train"),
                    "do_not_train_bytes": sum(f["size"] for f in files if f["role"] in acquire.NO_TRAIN_ROLES)}
    return out


@step("5 acquire")
def step_acquire(cfg: Cfg):
    res = acquire.acquire(cfg.root, cfg.acq_only(), log=cfg.log, allow_local=cfg.allow_local, stage_dir=cfg.stage_dir)
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
    man = acquire.load_manifest(cfg.root)
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
        for src in acquire.train_files(cfg.root, ds["key"]):
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
    texts, the dev/validation files that acquire.py keeps under do_not_train_*, and anything the owner drops into v49_eval_extra/."""
    sources = list(ef.sources)
    dnt = acquire.do_not_train_files(cfg.root)
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
        notes.append("vocabulary sha256 MISMATCH vs the owner's PC copy")
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
    try:
        ef = checks.find_eval_files(cfg.root, extra_dirs=[cfg.eval_extra_dir] if cfg.eval_extra_dir else [], strict=True)
    except checks.EvalFilesMissing as e:
        raise StepFailed(f"required eval files are missing, so decontamination would be incomplete: {e}")
    cfg.log(ef.report())
    labels = {s.label for s in ef.sources}
    if cfg.require_v48:
        missing = [n for n in V48_EVAL_NAMES if "v48:" + n not in labels]
        require(not missing, f"v48 unseen_eval texts not found: {missing}")
    sources = decontam_sources(cfg, ef)
    index = checks.build_eval_index(sources, ngram=cfg.ngram, progress=lambda label, n: cfg.log(f"  {label}: {n:,} shingles so far"))
    st = index.stats()
    required = {s.label for s in ef.sources if s.required}
    empty = [lab for lab in required if st["per_source"].get(lab, {}).get("items", 0) == 0]
    require(not empty, f"eval sources without any item: {empty}")
    require(st["distinct_shingles"] > 0, "the eval index is empty")
    for key in cfg.keys:
        want = {"jaquad": "jaquad_dev", "hotpot": "hotpot_validation"}.get(key)
        if want:
            require(st["per_source"].get(want, {}).get("items", 0) > 0, f"{want} was not indexed (dev/validation files for decontamination)")
    _CACHE["eval_index"], _CACHE["eval_sources"] = index, sources
    vd = V.as_dict()
    vd["source"] = vd["source"][len("cache:"):] if vd["source"].startswith("cache:") else vd["source"]          # same text on a re-run that reads the cache
    info = {"vocab": dict(vd, sha_matches_owner_pc_copy=vocab_sha_match),
            "eval": {"sources": [{"label": s.label, "where": s.display, "required": s.required} for s in ef.sources],
                     "extra_sources": [s.label if isinstance(s, checks.EvalSource) else s[0] for s in sources[len(ef.sources):]],
                     "missing_optional": ef.missing_optional, "notes": ef.notes, "index": {"ngram": st["ngram"], "distinct_shingles": st["distinct_shingles"],
                                                                                         "per_source": st["per_source"]}},
            "notes": notes}
    save_state(cfg, "step6b_vocab_eval", info)
    return (f"vocabulary {len(V.chars)} chars from {V.source}; {len(ef.sources)} eval sources + {len(sources) - len(ef.sources)} dev/validation/extra sources, "
            f"{st['distinct_shingles']:,} distinct {cfg.ngram}-gram shingles; " + "; ".join(notes)), info


# ====================================================================================================== 6c: raw checks
def _squad_text(r):
    return "\n".join([str(r.get("title", "")), str(r.get("context", "")), str(r.get("question", "")), " ".join(str(x) for x in r.get("answer_texts") or [])])


def _tengentoppa_text(r):
    return str(r.get("instruction", "")) + "\n" + str(r.get("refined_answer", ""))


def _toolace_text(r):
    return str(r.get("system", "")) + "\n" + "\n".join(str(m.get("value", "")) for m in (r.get("conversations") or []) if isinstance(m, dict))


RAW_TEXT = {"jaquad": _squad_text, "tengentoppa": _tengentoppa_text, "toolace": _toolace_text, "coding": checks.default_text,
            "wizard": checks.default_text, "hotpot": checks.default_text}


def raw_rows(cfg: Cfg, key: str):
    for p in mirror_train_files(cfg, key):
        yield from checks.iter_rows(str(p))


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

        def rows():
            for r in raw_rows(cfg, key):
                if key == "jaquad":
                    prefixes[str(r.get("id"))[:3]] += 1
                yield r
        t0 = time.time()
        oov = checks.oov_stats(rows(), V, text_fn=tf, progress=cfg.progress)
        ln = checks.length_stats(raw_rows(cfg, key), text_fn=tf, progress=cfg.progress)["ALL"]
        n = oov["n_rows"]
        require(n > 0 and ln["n"] == n, f"{key}: no records or the two passes disagree ({n} vs {ln['n']})")
        dump = cfg.out / "reading" / f"raw_{key}_100.txt"
        checks.sample_for_reading(raw_rows(cfg, key), n=100, seed=7, out_path=str(dump), text_fn=tf,
                                  title=f"{key} RAW records (before conversion)", vocab=V, progress=cfg.progress)
        exp = (cfg.expected_counts or {}).get(key)
        match = None if exp is None else (n == exp)
        if match is False:
            mismatch_banner(cfg, key, n, exp)
            mism.append({"dataset": key, "counted": n, "expected": exp, "source": COUNT_SOURCE[key]})
        if key == "jaquad":
            bad = {p: c for p, c in prefixes.items() if p in ("de-", "te-")}
            require(not bad, f"jaquad train files contain dev/test ids: {bad}")
        out[key] = {"n_raw": n, "expected": exp, "count_match": match, "count_source": COUNT_SOURCE[key],
                    "oov": {k: oov[k] for k in ("n_chars", "oov_frac", "rows_with_any_oov", "rows_over_threshold", "share_rows_over_threshold", "n_distinct_oov_chars")},
                    "top_oov": oov["top_oov"][:8],
                    "length": {k: ln[k] for k in ("mean", "p50", "p90", "p99", "max", "over_2048", "share_over_2048", "over_1900", "share_over_1900")},
                    "id_prefixes": dict(prefixes) if key == "jaquad" else None, "reading_file": str(dump)}
        cfg.log(f"  {key}: {n:,} records, OOV chars {oov['oov_frac']:.4%}, rows with >1% OOV {oov['share_rows_over_threshold']:.2%}, "
                f"rows over 2,048 chars {ln['share_over_2048']:.2%}  ({time.time() - t0:.0f} s)")
    if mism and cfg.stop_on_count_mismatch:
        raise StepFailed("record counts differ from the cards for: " + ", ".join(m["dataset"] for m in mism) + " (STOP_ON_COUNT_MISMATCH is on)")
    save_state(cfg, "step6c_raw_checks", {"datasets": out, "mismatches": mism})
    tail = "all counts match the cards" if not mism else f"MISMATCH in {len(mism)}: " + ", ".join(f"{m['dataset']} {m['counted']:,} vs {m['expected']:,}" for m in mism)
    return f"{len(out)} datasets read end to end (parse errors would have stopped here); {tail}", {"datasets": out, "mismatches": mism}


# ====================================================================================================== 7: converters
def _run_converters(cfg: Cfg, keys, limit: Optional[int], smoke: bool) -> dict:
    """convert.py as one subprocess per dataset (at most cfg.parallel at a time); heartbeat while they run."""
    logdir = Path(cfg.work) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    pending = [k for k in CONVERT_ORDER if k in keys]
    running, rcs, t0, last_beat = [], {}, time.time(), time.time()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONHASHSEED="0")
    while pending or running:
        while pending and len(running) < max(1, cfg.parallel):
            key = pending.pop(0)
            out_root = cfg.conv_dir(key, smoke).parent
            out_root.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "-X", "utf8", str(Path(convert.__file__)), "--dataset", key, "--raw-dir", str(cfg.mirror_dir(key)),
                   "--out-dir", str(out_root), "--vocab", str(cfg.vocab_file), "--max-chars", str(cfg.max_chars), "--sample-dump", "100"]
            if limit:
                cmd += ["--limit", str(limit)]
            if not cfg.run_sandbox:
                cmd += ["--no-sandbox"]
            lp = logdir / f"convert_{'smoke' if smoke else 'full'}_{cfg.code_tag()}_{key}.log"
            fh = open(lp, "w", encoding="utf-8")
            running.append({"key": key, "proc": subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=str(cfg.work)),
                            "fh": fh, "log": lp, "t0": time.time(), "out": out_root / key})
            cfg.log(f"  started convert.py --dataset {key}" + (f" --limit {limit}" if limit else "") + f"  (log {lp.name})")
        time.sleep(cfg.poll_s)
        for job in list(running):
            rc = job["proc"].poll()
            if rc is not None:
                job["fh"].close()
                rcs[job["key"]] = {"rc": rc, "seconds": round(time.time() - job["t0"], 1), "log": str(job["log"])}
                running.remove(job)
                cfg.log(f"  finished {job['key']}: rc={rc} after {rcs[job['key']]['seconds']} s")
        if running and time.time() - last_beat >= cfg.heartbeat_s:
            last_beat = time.time()
            for job in running:
                done = sum(p.stat().st_size for p in job["out"].glob("*.part-*")) if job["out"].is_dir() else 0
                cfg.log(f"  [{(time.time() - t0) / 60:5.1f} min] {job['key']} still running, {done / 1e6:.1f} MB of rows written so far")
    bad = {k: v for k, v in rcs.items() if v["rc"] != 0}
    if bad:
        tails = []
        for k, v in bad.items():
            tails.append(f"--- {k} (rc {v['rc']}) log tail ---\n" + "\n".join(Path(v["log"]).read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
        raise StepFailed("converter(s) failed: " + ", ".join(f"{k} rc={v['rc']}" for k, v in bad.items()) + "\n" + "\n".join(tails))
    return rcs


def read_manifest(d: Path) -> dict:
    p = Path(d) / "manifest.json"
    require(p.is_file(), f"{p} is missing")
    return json.loads(p.read_text(encoding="utf-8"))


def _check_manifest(cfg: Cfg, key: str, d: Path, limit: Optional[int], n_raw: Optional[int]) -> dict:
    man = read_manifest(d)
    require(man.get("balanced") is True, f"{key}: converter bookkeeping is not balanced (n_in != n_out + rejects)")
    require(man["n_out"] > 0, f"{key}: the converter emitted no row from {man['n_in']} records")
    if limit:
        want = limit if n_raw is None else min(limit, n_raw)
    else:
        want = n_raw
    if want is not None:
        require(man["n_in"] == want, f"{key}: converter read {man['n_in']:,} records, the raw check counted {want:,}")
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
        if g is not None:
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


@step("7a convert smoke")
def step_convert_smoke(cfg: Cfg):
    require(cfg.vocab_file.is_file(), "run step 6b first (vocabulary file)")
    raw = load_state(cfg, "step6c_raw_checks")
    t0 = time.time()
    rcs = _run_converters(cfg, cfg.keys, cfg.smoke_limit, smoke=True)
    ck = _convert_ck(cfg)
    res, est = {}, {}
    for key in cfg.keys:
        d = cfg.conv_dir(key, smoke=True)
        n_raw = ((raw or {}).get("datasets", {}).get(key) or {}).get("n_raw")
        man = _check_manifest(cfg, key, d, cfg.smoke_limit, n_raw)
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
        cfg.log(f"  {key}: {man['n_in']:,} in -> {man['n_out']:,} rows ({man['keep_rate']:.1%}), {rcs[key]['seconds']:.0f} s, sandbox: {(man.get('backends') or {}).get('sandbox')}")
    if est:
        cfg.log("  rough full-run estimate (minutes, from this smoke run; Colab may differ): " + ", ".join(f"{k} ~{v:g}" for k, v in est.items()))
    return (f"{len(res)} converters ran on the first {cfg.smoke_limit:,} records each; every row passed validate_row again after reading it back "
            f"({time.time() - t0:.0f} s)"), {"datasets": res, "estimate_minutes": est}


def restore_stage_a(cfg: Cfg, key: str) -> bool:
    """Copy the unfiltered rows of an earlier run from Drive to the local disk (resume).  False if there is none."""
    src = cfg.stage_a_drive(key)
    meta_p = src / "stageA_meta.json"
    if not meta_p.is_file():
        return False
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    require(meta.get("code") == cfg.code_hashes(),
            f"{src} was made by different code than the code in this notebook.  Nothing was overwritten.  Set RUN_SUFFIX (e.g. '_2') and run again.")
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
    meta = {"schema": "v49_stage_a_meta/1", "code": cfg.code_hashes(), "dataset_commit_sha": (read_manifest(src).get("provenance") or {}).get("commit_sha"),
            "vocab_sha256_chars": V.sha256_chars, "limit": None}
    write_once(dst / "stageA_meta.json", jdump(meta))


def ensure_stage_a(cfg: Cfg, key: str) -> None:
    if (cfg.conv_dir(key) / "manifest.json").is_file():
        return
    require(restore_stage_a(cfg, key), f"{key}: no converted rows locally or on Drive - run step 7b first")


@step("7b convert full")
def step_convert_full(cfg: Cfg):
    V = _CACHE.get("vocab")
    require(V is not None and cfg.vocab_file.is_file(), "run step 6b first (vocabulary)")
    raw = load_state(cfg, "step6c_raw_checks")
    todo, resumed = [], []
    for key in cfg.keys:
        if (cfg.conv_dir(key) / "manifest.json").is_file() or restore_stage_a(cfg, key):      # local rows or the Drive copy of an earlier run
            resumed.append(key)
        else:
            todo.append(key)
    if resumed:
        cfg.log("  resumed from earlier converted rows: " + ", ".join(resumed))
    if todo:
        _run_converters(cfg, todo, None, smoke=False)
    ck = _convert_ck(cfg)
    res = {}
    for key in cfg.keys:
        d = cfg.conv_dir(key)
        n_raw = ((raw or {}).get("datasets", {}).get(key) or {}).get("n_raw")
        man = _check_manifest(cfg, key, d, None, n_raw)
        require(not man.get("limit"), f"{key}: these rows were made with --limit {man.get('limit')}, not a full run")
        vr = validate_rows(cfg, key, ck)
        require(vr["rows"] == man["n_out"], f"{key}: {vr['rows']} rows read back, manifest says {man['n_out']}")
        cc = _converted_checks(cfg, key, V)
        publish_stage_a(cfg, key, V)                                                        # identical files are skipped, nothing is overwritten
        res[key] = {"manifest": man, "validated": vr, "checks": cc}
        cfg.log(f"  {key}: {man['n_in']:,} records -> {man['n_out']:,} rows; splits {man['splits']}; top rejects "
                + ", ".join(f"{k}={v:,}" for k, v in list(man["rejects"].items())[:3]))
    return (f"{len(res)} datasets converted ({len(todo)} now, {len(resumed)} resumed); rows re-validated, no id/group/split problem, "
            f"no row over the cap or over 1% OOV; unfiltered rows are kept on Drive under {OUT_NAME + cfg.suffix}/converted_unfiltered/"), {"datasets": res}


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


@step("8 filters")
def step_filters(cfg: Cfg):
    keys = cfg.keys
    for k in keys:
        ensure_stage_a(cfg, k)
    index = _CACHE.get("eval_index")
    sources = _CACHE.get("eval_sources")
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
    # ---- 1. exact duplicates and 2. MinHash near duplicates, inside one dataset (keep first)
    for k in keys:
        comb = Combined(cfg, [k], alive)
        ex = checks.exact_dups(comb.rows(), text_fn=dedupe_text, progress=cfg.progress)
        _drop(alive, reason, comb, ex["within_dup_positions"], R_EXACT)
        stats[k]["counts"]["after_exact_dedup"] = sum(alive[k])
        stats[k]["examples"]["exact"] = _examples(ex["examples"])
        comb = Combined(cfg, [k], alive)
        nd = checks.near_dups_within(comb.rows(), text_fn=dedupe_text, threshold=cfg.near_threshold, progress=cfg.progress)
        _drop(alive, reason, comb, nd["dup_positions"], R_NEAR)
        stats[k]["counts"]["after_near_dedup"] = sum(alive[k])
        stats[k]["examples"]["near"] = _examples(nd["examples"])
        stats[k]["near"] = {"clusters": nd["n_clusters_gt1"], "largest_cluster": nd["largest_cluster"], "backend": nd["hasher_backend"]}
        cfg.log(f"  {k}: converted {stats[k]['counts']['converted']:,} -> exact {stats[k]['counts']['after_exact_dedup']:,} -> near {stats[k]['counts']['after_near_dedup']:,}")
    # ---- 3. decontamination against the eval texts: one pass over all datasets
    comb = Combined(cfg, keys, alive)
    dc = checks.decontaminate(comb.rows(), index, text_fn=full_text, boilerplate_df=cfg.boilerplate_df, min_shared=cfg.min_shared,
                              max_examples=10, progress=cfg.progress)
    per = _drop(alive, reason, comb, dc["flagged_positions"], R_EVAL)
    for k in keys:
        stats[k]["counts"]["after_eval_decontam"] = sum(alive[k])
        stats[k]["examples"]["eval_overlap"] = []
    glob: Dict[str, Any] = {"decontam": {"n_rows": dc["n_rows"], "rows_flagged": dc["rows_flagged"], "rows_boilerplate_only": dc["rows_boilerplate_only"],
                                         "rows_with_any_shared_shingle": dc["rows_with_any_shared_shingle"], "shared_count_hist": dc["shared_count_hist"],
                                         "boilerplate": dc["boilerplate"], "examples": dc["examples"], "flagged_by_dataset": per,
                                         "ngram": cfg.ngram, "boilerplate_df": cfg.boilerplate_df, "min_shared": cfg.min_shared,
                                         "limits": dc["limits"]}}
    cfg.log(f"  decontamination: {dc['rows_flagged']:,} of {dc['n_rows']:,} rows share a {cfg.ngram}-gram with an eval text and are dropped "
            f"({dc['rows_boilerplate_only']:,} rows shared only boilerplate and are kept); by dataset {per}")
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
                                threshold=cfg.near_threshold, progress=cfg.progress)
    per_nr = _drop(alive, reason, comb, cr["matched_index_positions"], R_OLD_NEAR)
    glob["old_corpus"] = {"path": str(oc), "rows_found": n_old, "rows_expected": cfg.expected_old_rows, "bytes": oc.stat().st_size,
                          "exact_dropped_by_dataset": per_ex, "near_dropped_by_dataset": per_nr, "probe_rows_scanned": cr["probe_rows_scanned"],
                          "probe_rows_with_match": cr["probe_rows_with_match"], "examples": cr["examples"], "threshold": cfg.near_threshold}
    for k in keys:
        stats[k]["counts"]["after_old_corpus_dedup"] = sum(alive[k])
        stats[k]["counts"]["final_rows"] = sum(alive[k])
    cfg.log(f"  old corpus dedupe: exact {per_ex}, near {per_nr}")
    # ---- 5. optional reviews on the survivors (never remove rows; failures are recorded loudly, not hidden)
    final = Combined(cfg, keys, alive)
    glob["semantic"], glob["cross_dataset"] = None, None
    if cfg.run_semantic:
        try:
            qs = _eval_queries(cfg, sources or [])
            res = checks.semantic_neighbors(qs, final.rows(), text_fn=full_text, id_fn=lambda r, p: r.get("id"), top_n=3,
                                            min_cos=cfg.semantic_min_cos, progress=cfg.progress) if qs else []
            hit = [r for r in res if r["matches"]]
            hit.sort(key=lambda r: -r["matches"][0]["cosine"])
            write_once(cfg.out / "filters" / "semantic_review.txt", checks.format_semantic_report(hit[:300]))
            best = [r["matches"][0]["cosine"] for r in hit]
            glob["semantic"] = {"queries": len(qs), "queries_with_a_look_alike": len(hit), "best_cosine_ge_0.7": sum(1 for c in best if c >= 0.7),
                                "best_cosine_ge_0.9": sum(1 for c in best if c >= 0.9), "min_cos": cfg.semantic_min_cos,
                                "file": str(cfg.out / "filters" / "semantic_review.txt"), "note": "for reading only; no row was removed"}
            cfg.log(f"  semantic look-alikes: {len(hit)} of {len(qs)} eval questions have a row with cosine >= {cfg.semantic_min_cos} (see semantic_review.txt)")
        except StepFailed:
            raise
        except Exception as e:                                                      # noqa: BLE001
            glob["semantic"] = {"error": f"{type(e).__name__}: {e}"}
            cfg.log(f"!!! WARNING: the optional semantic review failed ({type(e).__name__}: {e}); recorded in the report, the run continues")
    if cfg.run_cross:
        try:
            cd = checks.cross_dataset_near_dups({k: final.rows_of(k) for k in keys}, text_fn=full_text, threshold=cfg.near_threshold, progress=cfg.progress)
            glob["cross_dataset"] = {"rows": cd["rows"], "rows_with_match_in": cd["rows_with_match_in"], "examples": cd["examples"], "threshold": cfg.near_threshold,
                                     "note": "report only; rows stay in their own dataset folder"}
        except Exception as e:                                                      # noqa: BLE001
            glob["cross_dataset"] = {"error": f"{type(e).__name__}: {e}"}
            cfg.log(f"!!! WARNING: the optional cross-dataset duplicate matrix failed ({type(e).__name__}: {e}); recorded, the run continues")
    # ---- the drop lists (one file per dataset) and the statistics
    for k in keys:
        c = stats[k]["counts"]
        seq = [c["converted"], c["after_exact_dedup"], c["after_near_dedup"], c["after_eval_decontam"], c["after_old_corpus_dedup"]]
        require(all(a >= b for a, b in zip(seq, seq[1:])), f"{k}: the funnel is not monotone: {seq}")
        require(c["final_rows"] == sum(alive[k]), f"{k}: final count disagrees with the survivor mask")
        c["n_raw"] = (raw.get("datasets", {}).get(k) or {}).get("n_raw")
        lines = ["index\tid\treason\n"]
        for i, (_, line) in enumerate(iter_stage_a(cfg, k)):
            if not alive[k][i]:
                lines.append(f"{i}\t{json.loads(line)['id']}\t{REASON[reason[k][i]]}\n")
        write_once(cfg.out / "filters" / f"{k}_dropped.tsv", "".join(lines))
        stats[k]["dropped_by_reason"] = dict(collections.Counter(l.rstrip("\n").split("\t")[2] for l in lines[1:]))
    out = {"datasets": stats, "global": glob}
    save_state(cfg, "step8_filters", out)
    fin = {k: stats[k]["counts"]["final_rows"] for k in keys}
    return (f"funnel done for {len(keys)} datasets; old corpus {n_old:,} rows found; decontamination dropped {dc['rows_flagged']:,}; "
            f"rows kept per dataset: {fin}"), out


# ====================================================================================================== 9: final rows, registry, report
def adoption_verdict(cfg: Cfg, key: str, final_rows: int, conv: dict, final_verification: Optional[dict] = None) -> dict:
    """A recommendation from the checks, never a fixed number.  The owner decides; every verdict is provisional until the 100 rows
    per dataset have been read by eye."""
    base = {"toolace": "hold", "wizard": "hold"}.get(key, "candidate")
    why: List[str] = []
    if final_rows == 0:
        return {"code": "reject", "ja": "不採用（採用できる行が0件）", "why": ["no row survived the checks"]}
    if key == "toolace":
        return {"code": "hold", "ja": "保留（生成モデルが未確認: カードにも論文にも明記なし）", "why": ["adoption is gated on the generator model being verified"]}
    if key == "wizard":
        a = conv.get("arith_error_rate") or {}
        rate, n_st = a.get("statement_error_rate"), a.get("statements_checked") or 0
        if rate is not None and n_st >= cfg.wizard_min_statements and rate <= cfg.wizard_max_error_rate:
            return {"code": "low_priority_candidate", "ja": f"低優先の採用候補（簡単な算術式の誤り率 {rate:.2%}（{n_st:,}式中）が基準 {cfg.wizard_max_error_rate:.0%} 以下。"
                                                           f"ただし検査できたのは単純な式だけ）",
                    "why": [f"statement error rate {rate:.4f} <= {cfg.wizard_max_error_rate} over {n_st} statements"]}
        shown = "測定不能" if rate is None else f"{rate:.2%}（{n_st:,}式中）"
        return {"code": "hold", "ja": f"保留（算術式の誤り率 {shown}。基準 {cfg.wizard_max_error_rate:.0%} 以下かつ{cfg.wizard_min_statements}式以上の確認が条件）",
                "why": [f"statement error rate {rate} over {n_st} statements; need <= {cfg.wizard_max_error_rate} over >= {cfg.wizard_min_statements}"]}
    if key == "tengentoppa":
        return {"code": "conditional", "ja": "条件付き候補（Qwen2.5-32B生成・人手確認なし。目視100行を読んでから判断）", "why": ["generated by Qwen2.5-32B-Instruct, not checked by humans"]}
    if key == "coding":
        v = final_verification if final_verification is not None else {x[len("verification_"):]: n for x, n in (conv.get("extra") or {}).items() if x.startswith("verification_")}
        n_eq = v.get("stored_output_equals_sandbox", 0)
        n_so = v.get("sandbox_only_no_stored_output", 0)
        return {"code": "candidate", "ja": f"採用候補（python行は実行検証済み: 出力一致 {n_eq:,}行・実行のみ {n_so:,}行。code_text行は未検証）",
                "why": ["python rows: stdout replaced by sandbox stdout", "code_text rows: not executed"]}
    if key == "jaquad":
        return {"code": "candidate", "ja": "採用候補（人手作成。CC BY-SA 3.0: 継承条件あり）", "why": ["human annotated"]}
    if key == "hotpot":
        return {"code": "candidate", "ja": "採用候補（人手作成・英語の短い行のみ。CC BY-SA 4.0: 継承条件あり）", "why": ["human annotated; English"]}
    return {"code": base, "ja": "採用候補", "why": why}


def _final_stats(rows_iter) -> dict:
    n, target_chars, masked, fam, ver = 0, 0, 0, collections.Counter(), collections.Counter()
    tool_rows = 0
    for r in rows_iter:
        n += 1
        target_chars += len(r["target"])
        m = sum(b - a for a, b in r.get("loss_mask_spans") or [])
        masked += m
        fam[r["family"]] += 1
        if r.get("verification"):
            ver[r["verification"]] += 1
        tool_rows += bool(r.get("tool_steps"))
    return {"rows": n, "target_chars": target_chars, "masked_chars": masked, "supervised_chars": target_chars - masked, "families": dict(fam),
            "verification": dict(ver), "rows_with_tool_steps": tool_rows}


def _attribution_md(info: dict, key: str, cfg: Cfg) -> str:
    L = ["# Attribution and licence (keep this file with the rows)", "", f"Dataset: {info['repo']} (pinned commit `{info['commit_sha']}`)",
         f"Licence: {info['license_name']} <{info['license_url']}>", f"Rows in this folder: converted to JSONL rows and filtered by v49_acquire (changes were made).", ""]
    L += [f"- {x}" for x in info["attribution"]]
    if info["share_alike"]:
        L += ["", "ShareAlike: these rows are an adaptation of the source data and must be shared under CC BY-SA (or a compatible licence) with the same attribution."]
    L += ["", "Each row's `source` and `license` fields carry the same information.", "", "Origin and quality notes:"]
    L += [f"- {x}" for x in info["generator_notes"] + info["quality_notes"]]
    return "\n".join(L) + "\n"


@step("9 final rows and report")
def step_finalize(cfg: Cfg):
    V = _CACHE.get("vocab")
    info5, info6b, raw, st8 = (load_state(cfg, n) for n in ("step5_acquire", "step6b_vocab_eval", "step6c_raw_checks", "step8_filters"))
    require(all(x is not None for x in (info5, info6b, raw, st8)), "earlier steps have not run (state missing): run cells 5, 6b, 6c and 8 first")
    for k in cfg.keys:
        ensure_stage_a(cfg, k)
    ck = _convert_ck(cfg)
    reg_ds: Dict[str, Any] = {}
    final_dir = Path(cfg.work) / f"final_{cfg.code_tag()}"
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
        rd = cfg.rows / k
        for name in files:
            publish_file(d / name, rd / name)
        reading = cfg.out / "reading" / f"final_{k}_100.txt"
        checks.sample_for_reading(final_rows(), n=100, seed=7, out_path=str(reading), title=f"{k} FINAL rows (after all filters)", vocab=V, progress=cfg.progress)
        a5 = info5[k]
        write_once(rd / "ATTRIBUTION.md", _attribution_md(a5, k, cfg))
        c = st8["datasets"][k]["counts"]
        manifest = {"schema": "v49_rows_manifest/1", "dataset": k, "repo": a5["repo"], "commit_sha": a5["commit_sha"], "license": a5["license_name"],
                    "row_license": a5["row_defaults"]["license"], "row_source": a5["row_defaults"]["source"], "counts": c, "files": files,
                    "supervised_chars": {"train": s_train["supervised_chars"], "all": s_all["supervised_chars"]}, "families": s_all["families"],
                    "adoption_at_acquire": a5["adoption_at_acquire"], "adoption_verdict": verdict, "code_sha256": cfg.code_hashes(),
                    "note": "adoption is provisional until the 100 reading rows have been read by eye"}
        write_once(rd / "manifest.json", jdump(manifest))
        # read back what is on Drive
        for name, e in files.items():
            require(checks.sha256_file(str(rd / name)) == e["sha256"], f"{k}: {name} on Drive differs from the file that was written")
        reg_ds[k] = {"key": k, "repo": a5["repo"], "commit_sha": a5["commit_sha"], "retrieved_utc": a5["retrieved_utc"],
                     "license": {"name": a5["license_name"], "ids": a5["license_ids"], "url": a5["license_url"], "share_alike": a5["share_alike"]},
                     "row_license": a5["row_defaults"]["license"], "row_source": a5["row_defaults"]["source"], "language": a5["language"],
                     "attribution": a5["attribution"], "generator_notes": a5["generator_notes"], "quality_notes": a5["quality_notes"],
                     "counts": c, "count_check": {"expected": raw["datasets"][k]["expected"], "counted": raw["datasets"][k]["n_raw"], "match": raw["datasets"][k]["count_match"],
                                                  "source": raw["datasets"][k]["count_source"]},
                     "raw": raw["datasets"][k], "convert": {x: man.get(x) for x in ("n_in", "n_out", "keep_rate", "rejects", "families", "splits", "extra", "backends",
                                                                                      "arith_error_rate", "language", "fit", "verification")},
                     "filters": st8["datasets"][k], "supervised_chars": {"train": s_train["supervised_chars"], "all": s_all["supervised_chars"]},
                     "final": {"by_split": dict(n_by), "families": s_all["families"], "verification": s_all["verification"], "rows_with_tool_steps": s_all["rows_with_tool_steps"],
                               "files": files}, "adoption": {"at_acquire": a5["adoption_at_acquire"], "verdict": verdict},
                     "paths": {"rows": str(rd), "reading_raw": raw["datasets"][k]["reading_file"], "reading_final": str(reading),
                               "converted_unfiltered": str(cfg.stage_a_drive(k)), "dropped_ids": str(cfg.out / "filters" / f"{k}_dropped.tsv")}}
        cfg.log(f"  {k}: {c['converted']:,} converted -> {want:,} final rows ({dict(n_by)}); verdict {verdict['code']}")
    mism = raw["mismatches"]
    reg = {"schema": "v49_acquire_registry/1", "pipeline_version": PIPELINE_VERSION, "owner_decision": acquire.OWNER_DECISION, "code_sha256": cfg.code_hashes(),
           "paths": {"out": str(cfg.out), "rows": str(cfg.rows), "external_data": str(cfg.ext)},
           "params": {"max_chars": cfg.max_chars, "oov_max_frac": convert.OOV_MAX_FRAC, "near_threshold": cfg.near_threshold, "ngram": cfg.ngram,
                      "boilerplate_df": cfg.boilerplate_df, "min_shared": cfg.min_shared, "split_pct_groups": convert.SPLIT_PCT,
                      "wizard_max_error_rate": cfg.wizard_max_error_rate, "wizard_min_statements": cfg.wizard_min_statements},
           "vocab": info6b["vocab"], "eval": info6b["eval"], "old_corpus": st8["global"]["old_corpus"], "decontam": st8["global"]["decontam"],
           "semantic_review": st8["global"]["semantic"], "cross_dataset_duplicates": st8["global"]["cross_dataset"], "count_mismatches": mism,
           "not_covered": NOT_COVERED, "datasets": reg_ds}
    write_once(cfg.out / "registry.json", jdump(reg))
    write_once(cfg.out / "REPORT_JA.md", render_report(reg, cfg))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        import numpy as _np
        np_ver = _np.__version__
    except Exception:                                                               # noqa: BLE001
        np_ver = None
    write_once(cfg.out / f"run_info_{now}.json", jdump({"utc": now, "python": sys.version.split()[0], "platform": platform.platform(), "numpy": np_ver,
                                                         "minhash_backend": checks.MinHasher().backend, "cpu_count": os.cpu_count(), "timings": cfg.timings}))
    tot = sum(v["counts"]["final_rows"] for v in reg_ds.values())
    return (f"{tot:,} rows in {len(reg_ds)} datasets written to {cfg.rows.name}/ and re-read from Drive; registry.json and REPORT_JA.md written to "
            f"{cfg.out.name}/; count mismatches: {len(mism)}"), {"registry": reg}


# ====================================================================================================== the Japanese report
def _n(x) -> str:
    return "-" if x is None else f"{x:,}"


def _pct(x) -> str:
    return "-" if x is None else f"{x:.2%}"


def render_report(reg: dict, cfg: Cfg) -> str:
    ds = reg["datasets"]
    L = ["# v49 外部データ 取得・検査レポート（自動生成）", "",
         "このファイルは colab_acquire_check.ipynb が作りました。数字はすべて今回の実行の実測値で、事前に決めた件数はありません。",
         "採用判定は検査結果からの暫定の提案です。各データセット100行の目視確認が済むまで確定しません。", "",
         "## 1. 結果の一覧", "",
         "| データセット | ライセンス | 元の件数 | 変換後 | 完全重複除去後 | 近似重複除去後 | 評価データ除外後 | 旧コーパス重複除去後（採用候補の行数） | 採用判定 | 教師あり文字数（train） | 備考 |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|"]
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
        L.append(f"| {k} | {d['license']['name']} | {_n(c.get('n_raw'))} | {_n(c['converted'])} | {_n(c['after_exact_dedup'])} | {_n(c['after_near_dedup'])} | "
                 f"{_n(c['after_eval_decontam'])} | {_n(c['after_old_corpus_dedup'])} | {d['adoption']['verdict']['ja']} | {_n(d['supervised_chars']['train'])} | {'；'.join(note) or '-'} |")
    oc, dc = reg["old_corpus"], reg["decontam"]
    L += ["", "「元の件数」は取得したファイルを端から端まで数えた件数です。「変換後」は形式・長さ（1,900文字以内）・語彙外文字（1%超は除外）・コードの実行検証などを通った行です。",
          "各段階は前の段階に残った行だけを対象にした順番の絞り込みです。", "",
          "## 2. 全体の確認結果", "",
          f"- 旧コーパス: {_n(oc['rows_found'])} 行を確認（計画の値 {_n(oc['rows_expected'])}）。ファイル `{oc['path']}`。",
          f"- 語彙: {reg['vocab']['n_single_chars']} 文字（{reg['vocab']['source']}）。コーパスからの推定ではなくモデル自身の語彙ファイルです。"
          + ("" if reg["vocab"].get("sha_matches_owner_pc_copy") else "（注意: PC上の語彙とハッシュが一致しません）"),
          f"- 評価データ: {len(reg['eval']['sources'])} 件の評価ファイルに加え、{', '.join(reg['eval']['extra_sources']) or 'なし'} を照合に使用。"
          f"13文字の断片（数字と空白を除いたもの）の種類数 {_n(reg['eval']['index']['distinct_shingles'])}。",
          f"- 評価データとの一致で除外した行: {_n(dc['rows_flagged'])} 行（定型句だけが一致した {_n(dc['rows_boilerplate_only'])} 行は残しました）。データセット別: {dc['flagged_by_dataset']}。"
          f"一致した断片の数の分布: {dc['shared_count_hist']}（1個だけの一致は、よくある言い回しの偶然の一致のことがあります。例は registry.json の decontam.examples）。",
          f"- 旧コーパスとの重複で除外した行: 完全一致 {oc['exact_dropped_by_dataset']}、近い重複（Jaccard {oc['threshold']} 以上）{oc['near_dropped_by_dataset']}。"]
    sem, cd = reg.get("semantic_review"), reg.get("cross_dataset_duplicates")
    if sem and "error" not in sem:
        L.append(f"- 意味的な類似の目視用リスト: 評価問題 {_n(sem['queries'])} 問のうち {_n(sem['queries_with_a_look_alike'])} 問に似た行があります（`filters/semantic_review.txt`）。行は除外していません。")
    elif sem:
        L.append(f"- 意味的な類似の確認は失敗しました（{sem['error']}）。")
    if cd and "error" not in cd:
        L.append(f"- データセット間の近似重複（Jaccard {cd['threshold']} 以上）: {cd['rows_with_match_in']}")
    elif cd:
        L.append(f"- データセット間の近似重複の確認は失敗しました（{cd['error']}）。")
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
              f"- 変換での主な除外理由: " + (", ".join(f"{r}={v:,}" for r, v in list((cv.get('rejects') or {}).items())[:5]) or "なし"),
              f"- 元データの語彙外文字の割合: {_pct(rw['oov']['oov_frac'])}（1%超の行 {_pct(rw['oov']['share_rows_over_threshold'])}）。2,048文字超の行: {_pct(rw['length']['share_over_2048'])}。",
              f"- 最終行数 {_n(c['final_rows'])}（分割 {d['final']['by_split']}）。ファミリー別: {d['final']['families']}。教師あり文字数 train {_n(d['supervised_chars']['train'])} / 全体 {_n(d['supervised_chars']['all'])}。",
              f"- 絞り込みで落ちた行の内訳: {d['filters'].get('dropped_by_reason')}。"]
        if d["final"]["verification"]:
            L.append(f"- 検証の内訳（最終行）: {d['final']['verification']}。")
        if cv.get("arith_error_rate"):
            a = cv["arith_error_rate"]
            L.append(f"- 算術式の検算: {_n(a['statements_checked'])} 式中 {_n(a['statements_wrong'])} 式が誤り（{_pct(a['statement_error_rate'])}）。行単位では {_pct(a['row_error_rate'])}。"
                     "検算できたのは単純な「a 演算 b = c」だけなので、実際の誤り率の下限です。")
        L += [f"- 目視用: 元データ100行 `{d['paths']['reading_raw']}`、最終行100行 `{d['paths']['reading_final']}`。",
              f"- 出力: `{d['paths']['rows']}`（train / dev / heldout の jsonl、manifest.json、ATTRIBUTION.md）。", ""]
    L += ["## 5. 検査の範囲と限界", ""] + [f"- {x}" for x in reg["not_covered"]]
    L += ["- 重複の判定: 同一データセット内は完全一致→MinHash（5文字断片・Jaccard 0.8以上）。JaQuAD と HotpotQA は1つの文章に複数の質問が付くため、質問＋根拠＋答えの部分で比べています。旧コーパスとは行全体で比べています。",
          "- 評価データとの照合は数字と空白を除いた13文字の完全一致です。言い換えや翻訳は検出できません。",
          "- 件数は実行ごとに決まります。同じコードと同じデータなら同じ結果になります（乱数は固定）。", ""]
    L += ["## 6. ライセンスと表示の義務", ""]
    for k in KEYS:
        d = ds.get(k)
        if d:
            L.append(f"- {k}: {d['license']['name']}" + ("。継承条件（ShareAlike）あり: 変換した行も同じライセンス系統で、出典を付けて共有する必要があります。" if d["license"]["share_alike"]
                                                          else "。ライセンス表示と変更点の明記が必要です。") + f" 出典の文言は `{d['paths']['rows']}/ATTRIBUTION.md` にあります。")
    L += ["", "## 7. 出力の場所", "", f"- 行データ: `{reg['paths']['rows']}/<データセット>/`", f"- 絞り込み前の変換結果・目視ファイル・状態・この報告: `{reg['paths']['out']}/`",
          f"- 取得した元ファイルと出所の記録: `{reg['paths']['external_data']}/`", ""]
    return "\n".join(L)
