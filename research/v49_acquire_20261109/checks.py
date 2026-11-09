#!/usr/bin/env python3
"""checks.py - streaming quality / contamination / duplicate checks for the GeoCPU v49 external-data pipeline.

Runs on a Google Colab CPU runtime (12 GB RAM).  Standard library only (+ optional numpy, pyarrow when the Colab image has
them).  Nothing here needs a secret, a network connection or a GPU.  Every function that reads rows takes any iterable of
dict rows (use iter_rows(path)); nothing ever loads a whole dataset; results are deterministic (fixed seeds, no use of the
per-process hash() for anything that is persisted or reported).

NOTEBOOK COOKBOOK (names are exactly the ones defined below)
    from checks import *
    V   = load_vocab(DRIVE_ROOT, out_dir=OUT)                        # 3,898-entry model vocabulary (3,893 single characters)
    rows = lambda: iter_rows(path)                                   # jsonl / parquet / SQuAD-like json, always streaming
    oov  = oov_stats(rows(), V)                                      # per-row OOV share, share of rows > 1 %, top OOV characters
    ln   = length_stats(rows(), group_key=lambda r: r.get("source")) # len(prompt+target)+1, share > 2048 and > 1900
    ex   = exact_dups(rows())                                        # exact duplicates inside one dataset
    nd   = near_dups_within(rows(), threshold=0.75)                  # MinHash (k=128, char 5-grams, LSH 32x4) near duplicates; see MinHasher for the cut
    cr   = cross_near_dups(rows(), iter_rows(OLD_CORPUS), sample_every=1)   # new rows vs the 423,348-row old corpus
    mx   = cross_dataset_near_dups({"jaquad": rows_a(), "hotpot": rows_b()})  # dataset x dataset duplicate matrix
    EF   = find_eval_files(DRIVE_ROOT)                               # eval manifests (bundle zip on Drive) -> EvalFiles
    EI   = build_eval_index(EF.sources)                              # digit-stripped 13-gram shingles of all eval texts
    dc   = decontaminate(rows(), EI)                                 # flagged_positions, boilerplate report, <=5 examples
    sm   = semantic_neighbors(eval_questions, rows())                # TF-IDF look-alikes for human reading (NOT exhaustive)
    ok, stdout, err = code_sandbox_check(code)                       # GeoCPU-executor-compatible re-run of python blocks
    sample_for_reading(rows(), n=100, seed=7, out_path=".../read_100_<dataset>.txt", vocab=V)   # the by-eye 100 rows
Rows that fail a check are not removed by these functions; they return positions / flags and the caller filters (a second
streaming pass with drop_flagged / oov_filter).  Outputs written to Google Drive go through write_new_file(): it only ever
creates files and refuses (FileExistsError) when a file exists with a different sha256.

MEMORY / TIME (measured on the owner's Windows PC with synthetic 2,000-character rows; a Colab Linux CPU is similar or faster)
    MinHash signature (k=128, 5-grams): ~2.8 ms/row with numpy (k=64 was 1.9), ~38 ms/row pure Python -> 300k rows: ~14 min / ~3 h
        (numpy is on every Colab image; MinHasher(use_numpy=False) gives the identical signatures, only slower)
    MinHashIndex: ~1.2 KB/row measured (512 B signature + 256 B band tables + ~290 B snippet + list slack) -> 300k rows ~ 360 MB;
        build() holds one transient copy of the signatures (154 MB at 300k rows).  The old corpus is streamed, never held;
        streaming all 423,348 old rows costs ~20-25 min (cross_near_dups(sample_every=N) divides that by N).
    EvalIndex: one python int per distinct eval 13-gram: 146k shingles = 12 MB RSS for the small manifests, so the 9 MB of
        all eval texts (incl. benchmark_8100_manifest.jsonl, 4.9 MB) is expected at a few hundred MB, not more.
    ExactIndex: ~100 B per row.  Sandbox: ~0.11 s per snippet wall time with 2 workers on Windows (process start), less on Linux.
    Everything else is O(1) memory per row.
READ THE RESULTS WITH THESE REAL-DATA LESSONS (see real_smoke_output.json, 2,000 rows of the JGLUE family against the real vocabulary
and the small eval manifests):  the 1 % OOV rule is per ROW, so one rare kanji drops an 80-character row (57 of 2,000 rows were over
1 % while only 0.09 % of all characters were OOV); rows that share one passage (JSQuAD / JaQuAD / HotpotQA: several questions per
context) are near duplicates of each other by character 5-grams (J 0.83-0.94) - pass text_fn=<question + answer only> to
near_dups_within when the passage is not what should be deduplicated; and a shared 13-gram is a lead, not a verdict
(a law title appears in both a Wikipedia row and a JamC-QA item): read decontaminate()['examples'].
"""
from __future__ import annotations

import array
import ast
import bisect
import collections
import dataclasses
import errno
import glob as _glob
import gzip
import hashlib
import heapq
import io
import itertools
import json
import math
import operator
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

if array.array("I").itemsize != 4:                       # pragma: no cover
    raise ImportError("checks.py needs a 4-byte array('I')")

# =====================================================================================================================
# Constants: where things live on Google Drive (found with the Drive connector on 2026-11-09)
# =====================================================================================================================
DRIVE_ROOT = "/content/drive/MyDrive/GeoCPU_v47_fullmodel_20261106"        # Drive folder id 1f4ny9Sr26rOnU8mOoEyrKxbQw60iH2Aj
DRIVE_ROOT_FOLDER_ID = "1f4ny9Sr26rOnU8mOoEyrKxbQw60iH2Aj"

# The existing corpus (inherited untouched).  Drive file id 1hBbEQ8eovdZZXNGj7PLHN3O4co7zGIBR, 614,482,850 bytes, 423,348 rows.
OLD_CORPUS_REL = "_v48_dataset_cache/train_40500_exact.jsonl"
OLD_CORPUS = DRIVE_ROOT + "/" + OLD_CORPUS_REL
OLD_CORPUS_ROWS = 423_348

# The model vocabulary is NOT a loose file in the run folders.  It is key "vocab" (token -> id, 3,898 entries = 3,893 single
# characters + <pad> <unk> <bos> <eos> <sep>, ids 0..3897) of base_metadata.json inside the nested zip
#   GeoCPU_v47_fullmodel_bundle_v1.zip (Drive id 1c87pqt4UoVV3I6ECv__Iip9Ob1KotkP0, 26,925,754 B; the _READY copy 1EYF4CmXLP-...
#   has the same size)  ->  assets/r6_bundle.zip  ->  base_metadata.json
# (the metadata also says parameters = 247,559,448 and model_protocol = character_next_token).
# Two stand-alone vocab.json / omni_domain_vocab_v4.json files (68,705 B) exist elsewhere on Drive (folders of 2026-10-17/21,
# not under GeoCPU_v47_fullmodel_20261106); they were NOT opened and are not used.
VOCAB_BUNDLE_ZIPS = ("GeoCPU_v47_fullmodel_bundle_v1.zip", "GeoCPU_v47_fullmodel_bundle_v1_READY.zip")
VOCAB_NESTED_ZIP = "assets/r6_bundle.zip"
VOCAB_MEMBER = "base_metadata.json"
VOCAB_DRIVE_PATH = DRIVE_ROOT + "/" + VOCAB_BUNDLE_ZIPS[0] + "::" + VOCAB_NESTED_ZIP + "::" + VOCAB_MEMBER + "[vocab]"
EXPECTED_VOCAB_SIZE = 3898
EXPECTED_SINGLE_CHARS = 3893
VOCAB_SPECIALS = ("<pad>", "<unk>", "<bos>", "<eos>", "<sep>")

# Eval manifests (Drive: inside GeoCPU_v47_fullmodel_bundle_v1.zip -> assets/r6_bundle.zip -> data/<name>; code/ in the outer zip).
EVAL_BUNDLE_ZIPS = VOCAB_BUNDLE_ZIPS
EVAL_NESTED_ZIP = VOCAB_NESTED_ZIP
EVAL_REQUIRED_JSONL = {
    "benchmark_8100_manifest.jsonl": "JMMLU / JamC-QA 8,100-item benchmark manifest",
    "jamcqa_manifest_test.jsonl": "JamC-QA test manifest",
    "eval_swe_step_control_heldout.jsonl": "SWE step-control heldout",
    "eval_swe_step_control_heldout_v43labels.jsonl": "SWE step-control heldout (v43 labels)",
    "eval_v33_state_tracking_heldout.jsonl": "state tracking heldout (v33)",
    "eval_v38_decomposed_state_bench200.jsonl": "decomposed state bench (v38)",
    "eval_v41_logical_instruction_operators_heldout.jsonl": "T5 logical instruction operators (v41)",
    "eval_v42_search_retrieval_aligned.jsonl": "T6 search retrieval, aligned prompts (v42)",
    "eval_v42_search_retrieval_heldout.jsonl": "T6 search retrieval, heldout prompts (v42)",
}
EVAL_REQUIRED_PY = {                                     # python sources whose string literals are eval prompts
    "tier8_readonly_v44fix.py": "T8 prompts (4 hard-coded CASES; outer bundle zip, code/)",
}
# v48 unseen_eval / dev sets: 4 families x 2 files x 128 rows, selfauthored templates.  They are NOT stored as stand-alone
# files on Drive (only the evaluation event folders step40500_v48sets/ and step61500/ are); local copy:
# research/v48_balanced_augmentation_20261108/data/.  Optional: found when somebody copies them under the Drive root.
EVAL_OPTIONAL_V48 = tuple(f"{fam}_{kind}.jsonl" for fam in ("evidence", "japanese", "logic", "python") for kind in ("unseen_eval", "dev"))
EVAL_OPTIONAL_SUITE_ZIPS = ("GeoCPU_v47_train_eval_suite_v2.zip", "GeoCPU_v47_continue30500_train_eval_v1.zip",
                            "GeoCPU_v47_python_tir_eval_25500_28000_v1.zip")      # small zips at the Drive root (30-218 KB)
# The other frozen eval suites of the owner's registry (research/v47_practical_20261107/frozen/registry.json, 'leakage_sources', the
# files that are not T1-T8 manifests).  Each is searched as a stand-alone file, in the bundle's data/ folder and (by file name) inside
# the suite zips; one that is not found is reported as NOT covered (find_eval_files(...).missing_optional / the report), never skipped
# silently.  python120 = selection_120.json; engineering64 = engineering_heldout.jsonl; visual10 = visual_heldout.jsonl;
# new_public:* = the *_dev.jsonl files.
EVAL_REGISTRY_EXTRA = {
    "dev_overlay.jsonl": "public_dev80: public search development overlay",
    "dev_numina_tir_strict_numeric.jsonl": "python120 / math_python: NuminaMath-TIR strict numeric dev",
    "dev_tldr_terminal.jsonl": "new_public:terminal dev",
    "japanese_math_python_dev.jsonl": "new_public:japanese_math_python dev",
    "plos_science_dev.jsonl": "new_public:plos_science dev",
    "python_docs_ja_core_dev.jsonl": "new_public:python_ja dev",
    "python_reference_dev.jsonl": "new_public:python_reference dev",
    "science_code_complete_dev.jsonl": "new_public:science_code dev",
    "science_engineering_verified_dev.jsonl": "new_public:science_engineering_verified dev",
    "search_post_result_dev.jsonl": "new_public:search_next dev",
    "engineering_heldout.jsonl": "engineering64 held-out",
    "visual_heldout.jsonl": "visual10 held-out",
    "dev_after_final_result_198.jsonl": "search_final198",
    "selection_120.json": "python120: 80 Japanese + 40 NuminaMath-TIR selection",
}
# which file stands for which tier of the owner's T1-T8 (registry.json 'suites'); T4 and T7 share one source file
EVAL_TIER_FILES = {
    "T1 held-out SWE step control": ("eval_swe_step_control_heldout.jsonl", "eval_swe_step_control_heldout_v43labels.jsonl"),
    "T2 decomposed state tracking": ("eval_v33_state_tracking_heldout.jsonl", "eval_v38_decomposed_state_bench200.jsonl"),
    "T3 JamC-QA manifest": ("jamcqa_manifest_test.jsonl",),
    "T4 broad 8100": ("benchmark_8100_manifest.jsonl",),
    "T5 logical instruction operators": ("eval_v41_logical_instruction_operators_heldout.jsonl",),
    "T6 search retrieval": ("eval_v42_search_retrieval_aligned.jsonl", "eval_v42_search_retrieval_heldout.jsonl"),
    "T7 closed/open-book A/B (same source file as T4)": ("benchmark_8100_manifest.jsonl",),
    "T8 code/explanation form probe": ("tier8_readonly_v44fix.py",),
}
EVAL_TEXT_KEYS = ("prompt", "question", "choices", "options", "target", "answer", "full_text", "input", "instruction", "text",
                  "context", "expected_query", "final_answer", "python_code")
MAX_LEN = 2048                                           # context length in characters; a row needs len(prompt+target)+1 <= MAX_LEN

ProgressCB = Callable[[str, int], None]
_SKIP_DIRS = {"checkpoints", "raw", "artifacts", "__pycache__"}


# =====================================================================================================================
# Small utilities: progress, safe writes, hashing
# =====================================================================================================================
class _Ticker:
    """Calls cb(label, n_done) about every `every` items.  cb may be None."""
    __slots__ = ("cb", "label", "every", "n", "_next")

    def __init__(self, cb: Optional[ProgressCB], label: str, every: int = 20000):
        self.cb, self.label, self.every, self.n = cb, label, max(1, int(every)), 0
        self._next = self.every

    def tick(self, k: int = 1) -> None:
        self.n += k
        if self.cb is not None and self.n >= self._next:
            self.cb(self.label, self.n)
            while self._next <= self.n:
                self._next += self.every

    def done(self) -> None:
        if self.cb is not None:
            self.cb(self.label + " [done]", self.n)


def make_printer(min_interval_s: float = 10.0, stream=None) -> ProgressCB:
    """A ready-made progress callback: prints 'elapsed label: n (rate/s)' at most every min_interval_s seconds."""
    t0, last = time.time(), [0.0]

    def cb(label: str, done: int) -> None:
        now = time.time()
        if label.endswith("[done]") or now - last[0] >= min_interval_s:
            last[0] = now
            el = now - t0
            print(f"[{el:8.1f}s] {label}: {done:,}" + (f" ({done / el:,.0f}/s)" if el > 0 else ""), file=stream or sys.stderr, flush=True)
    return cb


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


_FSYNC_UNSUPPORTED = {getattr(errno, n) for n in ("EINVAL", "ENOTSUP", "EOPNOTSUPP", "ENOSYS") if hasattr(errno, n)}


def fsync_file(fd: int) -> None:
    """os.fsync that ignores ONLY 'this file system cannot fsync' (EINVAL / ENOTSUP / EOPNOTSUPP / ENOSYS).  Every other error (EIO, ENOSPC,
    EDQUOT, ...) is raised: on a FUSE mount such as Google Drive the quota-full or I/O error that buffered writes defer shows up here, and
    swallowing it would record a failed write as a successful one."""
    try:
        os.fsync(fd)
    except OSError as e:
        if e.errno not in _FSYNC_UNSUPPORTED:
            raise


def quarantine_part(path: str) -> str:
    """Move a leftover `<x>.part` of a dead run aside as `<x>.part.stale-N` (never deleted: the owner's rule); returns the new name."""
    n = 1
    while os.path.lexists(f"{path}.stale-{n}"):
        n += 1
    dst = f"{path}.stale-{n}"
    os.rename(path, dst)
    return dst


def write_new_file(path: str, data: Union[str, bytes], *, encoding: str = "utf-8") -> str:
    """Create `path` with `data`.  Never overwrites: if the file exists with the same sha256 nothing happens ('identical'); with a
    different sha256 FileExistsError is raised.  The bytes go to `<path>.part` first (flushed and fsynced) and are renamed into place
    only when `path` still does not exist, so a runtime that dies in the middle of a write leaves a `.part`, never a truncated final
    file (a leftover `.part` of an earlier run is moved aside, not deleted).  Text is written UTF-8 with lone surrogates as backslash
    escapes (they occur in raw web text and would otherwise crash the write).  Returns 'created' or 'identical'."""
    if isinstance(data, str):
        data = data.encode(encoding, "backslashreplace")
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def compare() -> str:
        old = sha256_file(path)
        new = hashlib.sha256(data).hexdigest()
        if old == new:
            return "identical"
        raise FileExistsError(f"refusing to overwrite {path}: existing sha256 {old[:16]}... != new {new[:16]}...") from None
    if os.path.lexists(path):
        return compare()
    part = path + ".part"
    if os.path.lexists(part):
        quarantine_part(part)
    with open(part, "xb") as f:
        f.write(data)
        f.flush()
        fsync_file(f.fileno())
    if os.path.lexists(path):                                    # somebody created it while we wrote: never replace it
        return compare()
    os.rename(part, path)
    return "created"


def digest64(text: str) -> int:
    """Deterministic 64-bit digest of a string (blake2b); used for persisted / cross-process keys."""
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


def _string_leaves(obj: Any, depth: int = 0) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif depth < 5 and isinstance(obj, dict):
        for v in obj.values():
            yield from _string_leaves(v, depth + 1)
    elif depth < 5 and isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _string_leaves(v, depth + 1)


def default_text(row: Any) -> str:
    """The text of a row: full_text, else prompt + target, else chat messages, else every string value joined by newlines."""
    if isinstance(row, str):
        return row
    ft = row.get("full_text")
    if isinstance(ft, str):
        return ft
    p, t = row.get("prompt"), row.get("target")
    if isinstance(p, str) and isinstance(t, str):
        return p + t
    for key in ("messages", "conversations"):
        msgs = row.get(key)
        if isinstance(msgs, list) and msgs:
            return "\n".join(str(m.get("content", m.get("value", ""))) if isinstance(m, dict) else str(m) for m in msgs)
    return "\n".join(_string_leaves(row))


def default_id(row: Any, pos: int = -1) -> str:
    if isinstance(row, dict):
        for k in ("id", "_id", "uid", "q_id", "question_id", "sentence_pair_id"):
            if row.get(k) is not None:
                return str(row[k])
    return f"#{pos}"


# =====================================================================================================================
# iter_rows: jsonl / parquet / SQuAD-like json, always streaming
# =====================================================================================================================
_WS_COMMA = re.compile(r"[\s,]*")
_JSON_TOP_ARRAY = re.compile(r"\A\s*\[")
_JSON_DATA_KEY = re.compile(r'\A\s*\{(?:\s*"[^"\\]*"\s*:\s*(?:"(?:[^"\\]|\\.)*"|-?\d[\d.eE+-]*|true|false|null)\s*,)*\s*"data"\s*:\s*\[')


def _flatten_squad_article(art: dict) -> Iterator[dict]:
    """SQuAD / JaQuAD / JSQuAD article -> one flat dict per question."""
    title = art.get("title", "")
    for pi, para in enumerate(art.get("paragraphs", [])):
        ctx = para.get("context", "")
        for qa in para.get("qas", []):
            answers = qa.get("answers") or []
            row = {k: v for k, v in qa.items() if k not in ("id", "question", "answers", "is_impossible")}
            row.update({"id": qa.get("id"), "title": title, "para_index": pi, "context": ctx, "question": qa.get("question", ""),
                        "answers": answers, "answer_texts": [a.get("text") for a in answers if isinstance(a, dict)],
                        "answer_starts": [a.get("answer_start") for a in answers if isinstance(a, dict)],
                        "is_impossible": bool(qa.get("is_impossible", False))})
            yield row


def _stream_array_items(f, buf: str, chunk: int = 1 << 20) -> Iterator[Any]:
    """Yield the items of a JSON array whose '[' is already consumed; `buf` is the text right after it.  Items must be
    objects (a number cut at a chunk boundary could be mis-read).  Only the current item plus < 1 chunk is in memory."""
    dec = json.JSONDecoder()
    pos = 0
    while True:
        while True:
            pos = _WS_COMMA.match(buf, pos).end()
            if pos < len(buf):
                break
            more = f.read(chunk)
            if not more:
                raise ValueError("unexpected end of JSON array")
            buf, pos = buf[pos:] + more, 0
        if buf[pos] == "]":
            return
        grow = chunk
        while True:
            try:
                obj, end = dec.raw_decode(buf, pos)
                break
            except json.JSONDecodeError:
                more = f.read(grow)
                if not more:
                    raise
                buf, pos = buf[pos:] + more, 0
                grow = min(grow * 2, 1 << 26)
        yield obj
        pos = end
        if pos > (1 << 20) and pos > len(buf) // 2:
            buf, pos = buf[pos:], 0


def _iter_jsonl(path: str, strict: bool, stats: Optional[dict] = None) -> Iterator[Any]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    # strict: undecodable bytes and bad lines raise.  lenient (strict=False): undecodable bytes become U+FFFD (the converter reads the
    # same way), bad lines are skipped and counted in stats["bad_lines"] / stats["decode_replacements"] when a dict is given.
    with opener(path, "rt", encoding="utf-8", errors="strict" if strict else "replace", newline="\n") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if stats is not None and not strict and "�" in line:
                stats["decode_replacements"] = stats.get("decode_replacements", 0) + 1
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    raise ValueError(f"{path}:{ln}: bad JSON line: {e}") from e
                if stats is not None:
                    stats["bad_lines"] = stats.get("bad_lines", 0) + 1


def _iter_json(path: str, squad_flatten: bool) -> Iterator[Any]:
    opener = gzip.open if path.lower().endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as f:
        head = f.read(1 << 20)
        m = _JSON_TOP_ARRAY.match(head)
        if not m:
            m = _JSON_DATA_KEY.match(head)
        if m:
            items: Iterable[Any] = _stream_array_items(f, head[m.end():])
        else:                                            # unknown layout: whole-file parse (files of this kind are < 100 MB)
            obj = json.loads(head + f.read())
            if isinstance(obj, list):
                items = obj
            elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
                items = obj["data"]
            else:
                items = [obj]
        for it in items:
            if squad_flatten and isinstance(it, dict) and "paragraphs" in it:
                yield from _flatten_squad_article(it)
            else:
                yield it


def _iter_parquet(path: str, columns: Optional[Sequence[str]], batch_size: int) -> Iterator[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:                             # pragma: no cover
        raise ImportError("reading .parquet needs pyarrow (pip install -q pyarrow)") from e
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(columns) if columns else None):
        yield from batch.to_pylist()


def iter_rows(path: str, *, columns: Optional[Sequence[str]] = None, limit: Optional[int] = None, batch_size: int = 512,
              squad_flatten: bool = True, strict: bool = True, progress: Optional[ProgressCB] = None,
              progress_every: int = 50000, label: Optional[str] = None, stats: Optional[dict] = None) -> Iterator[Any]:
    """Stream the rows of a .jsonl(.gz) / .parquet / .json(.gz) file as dicts.
      .jsonl   one JSON value per line (blank lines skipped; strict=True raises on a bad line or an undecodable byte; False replaces
               undecodable bytes, skips bad lines and counts them in `stats` (bad_lines, decode_replacements) when a dict is passed)
      .parquet pyarrow record batches of `batch_size` rows (only `columns` are read); RAM ~ one row group (~100-300 MB for 330 MB files)
      .json    a top-level array of objects, or {"data": [...]} (SQuAD / JaQuAD); decoded item by item.  With squad_flatten
               every article with "paragraphs" is flattened to one dict per question: id, title, context, question, answers,
               answer_texts, answer_starts, is_impossible.  Any other layout is parsed whole (use only for small files).
    """
    p = os.fspath(path)
    low = p.lower()
    if low.endswith((".jsonl", ".jsonl.gz", ".ndjson")):
        gen = _iter_jsonl(p, strict, stats)
    elif low.endswith(".parquet"):
        gen = _iter_parquet(p, columns, batch_size)
    elif low.endswith((".json", ".json.gz")):
        gen = _iter_json(p, squad_flatten)
    else:
        raise ValueError(f"unsupported file type: {p}")
    tick = _Ticker(progress, label or os.path.basename(p), progress_every)
    try:
        for i, row in enumerate(gen):
            if limit is not None and i >= limit:
                break
            tick.tick()
            yield row
    finally:
        gen.close()
    tick.done()


def row_count_hint(path: str) -> Optional[int]:
    """Exact row count for parquet (footer only); None for other formats."""
    if os.fspath(path).lower().endswith(".parquet"):
        import pyarrow.parquet as pq
        return pq.ParquetFile(path).metadata.num_rows
    return None


# =====================================================================================================================
# Vocabulary + OOV
# =====================================================================================================================
@dataclasses.dataclass
class VocabInfo:
    chars: frozenset
    source: str
    n_entries: int = 0
    specials: Tuple[str, ...] = ()
    derived: bool = False
    warnings: List[str] = dataclasses.field(default_factory=list)

    def __contains__(self, c: str) -> bool:
        return c in self.chars

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def sha256_chars(self) -> str:
        return hashlib.sha256("".join(sorted(self.chars)).encode("utf-8", "surrogatepass")).hexdigest()

    def as_dict(self) -> dict:
        return {"source": self.source, "n_entries": self.n_entries, "n_single_chars": len(self.chars), "specials": list(self.specials),
                "derived": self.derived, "sha256_of_sorted_chars": self.sha256_chars, "warnings": list(self.warnings)}


def _vocab_dict_from_obj(obj: Any) -> Optional[dict]:
    if isinstance(obj, dict):
        v = obj.get("vocab")
        if isinstance(v, dict) and v:
            return v
        m = obj.get("model")
        if isinstance(m, dict) and isinstance(m.get("vocab"), dict):
            return m["vocab"]
        if len(obj) >= 100 and all(isinstance(x, int) for x in obj.values()):
            return obj
    return None


def vocab_from_json_bytes(data: bytes, source: str, *, expected_size: Optional[int] = EXPECTED_VOCAB_SIZE,
                          expected_single: Optional[int] = EXPECTED_SINGLE_CHARS) -> VocabInfo:
    """VocabInfo from the bytes of a vocab.json (token -> id) or a base_metadata.json (key 'vocab').  Raises ValueError when the
    file has no vocab or its size differs from the expected 3,898 entries / 3,893 single characters (None disables a check)."""
    vd = _vocab_dict_from_obj(json.loads(data.decode("utf-8-sig")))
    if vd is None:
        raise ValueError("no vocab dict in " + source)
    chars = frozenset(k for k in vd if isinstance(k, str) and len(k) == 1)
    specials = tuple(sorted(k for k in vd if isinstance(k, str) and len(k) != 1))
    if expected_size is not None and len(vd) != expected_size:
        raise ValueError(f"{source}: {len(vd)} vocab entries, expected {expected_size}")
    if expected_single is not None and len(chars) != expected_single:
        raise ValueError(f"{source}: {len(chars)} single-character entries, expected {expected_single}")
    return VocabInfo(chars=chars, source=source, n_entries=len(vd), specials=specials)


def _walk_limited(root: str, max_depth: int) -> Iterator[Tuple[str, List[str]]]:
    root = os.path.abspath(root)
    base = root.rstrip(os.sep).count(os.sep)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = dirpath.rstrip(os.sep).count(os.sep) - base
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")) if depth < max_depth else []
        yield dirpath, sorted(filenames)


def _bundle_zip_paths(drive_root: str) -> List[str]:
    out = [os.path.join(drive_root, n) for n in VOCAB_BUNDLE_ZIPS if os.path.isfile(os.path.join(drive_root, n))]
    for p in sorted(_glob.glob(os.path.join(drive_root, "GeoCPU_v47_fullmodel_bundle*.zip"))):
        if p not in out:
            out.append(p)
    return out


def load_vocab(drive_root: str = DRIVE_ROOT, out_dir: Optional[str] = None, *, corpus_path: Optional[str] = None,
               allow_derive: bool = True, max_depth: int = 3, expected_size: Optional[int] = EXPECTED_VOCAB_SIZE,
               expected_single: Optional[int] = EXPECTED_SINGLE_CHARS, cache_name: str = "vocab_cache.json",
               progress: Optional[ProgressCB] = None) -> VocabInfo:
    """The model's character vocabulary.  Search order:
      0. out_dir/vocab_cache.json (only a cache of a NON-derived result is trusted);
      1. base_metadata.json['vocab'] inside GeoCPU_v47_fullmodel_bundle_v1.zip -> assets/r6_bundle.zip (VOCAB_DRIVE_PATH): the model's
         own file, always preferred;
      1b. only when no bundle yields it: a stand-alone vocab / tokenizer / base_metadata JSON in the Drive run folders (names vocab*.json,
         *vocab.json, tokenizer*.json, base_metadata.json, depth <= max_depth); only a file with exactly 3,898 entries / 3,893 single
         characters is accepted - anything else is ignored with a warning (and the source says it was a loose file);
      2. else (allow_derive) the union of the characters of every full_text of OLD_CORPUS (one streaming pass).  That is the
         census of what the model has SEEN (3,498 characters on 2026-11-09), not its vocabulary: it lacks ~400 vocabulary
         characters and has 5 the vocabulary lacks.  A warning says so; the result is cached with derived=True.
    The result (with its source and warnings) is cached as JSON in out_dir when given."""
    warnings: List[str] = []
    cache_path = os.path.join(out_dir, cache_name) if out_dir else None
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path, encoding="utf-8") as cf:
                c = json.load(cf)
            if not c.get("derived") and isinstance(c.get("chars"), str):
                return VocabInfo(chars=frozenset(c["chars"]), source="cache:" + str(c.get("source")), n_entries=int(c.get("n_entries", 0)),
                                 specials=tuple(c.get("specials", ())), warnings=list(c.get("warnings", [])))
        except (ValueError, OSError):
            warnings.append("ignored unreadable vocab cache " + cache_path)
    info: Optional[VocabInfo] = None
    names = re.compile(r"(?i)^(vocab.*\.json|.*vocab\.json|tokenizer.*\.json|base_metadata\.json)$")
    if os.path.isdir(drive_root):
        # the model's own file first: base_metadata.json inside the bundle.  A stray vocab*.json elsewhere on Drive that merely has
        # 3,898 entries must never override it (a stale vocabulary would silently drive the OOV filter for every row).
        for zp in _bundle_zip_paths(drive_root):
            try:
                with zipfile.ZipFile(zp) as z:
                    if VOCAB_NESTED_ZIP not in z.namelist():
                        continue
                    with zipfile.ZipFile(io.BytesIO(z.read(VOCAB_NESTED_ZIP))) as inner:
                        if VOCAB_MEMBER not in inner.namelist():
                            continue
                        info = vocab_from_json_bytes(inner.read(VOCAB_MEMBER), f"{zp}::{VOCAB_NESTED_ZIP}::{VOCAB_MEMBER}[vocab]",
                                                     expected_size=expected_size, expected_single=expected_single)
                        break
            except (ValueError, OSError, zipfile.BadZipFile) as e:
                warnings.append(f"rejected bundle {zp}: {e}")
        if info is None:
            for dirpath, files in _walk_limited(drive_root, max_depth):
                for fn in files:
                    if not names.match(fn):
                        continue
                    p = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(p) > 5 << 20:
                            continue
                        with open(p, "rb") as vf:
                            info = vocab_from_json_bytes(vf.read(), p, expected_size=expected_size, expected_single=expected_single)
                        warnings.append("vocabulary taken from a loose file, not from the model bundle: " + p)
                        break
                    except (ValueError, OSError) as e:
                        warnings.append(f"rejected vocab candidate: {e}")
                if info:
                    break
    if info is None:
        if not allow_derive:
            raise FileNotFoundError("model vocabulary not found under " + drive_root + "; warnings: " + "; ".join(warnings))
        cp = corpus_path or os.path.join(drive_root, OLD_CORPUS_REL)
        chars: set = set()
        tick = _Ticker(progress, "derive vocab from " + os.path.basename(cp), 50000)
        for row in iter_rows(cp):
            chars.update(default_text(row))
            tick.tick()
        tick.done()
        warnings.append("DERIVED from the old corpus census (union of full_text characters), not the model's own vocabulary file")
        info = VocabInfo(chars=frozenset(chars), source="derived:" + cp, n_entries=len(chars), derived=True)
    info.warnings = warnings + info.warnings
    if cache_path:
        try:
            write_new_file(cache_path, json.dumps({"chars": "".join(sorted(info.chars)), "source": info.source, "n_entries": info.n_entries,
                                                   "specials": list(info.specials), "derived": info.derived, "warnings": info.warnings,
                                                   "sha256_of_sorted_chars": info.sha256_chars}, ensure_ascii=False))
        except FileExistsError:
            info.warnings.append("vocab cache exists with different content; left untouched")
    return info


# The training runtime maps every row through this table before it looks characters up (qwenputer_controller_v30.py,
# _BASE_NORM_DICT + the full-width ASCII loop; convert.py keeps the same copy).  OOV must be measured AFTER it.
_RUNTIME_NORM = {
    "ϵ": "ε", "ϕ": "φ", "ϑ": "θ", "ϰ": "κ", "ϱ": "ρ", "ς": "σ", "ο": "o",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X", "＞": ">", "＜": "<", "⇔": "↔", "↦": "→", "⊨": "⊢", "⊤": "T", "∅": "φ", "≢": "≠",
    "÷": "/", "⋅": "・", "∙": "・", "⟨": "(", "⟩": ")", "⟪": "(", "⟫": ")", "⟦": "[", "⟧": "]", "⌈": "[", "⌉": "]",
    "⌊": "[", "⌋": "]", "−": "-",
}
for _c in range(0xFF01, 0xFF5F):
    if chr(_c) not in ("（", "）", "！", "？", "：", "～"):
        _RUNTIME_NORM[chr(_c)] = chr(_c - 0xFEE0)
RUNTIME_NORMALIZE = str.maketrans(_RUNTIME_NORM)


def verify_runtime_normalizer(controller_source: str) -> Tuple[bool, str]:
    """Compare RUNTIME_NORMALIZE with `_BASE_NORM_DICT` (+ full-width loop) parsed from the text of qwenputer_controller_v30.py
    (inside assets/r6_bundle.zip).  Returns (identical, explanation)."""
    tree = ast.parse(controller_source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_BASE_NORM_DICT" for t in node.targets):
            d = dict(ast.literal_eval(node.value))
            for c in range(0xFF01, 0xFF5F):
                if chr(c) not in ("（", "）", "！", "？", "：", "～"):
                    d[chr(c)] = chr(c - 0xFEE0)
            if d == _RUNTIME_NORM:
                return True, f"identical ({len(d)} entries)"
            only_a = sorted(set(d) - set(_RUNTIME_NORM))
            only_b = sorted(set(_RUNTIME_NORM) - set(d))
            diff = sorted(k for k in set(d) & set(_RUNTIME_NORM) if d[k] != _RUNTIME_NORM[k])
            return False, f"controller-only keys {only_a[:10]}, checks-only keys {only_b[:10]}, differing values {diff[:10]}"
    return False, "_BASE_NORM_DICT not found"


def _chars_of(vocab: Any) -> frozenset:
    if isinstance(vocab, VocabInfo):
        return vocab.chars
    return vocab if isinstance(vocab, frozenset) else frozenset(vocab)


def oov_fraction(text: str, chars: frozenset) -> Tuple[float, int]:
    """(share of characters not in `chars`, their count)."""
    if not text:
        return 0.0, 0
    bad = set(text) - chars
    if not bad:
        return 0.0, 0
    n = sum(text.count(c) for c in bad)
    return n / len(text), n


def oov_stats(rows: Iterable[Any], vocab: Any, *, text_fn: Optional[Callable[[Any], str]] = None, normalize: Union[bool, dict] = True,
              threshold: float = 0.01, top_k: int = 30, keep_per_row: bool = False, group_key: Optional[Callable[[Any], str]] = None,
              progress: Optional[ProgressCB] = None, progress_every: int = 50000) -> dict:
    """Out-of-vocabulary statistics of the row texts.  The texts are first mapped through the runtime's symbol normaliser
    (normalize=True; False measures raw characters; a str.maketrans table is also accepted).
    Returns n_rows, n_chars, oov_chars, oov_frac (overall), rows_with_any_oov, rows_over_threshold and share_rows_over_threshold
    (strictly more than `threshold` = 1 % of a row's characters), rows_changed_by_normalizer, histogram of per-row fractions,
    top_oov (char, codepoint, name, count, rows), per_group (same numbers per group_key value) and, with keep_per_row,
    per_row_frac (array('f') of the per-row fractions in stream order)."""
    chars = _chars_of(vocab)
    table = RUNTIME_NORMALIZE if normalize is True else (normalize if isinstance(normalize, dict) else None)
    text_fn = text_fn or default_text
    tick = _Ticker(progress, "oov_stats", progress_every)

    def new() -> dict:
        return {"n_rows": 0, "n_chars": 0, "oov_chars": 0, "rows_with_any_oov": 0, "rows_over_threshold": 0, "empty_rows": 0}
    total, groups = new(), collections.defaultdict(new)
    char_count: collections.Counter = collections.Counter()
    char_rows: collections.Counter = collections.Counter()
    hist = collections.Counter()
    per_row = array.array("f") if keep_per_row else None
    changed = 0
    for row in rows:
        text = text_fn(row)
        if table is not None:
            t2 = text.translate(table)
            if t2 != text:
                changed += 1
            text = t2
        frac, n_bad = oov_fraction(text, chars)
        targets = [total] + ([groups[group_key(row)]] if group_key else [])
        for s in targets:
            s["n_rows"] += 1
            s["n_chars"] += len(text)
            s["oov_chars"] += n_bad
            s["empty_rows"] += not text
            if n_bad:
                s["rows_with_any_oov"] += 1
            if frac > threshold:
                s["rows_over_threshold"] += 1
        if n_bad:
            for c in set(text) - chars:
                char_count[c] += text.count(c)
                char_rows[c] += 1
        hist["0" if frac == 0 else ("(0,0.1%]" if frac <= 0.001 else ("(0.1%,1%]" if frac <= 0.01 else ("(1%,5%]" if frac <= 0.05 else ">5%")))] += 1
        if per_row is not None:
            per_row.append(frac)
        tick.tick()
    tick.done()

    def fin(s: dict) -> dict:
        s = dict(s)
        s["oov_frac"] = s["oov_chars"] / s["n_chars"] if s["n_chars"] else 0.0
        s["share_rows_over_threshold"] = s["rows_over_threshold"] / s["n_rows"] if s["n_rows"] else 0.0
        s["share_rows_with_any_oov"] = s["rows_with_any_oov"] / s["n_rows"] if s["n_rows"] else 0.0
        return s
    out = fin(total)
    out.update({"threshold": threshold, "normalized": table is not None, "rows_changed_by_normalizer": changed,
                "vocab_size": len(chars), "histogram": {k: hist[k] for k in ("0", "(0,0.1%]", "(0.1%,1%]", "(1%,5%]", ">5%")},
                "top_oov": [{"char": c, "codepoint": f"U+{ord(c):04X}", "name": unicodedata.name(c, ""), "count": n, "rows": char_rows[c]}
                            for c, n in sorted(char_count.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]],
                "n_distinct_oov_chars": len(char_count)})
    if group_key:
        out["per_group"] = {g: fin(s) for g, s in sorted(groups.items())}
    if per_row is not None:
        out["per_row_frac"] = per_row
    return out


def oov_filter(rows: Iterable[Any], vocab: Any, *, text_fn: Optional[Callable[[Any], str]] = None, threshold: float = 0.01,
               normalize: Union[bool, dict] = True) -> Iterator[Tuple[Any, float, bool]]:
    """Yield (row, oov_fraction, keep) with keep = fraction <= threshold (rows with MORE than 1 % are dropped)."""
    chars = _chars_of(vocab)
    table = RUNTIME_NORMALIZE if normalize is True else (normalize if isinstance(normalize, dict) else None)
    text_fn = text_fn or default_text
    for row in rows:
        text = text_fn(row)
        if table is not None:
            text = text.translate(table)
        frac, _ = oov_fraction(text, chars)
        yield row, frac, frac <= threshold


# =====================================================================================================================
# Length statistics
# =====================================================================================================================
def row_length(row: Any, text_fn: Optional[Callable[[Any], str]] = None) -> int:
    """len(prompt + target) + 1 (the +1 is the end token); a row needs <= MAX_LEN (2048)."""
    if text_fn is None and isinstance(row, dict) and isinstance(row.get("prompt"), str) and isinstance(row.get("target"), str):
        return len(row["prompt"]) + len(row["target"]) + 1
    return len((text_fn or default_text)(row)) + 1


def _percentile(items: List[Tuple[int, int]], total: int, q: float) -> int:
    rank, cum = max(1, math.ceil(q * total)), 0
    for v, c in items:
        cum += c
        if cum >= rank:
            return v
    return items[-1][0] if items else 0


def length_stats(rows: Iterable[Any], *, text_fn: Optional[Callable[[Any], str]] = None, group_key: Optional[Callable[[Any], str]] = None,
                 limit: int = MAX_LEN, soft_limits: Sequence[int] = (1900,), progress: Optional[ProgressCB] = None,
                 progress_every: int = 50000) -> dict:
    """Distribution of L = len(prompt+target)+1.  Per group (and 'ALL'): n, mean, min, max, p50/p90/p95/p99 (exact), over_2048 =
    rows with L > 2048 (cannot be trained), over_1900 (L > 1900), chars_over_2000 (len(text) > 2000, the plan's cap),
    their shares, full_text_mismatch (rows whose full_text != prompt + target) and a histogram in steps of 256."""
    counters: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    mism: collections.Counter = collections.Counter()
    tick = _Ticker(progress, "length_stats", progress_every)
    for row in rows:
        L = row_length(row, text_fn)
        names = ["ALL"] + ([str(group_key(row))] if group_key else [])
        bad = isinstance(row, dict) and isinstance(row.get("full_text"), str) and isinstance(row.get("prompt"), str) \
            and isinstance(row.get("target"), str) and row["full_text"] != row["prompt"] + row["target"]
        for n in names:
            counters[n][L] += 1
            mism[n] += bad
        tick.tick()
    tick.done()
    out = {}
    for name, cnt in sorted(counters.items()):
        items = sorted(cnt.items())
        n = sum(cnt.values())
        d = {"n": n, "mean": sum(v * c for v, c in items) / n if n else 0.0, "min": items[0][0] if items else 0, "max": items[-1][0] if items else 0}
        for q in (0.5, 0.9, 0.95, 0.99):
            d[f"p{int(q * 100)}"] = _percentile(items, n, q)
        over = sum(c for v, c in items if v > limit)
        d["over_2048"], d["share_over_2048"] = over, (over / n if n else 0.0)
        for s in soft_limits:
            o = sum(c for v, c in items if v > s)
            d[f"over_{s}"], d[f"share_over_{s}"] = o, (o / n if n else 0.0)
        o = sum(c for v, c in items if v > 2001)
        d["chars_over_2000"], d["share_chars_over_2000"] = o, (o / n if n else 0.0)
        d["full_text_mismatch"] = mism[name]
        hist: collections.Counter = collections.Counter()
        for v, c in items:
            hist[min(v // 256, 10)] += c
        d["histogram_256"] = {(f"{k * 256}-{k * 256 + 255}" if k < 10 else "2560+"): hist[k] for k in sorted(hist)}
        out[name] = d
    return out


# =====================================================================================================================
# Exact duplicates
# =====================================================================================================================
def _norm_key(text: str, mode: str) -> str:
    if mode == "none":
        return text
    if mode == "nfkc":
        text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


class ExactIndex:
    """digest -> first tag.  ~100 B per row; add() returns the tag of the earlier identical text or None."""

    def __init__(self, normalize: str = "ws"):
        self.normalize = normalize
        self._seen: Dict[int, Any] = {}

    def __len__(self) -> int:
        return len(self._seen)

    def key(self, text: str) -> int:
        return digest64(_norm_key(text, self.normalize))

    def add(self, text: str, tag: Any = None) -> Any:
        k = self.key(text)
        prev = self._seen.get(k)
        if prev is None:
            self._seen[k] = tag if tag is not None else True
            return None
        return prev

    def contains(self, text: str) -> Any:
        return self._seen.get(self.key(text))


def build_exact_index(rows: Iterable[Any], *, text_fn: Optional[Callable[[Any], str]] = None, normalize: str = "ws",
                      progress: Optional[ProgressCB] = None) -> ExactIndex:
    """Index of a (large) reference stream, e.g. the 423,348-row old corpus: ~45 MB, one streaming pass."""
    idx, text_fn = ExactIndex(normalize), text_fn or default_text
    tick = _Ticker(progress, "build_exact_index", 50000)
    for i, row in enumerate(rows):
        idx.add(text_fn(row), i)
        tick.tick()
    tick.done()
    return idx


def exact_dups(rows: Iterable[Any], *, text_fn: Optional[Callable[[Any], str]] = None, normalize: str = "ws",
               against: Optional[ExactIndex] = None, max_examples: int = 5, progress: Optional[ProgressCB] = None,
               progress_every: int = 50000) -> dict:
    """Exact duplicates (after whitespace normalisation; text_fn picks full_text / prompt / question ...).
    within_dup_positions: stream positions of rows equal to an EARLIER row of the same stream (keep-first);
    against_dup_positions: positions of rows already present in `against` (cross-dataset / vs-old-corpus)."""
    text_fn = text_fn or default_text
    own, within, cross, examples = ExactIndex(normalize), [], [], []
    n = 0
    tick = _Ticker(progress, "exact_dups", progress_every)
    for i, row in enumerate(rows):
        n += 1
        text = text_fn(row)
        first = own.add(text, i)
        if first is not None:
            within.append(i)
            if len(examples) < max_examples:
                examples.append({"position": i, "id": default_id(row, i), "first_position": first if first is not True else None, "text": text[:160]})
        if against is not None:
            hit = against.contains(text)
            if hit is not None:
                cross.append(i)
        tick.tick()
    tick.done()
    out = {"n_rows": n, "n_unique": len(own), "within_dups": len(within), "within_dup_rate": len(within) / n if n else 0.0,
           "within_dup_positions": within, "examples": examples}
    if against is not None:
        out.update({"against_dups": len(cross), "against_dup_rate": len(cross) / n if n else 0.0, "against_dup_positions": cross,
                    "against_index_rows": len(against)})
    return out


# =====================================================================================================================
# MinHash (k = 128 permutations, character 5-grams) + LSH banding (32 bands x 4 rows)
# =====================================================================================================================
_M32 = 0xFFFFFFFF


class MinHasher:
    """MinHash signatures of the character n-gram set of a text.
      * text -> NFKC, casefold, whitespace collapsed; shingles = distinct character n-grams (default 5); each shingle is hashed with
        zlib.crc32 over its UTF-32 bytes (32 bit);
      * permutation j: h_j(x) = (a_j * x + b_j) mod 2^32 with odd a_j (a bijection of the 32-bit space), a_j / b_j derived from
        sha256("checks.minhash:<seed>:<j>"); signature[j] = min over shingles of h_j(shingle).
    The expected share of equal signature slots of two texts is the Jaccard similarity of their shingle sets (standard error
    0.035 at J = 0.8 with k = 128).  numpy is used when importable (identical output; pass use_numpy=False for pure Python).

    HOW SHARP THE CUT IS (binomial, k = 128): a pair is reported when the share of equal slots is >= the cut.  With the cut at
    0.75 (the pipeline's choice: target 0.8 minus a 0.05 margin) a pair of true Jaccard 0.90 is found with probability ~1.00,
    0.85 ~0.999, 0.80 ~0.93, 0.75 ~0.55, 0.70 ~0.13, 0.65 ~0.01, 0.60 ~0.00.  With the cut at exactly the target the pair AT the target is
    found only half of the time, which is a property of any MinHash estimate, not a defect.  near_dups_within drops a row only when it
    is near a row that is kept (no chains: the survivor of every dropped row is itself near it)."""

    def __init__(self, k: int = 128, ngram: int = 5, seed: int = 20261109, use_numpy: Optional[bool] = None):
        self.k, self.ngram, self.seed = int(k), int(ngram), int(seed)
        a, b = [], []
        for j in range(self.k):
            d = hashlib.sha256(f"checks.minhash:{self.seed}:{j}".encode()).digest()
            a.append(int.from_bytes(d[0:4], "little") | 1)
            b.append(int.from_bytes(d[4:8], "little"))
        self._coeffs = list(zip(a, b))
        self._np = None
        if use_numpy is not False:
            try:
                import numpy as np
                self._np, self._A, self._B = np, np.array(a, dtype=np.uint64), np.array(b, dtype=np.uint64)
            except Exception:                            # noqa: BLE001
                if use_numpy:
                    raise

    @property
    def backend(self) -> str:
        return "numpy" if self._np is not None else "pure-python"

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(unicodedata.normalize("NFKC", text).casefold().split())

    def shingles(self, text: str) -> List[int]:
        s = self.normalize(text)
        n = self.ngram
        if not s:
            return []
        if len(s) < n:
            return [zlib.crc32(s.encode("utf-32-le", "surrogatepass"))]
        b = s.encode("utf-32-le", "surrogatepass")
        crc = zlib.crc32
        return list({crc(b[4 * i:4 * (i + n)]) for i in range(len(s) - n + 1)})

    def signature(self, text: str) -> Optional[array.array]:
        """array('I') of k slots, or None when the text has no shingles (empty after normalisation)."""
        xs = self.shingles(text)
        if not xs:
            return None
        if self._np is not None:
            np = self._np
            arr = np.asarray(xs, dtype=np.uint64)
            best = None
            mask = np.uint64(_M32)
            for s in range(0, len(arr), 4096):
                m = ((arr[s:s + 4096, None] * self._A[None, :] + self._B[None, :]) & mask).min(axis=0)
                best = m if best is None else np.minimum(best, m)
            sig = array.array("I")
            sig.frombytes(best.astype(np.uint32).tobytes())
            return sig
        return array.array("I", [min([(a * x + b) & _M32 for x in xs]) for a, b in self._coeffs])


class MinHashIndex:
    """Signatures of added rows + LSH tables (`bands` bands of k/bands slots; default 32 x 4 for k = 128: a pair with Jaccard 0.8
    collides in some band with probability ~1.0000, one with 0.7 with 0.9998, 0.6 with 0.988, 0.5 with 0.873; candidates are verified by
    the share of equal slots).
    Tables are sorted arrays (build() once after adding), queried with bisect; at most `max_per_band` earliest rows per bucket
    are inspected (a bucket that large is boilerplate: its earliest members stand for it)."""

    def __init__(self, hasher: Optional[MinHasher] = None, bands: int = 32, snippet_chars: int = 120):
        self.hasher = hasher or MinHasher()
        self.k = self.hasher.k
        if self.k % bands:
            raise ValueError("k must be a multiple of bands")
        self.bands, self.r = bands, self.k // bands
        self.sigs = array.array("I")
        self.valid = bytearray()
        self.labels: List[Any] = []
        self.tags: List[Any] = []
        self.snippet_chars = snippet_chars
        self.snippets: List[str] = []
        self._tables: List[Tuple[array.array, array.array]] = []
        self._built = False

    def __len__(self) -> int:
        return len(self.valid)

    def add_signature(self, sig: Optional[array.array], label: Any = None, tag: Any = None, snippet: str = "") -> int:
        if self._built:
            raise RuntimeError("index already built; create a new MinHashIndex to add rows")
        pos = len(self.valid)
        if sig is None:
            self.sigs.extend([0] * self.k)
            self.valid.append(0)
        else:
            self.sigs.extend(sig)
            self.valid.append(1)
        self.labels.append(label)
        self.tags.append(tag)
        if self.snippet_chars:
            self.snippets.append(snippet[:self.snippet_chars])
        return pos

    def add_text(self, text: str, label: Any = None, tag: Any = None) -> int:
        return self.add_signature(self.hasher.signature(text), label, tag, text if self.snippet_chars else "")

    def sig_at(self, pos: int) -> array.array:
        return self.sigs[pos * self.k:(pos + 1) * self.k]

    def build(self) -> "MinHashIndex":
        n, k4, r4 = len(self.valid), self.k * 4, self.r * 4
        ids = [i for i in range(n) if self.valid[i]]
        self._tables = []
        raw = memoryview(self.sigs).cast("B")
        try:
            for b in range(self.bands):
                off = b * r4
                keys = [zlib.crc32(raw[i * k4 + off:i * k4 + off + r4]) for i in ids]
                order = sorted(range(len(ids)), key=keys.__getitem__)                 # stable: ids ascend inside a bucket
                self._tables.append((array.array("I", (keys[o] for o in order)), array.array("I", (ids[o] for o in order))))
        finally:
            raw.release()
        self._built = True
        return self

    def agreement(self, sig: array.array, pos: int) -> float:
        other = self.sigs[pos * self.k:(pos + 1) * self.k]
        return sum(map(operator.eq, sig, other)) / self.k

    def query_sig(self, sig: array.array, threshold: float = 0.8, *, below: Optional[int] = None, exclude: Optional[int] = None,
                  max_per_band: int = 64) -> List[Tuple[int, float]]:
        """[(position, estimated Jaccard)] of indexed rows with agreement >= threshold; below=i only looks at rows < i."""
        if not self._built:
            raise RuntimeError("call build() before querying")
        raw, r4, cand = sig.tobytes(), self.r * 4, set()
        for b, (sk, sid) in enumerate(self._tables):
            key = zlib.crc32(raw[b * r4:(b + 1) * r4])
            j, m, cnt = bisect.bisect_left(sk, key), len(sk), 0
            while j < m and sk[j] == key and cnt < max_per_band:
                pid = sid[j]
                if below is not None and pid >= below:
                    break
                cand.add(pid)
                j += 1
                cnt += 1
        out = []
        for pid in sorted(cand):
            if pid == exclude:
                continue
            est = self.agreement(sig, pid)
            if est >= threshold:
                out.append((pid, est))
        return out


def near_dups_within(rows: Iterable[Any], *, text_fn: Optional[Callable[[Any], str]] = None, id_fn: Optional[Callable[[Any, int], str]] = None,
                     threshold: float = 0.8, k: int = 128, bands: int = 32, max_per_band: int = 64, max_examples: int = 5,
                     snippet_chars: int = 120, hasher: Optional[MinHasher] = None, progress: Optional[ProgressCB] = None,
                     progress_every: int = 20000) -> dict:
    """Near-duplicate clusters inside one stream, greedy keep-first: rows are taken in order and a row is dropped (listed in
    dup_positions) only when it is similar (estimated Jaccard >= threshold) to a row that is KEPT; it is then attached to the most
    similar kept row (root_of[i], the survivor).  There are no chains: a~b~c with J(a, c) < threshold keeps a and c (an earlier
    version used connected components, where a chain of 31 near pairs removed rows with Jaccard ~0 to the survivor).  Two survivors can
    still be near each other when the LSH or the max_per_band cap misses the pair.  cluster_size_hist says how many clusters (a survivor
    plus its dropped rows) of each size there are (sizes above 9 are pooled as '10+') and largest_cluster the biggest.  `threshold` is
    applied to the ESTIMATE (see MinHasher for how sharp that is).
    Memory: see the module docstring (~1.2 KB per row at k = 128)."""
    text_fn, id_fn = text_fn or default_text, id_fn or default_id
    idx = MinHashIndex(hasher or MinHasher(k=k), bands=bands, snippet_chars=snippet_chars)
    tick = _Ticker(progress, "minhash signatures", progress_every)
    for i, row in enumerate(rows):
        idx.add_text(text_fn(row), id_fn(row, i))
        tick.tick()
    tick.done()
    idx.build()
    n = len(idx)
    roots = list(range(n))                                            # roots[i] == i: kept; else the kept row that i duplicates
    examples, tick = [], _Ticker(progress, "minhash query", progress_every)
    for i in range(n):
        tick.tick()
        if not idx.valid[i]:
            continue
        best = None
        for j, est in idx.query_sig(idx.sig_at(i), threshold, below=i, max_per_band=max_per_band):
            if roots[j] == j and (best is None or est > best[1]):         # candidates ascend by position: ties keep the earliest
                best = (j, est)
        if best is not None:
            j, est = best
            roots[i] = j
            if len(examples) < max_examples:
                examples.append({"position": i, "id": idx.labels[i], "similar_to_position": j, "similar_to_id": idx.labels[j],
                                 "est_jaccard": round(est, 3), "text": idx.snippets[i] if idx.snippets else "",
                                 "other_text": idx.snippets[j] if idx.snippets else ""})
    tick.done()
    dup_positions = [i for i in range(n) if roots[i] != i]
    sizes = collections.Counter(roots)
    size_hist = collections.Counter(("10+" if c >= 10 else str(c)) for c in sizes.values() if c > 1)
    return {"n_rows": n, "n_rows_without_shingles": n - sum(idx.valid), "threshold": threshold, "k": idx.k, "bands": bands,
            "cluster_size_hist": {key: size_hist[key] for key in sorted(size_hist, key=lambda x: (len(x), x))},
            "hasher_backend": idx.hasher.backend, "n_dups": len(dup_positions), "dup_rate": len(dup_positions) / n if n else 0.0,
            "dup_positions": dup_positions, "root_of": roots, "n_clusters_gt1": sum(1 for r, c in sizes.items() if c > 1),
            "largest_cluster": max(sizes.values()) if sizes else 0, "examples": examples}


def cross_near_dups(index_rows: Iterable[Any], probe_rows: Iterable[Any], *, text_fn: Optional[Callable[[Any], str]] = None,
                    probe_text_fn: Optional[Callable[[Any], str]] = None, id_fn: Optional[Callable[[Any, int], str]] = None,
                    threshold: float = 0.8, k: int = 128, bands: int = 32, max_per_band: int = 64, sample_every: int = 1,
                    max_examples: int = 5, snippet_chars: int = 120, progress: Optional[ProgressCB] = None,
                    progress_every: int = 20000) -> dict:
    """New rows (index side, held in memory, <= ~300k) against a big reference stream (probe side, streamed once, never held -
    e.g. the 423,348-row old corpus).  Returns how many index rows have a near duplicate in the probe stream
    (index_rows_with_match / index_match_rate; matched_index_positions to filter them) and how many probe rows matched.
    sample_every=N inspects only every N-th probe row (probe position % N == 0): the index match rate is then a LOWER bound
    (about 1/N of the true matches are seen) while probe_match_rate stays an unbiased estimate."""
    text_fn = text_fn or default_text
    probe_text_fn = probe_text_fn or text_fn
    id_fn = id_fn or default_id
    idx = MinHashIndex(MinHasher(k=k), bands=bands, snippet_chars=snippet_chars)
    tick = _Ticker(progress, "cross: index signatures", progress_every)
    for i, row in enumerate(index_rows):
        idx.add_text(text_fn(row), id_fn(row, i))
        tick.tick()
    tick.done()
    idx.build()
    matched, scanned, probe_hits, examples = set(), 0, 0, []
    tick = _Ticker(progress, "cross: probe stream", progress_every)
    for pi, row in enumerate(probe_rows):
        tick.tick()
        if sample_every > 1 and pi % sample_every:
            continue
        text = probe_text_fn(row)
        sig = idx.hasher.signature(text)
        if sig is None:
            continue
        scanned += 1
        hits = idx.query_sig(sig, threshold, max_per_band=max_per_band)
        if hits:
            probe_hits += 1
            matched.update(j for j, _ in hits)
            if len(examples) < max_examples:
                j, est = max(hits, key=lambda h: h[1])
                examples.append({"probe_position": pi, "probe_id": id_fn(row, pi), "probe_text": text[:snippet_chars or 120],
                                 "index_position": j, "index_id": idx.labels[j], "index_text": idx.snippets[j] if idx.snippets else "",
                                 "est_jaccard": round(est, 3)})
    tick.done()
    n_idx = sum(idx.valid)
    return {"index_rows": len(idx), "index_rows_with_match": len(matched), "index_match_rate": len(matched) / n_idx if n_idx else 0.0,
            "probe_rows_scanned": scanned, "probe_rows_with_match": probe_hits, "probe_match_rate": probe_hits / scanned if scanned else 0.0,
            "sample_every": sample_every, "index_rate_is_lower_bound": sample_every > 1, "threshold": threshold,
            "matched_index_positions": sorted(matched), "examples": examples}


def cross_dataset_near_dups(named_rows: Mapping[str, Iterable[Any]], *, text_fn: Optional[Callable[[Any], str]] = None,
                            threshold: float = 0.8, k: int = 128, bands: int = 32, max_per_band: int = 64, snippet_chars: int = 80,
                            progress: Optional[ProgressCB] = None, progress_every: int = 20000, deadline: Optional[float] = None,
                            info: Optional[dict] = None) -> dict:
    """Dataset x dataset duplicate matrix.  rows_with_match_in[A][B] = number of rows of A that have a near duplicate (estimated
    Jaccard >= threshold) among the rows of B; for A == B it is the number of rows that belong to some duplicate pair inside A
    (both members count).  All datasets share one index (~1.2 KB per row in total).
    deadline = an absolute time.time() value: when it passes, the signature phase or the query phase stops and info["truncated"]
    (a dict passed by the caller) says which; the matrix then only covers the rows handled so far."""
    text_fn = text_fn or default_text
    idx = MinHashIndex(MinHasher(k=k), bands=bands, snippet_chars=snippet_chars)
    names = list(named_rows)
    late = lambda: deadline is not None and time.time() > deadline            # noqa: E731
    stopped = False
    for ds, rows in named_rows.items():
        tick = _Ticker(progress, f"signatures[{ds}]", progress_every)
        for i, row in enumerate(rows):
            if i % 512 == 0 and late():
                stopped = True
                break
            idx.add_text(text_fn(row), default_id(row, i), tag=ds)
            tick.tick()
        tick.done()
        if stopped:
            break
    if stopped and info is not None:
        info["truncated"] = "signatures"
    idx.build()
    n_rows = collections.Counter(t for t, v in zip(idx.tags, idx.valid) if v)
    matrix = {a: collections.Counter() for a in names}
    examples: List[dict] = []
    tick = _Ticker(progress, "cross-dataset query", progress_every)
    for i in range(len(idx)):
        tick.tick()
        if i % 512 == 0 and late():
            if info is not None:
                info.setdefault("truncated", "query")
                info["queried_rows"] = i
            break
        if not idx.valid[i]:
            continue
        hits = idx.query_sig(idx.sig_at(i), threshold, exclude=i, max_per_band=max_per_band)
        for other in {idx.tags[j] for j, _ in hits}:
            matrix[idx.tags[i]][other] += 1
        for j, est in hits:
            if idx.tags[j] != idx.tags[i] and len(examples) < 5:
                examples.append({"a": idx.tags[i], "a_id": idx.labels[i], "a_text": idx.snippets[i], "b": idx.tags[j], "b_id": idx.labels[j],
                                 "b_text": idx.snippets[j], "est_jaccard": round(est, 3)})
                break
    tick.done()
    return {"threshold": threshold, "rows": dict(n_rows), "rows_with_match_in": {a: dict(c) for a, c in matrix.items()}, "examples": examples}


# =====================================================================================================================
# Eval files, eval index, decontamination (digit-stripped 13-gram character shingles)
# =====================================================================================================================
class EvalFilesMissing(FileNotFoundError):
    """A required eval file could not be found (or an eval file is unreadable)."""


@dataclasses.dataclass(frozen=True)
class EvalSource:
    label: str
    chain: Tuple[str, ...]            # (file,) or (zip, member[, nested member...]); nested zips are read through memory
    kind: str                         # 'jsonl' | 'json' | 'py'
    required: bool = True
    description: str = ""

    @property
    def display(self) -> str:
        return "::".join(self.chain)


def _guess_kind(name: str) -> str:
    low = name.lower()
    return "jsonl" if low.endswith((".jsonl", ".ndjson")) else ("py" if low.endswith(".py") else "json")


def _read_chain(chain: Sequence[str]) -> bytes:
    if len(chain) == 1:
        with open(chain[0], "rb") as f:
            return f.read()
    z = zipfile.ZipFile(chain[0])
    try:
        for member in chain[1:-1]:
            inner = zipfile.ZipFile(io.BytesIO(z.read(member)))
            z.close()
            z = inner
        return z.read(chain[-1])
    finally:
        z.close()


def eval_item_strings(obj: Any, text_keys: Sequence[str] = EVAL_TEXT_KEYS) -> List[str]:
    """The text fields of one eval record (each choice is its own string, so 13-grams never span fields)."""
    out: List[str] = []
    if isinstance(obj, dict):
        for k in text_keys:
            v = obj.get(k)
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, (list, tuple)):
                out.extend(x for x in v if isinstance(x, str))
        if not out:
            out = [s for s in _string_leaves(obj) if len(s) >= 13]
    elif isinstance(obj, str):
        out.append(obj)
    return out


def iter_eval_items(src: Union[EvalSource, str], text_keys: Sequence[str] = EVAL_TEXT_KEYS) -> Iterator[List[str]]:
    """Items (list of strings) of an eval source.  jsonl/json: the EVAL_TEXT_KEYS fields of every record;
    py: every string literal of >= 20 characters (the Tier-8 prompts are literals in tier8_readonly_v44fix.py)."""
    if isinstance(src, str):
        src = EvalSource(os.path.basename(src), (src,), _guess_kind(src))
    data = _read_chain(src.chain)
    if src.kind == "py":
        for node in ast.walk(ast.parse(data.decode("utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) >= 20:
                yield [node.value]
    elif src.kind == "jsonl":
        for line in data.decode("utf-8-sig").splitlines():
            if line.strip():
                items = eval_item_strings(json.loads(line), text_keys)
                if items:
                    yield items
    else:
        obj = json.loads(data.decode("utf-8-sig"))
        for rec in (obj if isinstance(obj, list) else (obj.get("data", [obj]) if isinstance(obj, dict) else [obj])):
            items = eval_item_strings(rec, text_keys)
            if items:
                yield items


@dataclasses.dataclass
class EvalFiles:
    sources: List[EvalSource] = dataclasses.field(default_factory=list)
    missing_required: List[str] = dataclasses.field(default_factory=list)
    missing_optional: List[str] = dataclasses.field(default_factory=list)
    notes: List[str] = dataclasses.field(default_factory=list)

    def report(self) -> str:
        lines = [f"{len(self.sources)} eval sources"]
        lines += [f"  {'R' if s.required else 'o'} {s.label}: {s.display}" for s in self.sources]
        lines += [f"  MISSING (required): {m}" for m in self.missing_required]
        lines += [f"  missing (optional): {m}" for m in self.missing_optional]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


def _index_loose(roots: Sequence[str], max_depth: int) -> Dict[str, str]:
    """name -> path of every stand-alone file under the roots (depth-limited walk, first root / first path wins).  One pass."""
    out: Dict[str, str] = {}
    for root in roots:
        if os.path.isdir(root):
            for dirpath, files in _walk_limited(root, max_depth):
                for fn in files:
                    out.setdefault(fn, os.path.join(dirpath, fn))
    return out


def _glob_fallback(name: str, roots: Sequence[str]) -> Optional[str]:
    """Recursive glob (slow on a Drive mount; also looks into the folders the walk skips).  Only used for names that the
    index and the bundle zips did not provide."""
    for root in roots:
        hits = sorted(_glob.glob(os.path.join(root, "**", name), recursive=True)) if os.path.isdir(root) else []
        if hits:
            return hits[0]
    return None


def locate_registry(name: str, loose: Mapping[str, str], bundles: Sequence[tuple], members: Mapping[str, Tuple[str, str]]) -> Optional[Tuple[str, ...]]:
    """Where an EVAL_REGISTRY_EXTRA file is: a stand-alone file, the bundle's nested data/ folder, or a suite zip member (by file name)."""
    if name in loose:
        return (loose[name],)
    for zp, outer, nested in bundles:
        if nested is not None and f"data/{name}" in nested:
            return (zp, EVAL_NESTED_ZIP, f"data/{name}")
    return members.get(name)


def find_eval_files(drive_root: str = DRIVE_ROOT, *, extra_dirs: Sequence[str] = (), strict: bool = True, max_depth: int = 4) -> EvalFiles:
    """Locate the eval texts the training rows must not overlap.  For every name in EVAL_REQUIRED_JSONL / EVAL_REQUIRED_PY:
    a stand-alone file under drive_root / extra_dirs wins; otherwise the file is taken from the Drive bundle
    GeoCPU_v47_fullmodel_bundle_v1.zip (-> assets/r6_bundle.zip -> data/<name>; code/<name> for .py); otherwise a recursive glob
    is tried.  Optional: the v48 unseen_eval/dev jsonl (EVAL_OPTIONAL_V48), the other frozen suites of the owner's registry
    (EVAL_REGISTRY_EXTRA: python120, engineering64, visual10, the new_public dev sets ... as stand-alone files, in the bundle or in the
    suite zips; the ones not found are listed in missing_optional) and the .jsonl members of the small suite zips whose names contain
    eval / heldout / manifest / bench.  strict=True raises EvalFilesMissing, naming every missing required file."""
    ef = EvalFiles()
    roots = [drive_root] + [d for d in extra_dirs]
    bundles = []                                           # (zip path, outer names, nested names or None)
    for zp in _bundle_zip_paths(drive_root) if os.path.isdir(drive_root) else []:
        try:
            with zipfile.ZipFile(zp) as z:
                outer = set(z.namelist())
                nested = None
                if EVAL_NESTED_ZIP in outer:
                    with zipfile.ZipFile(io.BytesIO(z.read(EVAL_NESTED_ZIP))) as inner:
                        nested = set(inner.namelist())
                bundles.append((zp, outer, nested))
        except (OSError, zipfile.BadZipFile) as e:
            ef.notes.append(f"unreadable bundle {zp}: {e}")

    loose = _index_loose(roots, max_depth)

    def locate(name: str, folder: str) -> Optional[Tuple[str, ...]]:
        if name in loose:
            return (loose[name],)
        for zp, outer, nested in bundles:
            if nested is not None and f"data/{name}" in nested:
                return (zp, EVAL_NESTED_ZIP, f"data/{name}")
            if f"{folder}/{name}" in outer:
                return (zp, f"{folder}/{name}")
        p = _glob_fallback(name, roots)
        return (p,) if p else None
    for table, folder, kind in ((EVAL_REQUIRED_JSONL, "data", "jsonl"), (EVAL_REQUIRED_PY, "code", "py")):
        for name, desc in table.items():
            chain = locate(name, folder)
            if chain is None:
                ef.missing_required.append(f"{name} ({desc})")
            else:
                ef.sources.append(EvalSource(label=name, chain=chain, kind=kind, required=True, description=desc))
    for name in EVAL_OPTIONAL_V48:
        p = loose.get(name)
        if p:
            ef.sources.append(EvalSource(label="v48:" + name, chain=(p,), kind="jsonl", required=False, description="v48 additive eval/dev set"))
        else:
            ef.missing_optional.append("v48:" + name)
    if ef.missing_optional:
        ef.notes.append("v48 unseen_eval/dev jsonl are not on Drive as stand-alone files (local copy: research/v48_balanced_augmentation_"
                        "20261108/data/); copy them under the Drive root or pass extra_dirs to include them")
    pat = re.compile(r"(?i)(eval|heldout|manifest|bench).*\.jsonl$")
    suite_zips = [loose[zn] for zn in EVAL_OPTIONAL_SUITE_ZIPS if zn in loose]
    suite_zips += [pth for name, pth in sorted(loose.items()) if re.fullmatch(r"(?i)GeoCPU_v47_.*eval.*\.zip", name) and pth not in suite_zips]
    members: Dict[str, Tuple[str, str]] = {}                 # basename -> (zip, member) for the registry names found inside suite zips
    for zp in suite_zips:
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if base in EVAL_REGISTRY_EXTRA and info.file_size <= 50 << 20:
                        members.setdefault(base, (zp, info.filename))
        except (OSError, zipfile.BadZipFile) as e:
            ef.notes.append(f"unreadable suite zip {zp}: {e}")
    for name, desc in EVAL_REGISTRY_EXTRA.items():
        chain = locate_registry(name, loose, bundles, members)
        if chain is None:
            ef.missing_optional.append("registry:" + name + f" ({desc})")
        else:
            ef.sources.append(EvalSource(label="registry:" + name, chain=chain, kind=_guess_kind(name), required=False, description=desc))
    for zp in suite_zips:
        try:
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    base = os.path.basename(info.filename)
                    if pat.search(base) and base not in EVAL_REGISTRY_EXTRA and info.file_size <= 20 << 20:
                        ef.sources.append(EvalSource(label=f"suite:{os.path.basename(zp)}:{base}", chain=(zp, info.filename),
                                                     kind="jsonl", required=False, description="eval member of a suite zip"))
        except (OSError, zipfile.BadZipFile):
            pass                                              # already noted above
    ef.sources.sort(key=lambda s: (not s.required, s.label))
    if ef.missing_required and strict:
        raise EvalFilesMissing("required eval files not found under " + ", ".join(roots) + ":\n  " + "\n  ".join(ef.missing_required)
                               + "\n(searched stand-alone files, " + ", ".join(VOCAB_BUNDLE_ZIPS) + " -> " + EVAL_NESTED_ZIP
                               + " -> data/, and a recursive glob)")
    return ef


_STRIP_DIGITS_WS = re.compile(r"[\d\s]+")


WHOLE_ITEM_MIN_CHARS = 40          # an eval text of at least this many normalised characters is a whole item: a row that contains it is flagged


def normalize_for_overlap(text: str) -> str:
    """The training loader's symbol map (RUNTIME_NORMALIZE: U+2212 -> '-', Greek capitals -> Latin, brackets, ...), NFKC, casefold, every
    decimal digit (any script) and every whitespace character removed: '12冊で' and '15冊で' become equal, and text that the loader turns
    into identical tokens is identical here too (a superset of what the loader does)."""
    t = unicodedata.normalize("NFKC", text.translate(RUNTIME_NORMALIZE)).translate(RUNTIME_NORMALIZE)
    return _STRIP_DIGITS_WS.sub("", t.casefold())


class EvalIndex:
    """Set of the digit-stripped n-gram (default 13) shingles of all eval texts + the normalised eval texts (for locating examples).
    Shingles are held as python hash() ints of the n-gram strings: the process-local seed is irrelevant because nothing hashed is
    persisted or reported (shingle strings are); the chance that one false match occurs in a whole run is ~1e-4.
    Per source: items, texts, indexable_texts (>= n normalised characters), shingles (distinct shingles the source CONTAINS, summed over its
    texts - not only the ones new to the index: a source that repeats another one still holds its own), new_shingles (the ones it added).
    `whole`: the eval texts of >= whole_min_chars normalised characters, keyed by the hash of their first n-gram (see whole_matches)."""

    def __init__(self, ngram: int = 13, whole_min_chars: int = WHOLE_ITEM_MIN_CHARS):
        self.ngram = ngram
        self.whole_min_chars = max(int(whole_min_chars), ngram)
        self.whole: Dict[int, List[Tuple[int, int]]] = {}       # first-shingle hash -> [(last-shingle hash, doc id)]
        self._whole_seen: set = set()
        self.hashes: set = set()
        self.docs: List[Tuple[str, int, str]] = []        # (source label, item index, normalised text)
        self.per_source: Dict[str, Dict[str, int]] = {}
        # shingle -> number of distinct eval ITEMS (label, item) that contain it; only shingles of >= 2 items are stored (absent = 1).
        # This, not the number of training rows, decides what is boilerplate: a shingle that many eval items share is a template
        # phrase, a shingle that one eval item owns is content - however many training rows copy it.
        self.eval_df: Dict[int, int] = {}
        self._cur_key: Optional[Tuple[str, int]] = None
        self._cur_seen: set = set()

    def __len__(self) -> int:
        return len(self.hashes)

    def add_text(self, label: str, item: int, text: str) -> int:
        norm = normalize_for_overlap(text)
        n = self.ngram
        st = self.per_source.setdefault(label, {"items": 0, "texts": 0, "indexable_texts": 0, "shingles": 0, "new_shingles": 0})
        st["texts"] += 1
        if len(norm) < n:
            return 0
        st["indexable_texts"] += 1
        hset = {hash(norm[i:i + n]) for i in range(len(norm) - n + 1)}
        if (label, item) != self._cur_key:                # texts of ONE item (question + choices) count once per shingle
            self._cur_key, self._cur_seen = (label, item), set()
        fresh = hset - self._cur_seen
        self._cur_seen |= hset
        before = len(self.hashes)
        for h in fresh:
            if h in self.hashes:
                self.eval_df[h] = self.eval_df.get(h, 1) + 1
        self.hashes.update(fresh)
        doc_id = len(self.docs)
        self.docs.append((label, item, norm))
        if len(norm) >= self.whole_min_chars:             # a whole item: findable by (first n-gram, last n-gram) whatever the df of its shingles
            key = hash(norm)
            if key not in self._whole_seen:               # items that differ only in digits have the same normalised text: one entry
                self._whole_seen.add(key)
                self.whole.setdefault(hash(norm[:n]), []).append((hash(norm[-n:]), doc_id))
        st["shingles"] += len(hset)
        st["new_shingles"] += len(self.hashes) - before
        return len(self.hashes) - before

    def whole_matches(self, norm: str, hset: set, inter: set, limit: int = 1) -> List[Tuple[str, int, str]]:
        """[(source, item, eval text)] of whole eval texts (>= whole_min_chars normalised characters) that occur verbatim inside the
        normalised row text `norm`.  hset = the row's n-gram hashes, inter = the ones that are also eval shingles.  A verbatim copy of an
        eval item is contamination however many other items share its template phrases (their shingles may all be boilerplate)."""
        out: List[Tuple[str, int, str]] = []
        if not self.whole:
            return out
        for h in inter & self.whole.keys():
            for last, doc in self.whole[h]:
                if last in hset:
                    label, item, text = self.docs[doc]
                    if text in norm:
                        out.append((label, item, text))
                        if len(out) >= limit:
                            return out
        return out

    def find_sources(self, shingle: str, limit: int = 3) -> List[dict]:
        out = []
        for label, item, norm in self.docs:
            p = norm.find(shingle)
            if p >= 0:
                out.append({"source": label, "item": item, "context": norm[max(0, p - 25):p + len(shingle) + 25]})
                if len(out) >= limit:
                    break
        return out

    def stats(self) -> dict:
        return {"ngram": self.ngram, "distinct_shingles": len(self.hashes), "normalised_texts": len(self.docs), "per_source": self.per_source,
                "shingles_in_several_eval_items": len(self.eval_df), "max_eval_items_per_shingle": max(self.eval_df.values(), default=1),
                "whole_item_texts": len(self._whole_seen), "whole_item_min_chars": self.whole_min_chars}


def build_eval_index(sources: Iterable[Union[EvalSource, str, Tuple[str, Iterable[str]]]], *, ngram: int = 13,
                     text_keys: Sequence[str] = EVAL_TEXT_KEYS, progress: Optional[ProgressCB] = None) -> EvalIndex:
    """EvalIndex over EvalSource objects (find_eval_files(...).sources), plain file paths, or (label, iterable of texts) pairs."""
    idx = EvalIndex(ngram)
    for src in sources:
        if isinstance(src, tuple) and len(src) == 2 and not isinstance(src[0], EvalSource) and not isinstance(src[1], str):
            label, texts = src
            for i, t in enumerate(texts):
                idx.add_text(label, i, t)
                idx.per_source[label]["items"] += 1
            continue
        s = src if isinstance(src, EvalSource) else EvalSource(os.path.basename(src), (src,), _guess_kind(src))
        try:
            for i, strings in enumerate(iter_eval_items(s, text_keys)):
                for t in strings:
                    idx.add_text(s.label, i, t)
                idx.per_source[s.label]["items"] += 1
        except (OSError, ValueError, zipfile.BadZipFile, KeyError, SyntaxError) as e:
            raise EvalFilesMissing(f"cannot read eval source {s.display}: {e}") from e
        if progress:
            progress(f"eval index: {s.label}", len(idx))
    return idx


def decontaminate(rows: Iterable[Any], index: EvalIndex, *, text_fn: Optional[Callable[[Any], str]] = None,
                  id_fn: Optional[Callable[[Any, int], str]] = None, boilerplate_eval_df: int = 20, min_shared: int = 1,
                  max_examples: int = 5, row_prefix_chars: int = 160, progress: Optional[ProgressCB] = None,
                  progress_every: int = 20000) -> dict:
    """Flag rows that share a digit-stripped n-gram (13) with any eval text.  A shingle is BOILERPLATE (instruction templates, 'which
    of the following ...') when it occurs in at least `boilerplate_eval_df` (20) distinct EVAL ITEMS - the count is taken on the eval
    side (EvalIndex.eval_df), never on the training side: an eval question owned by one item has eval df 1, so it flags every row that
    copies it however many rows do (a train-side count would wave a widely copied leak through).  Boilerplate is reported and never
    flags a row.  A row is flagged when it shares >= min_shared non-boilerplate shingles, OR when it contains a WHOLE eval text of
    >= index.whole_min_chars (40) normalised characters verbatim (rows_flagged_whole_item): eval items that differ only in digits
    (T1, T6, the v48 unseen_eval sets) share every shingle, so all their shingles are boilerplate by the df rule and a verbatim copy
    of a whole item would otherwise pass; copying a template phrase alone (a part of the item) is still only boilerplate.  Two limits of
    this rule that were measured and are NOT fixed (the thresholds 13 / 40 / 20 stay: false positives cannot be measured without the
    real rows): an eval text of 13-39 normalised characters whose 13-grams are ALL boilerplate can be copied verbatim without being
    flagged (the v33 prompt template, 50 items, and one v42 question), and a template phrase that is itself an eval text of 40+
    characters flags EVERY row that contains it - read rows_flagged_whole_item per dataset before trusting the drop counts.
    One streaming pass.  Returns n_rows, rows_with_any_shared_shingle, rows_boilerplate_only, rows_flagged, flagged_positions (stream
    positions, to drop with drop_flagged), shared_count_hist, boilerplate {n_shingles, top[20]: shingle, eval_items, rows},
    real_shingles_top (the non-boilerplate shingles that most rows share: a big number means one eval text is copied widely),
    examples (<= max_examples: id, shared shingles, the row's first characters, where the shingle sits in the eval text) and the index
    stats.  Limits: exact digit-stripped substring overlap only - paraphrases, translations and re-worded numbers are NOT caught (see
    semantic_neighbors for a small look-alike search)."""
    text_fn, id_fn = text_fn or default_text, id_fn or default_id
    ng, ev, eval_df = index.ngram, index.hashes, index.eval_df
    df: collections.Counter = collections.Counter()
    shingle_text: Dict[int, str] = {}
    hits: Dict[int, Tuple[str, Tuple[int, ...], str]] = {}
    whole_pos: Dict[int, Tuple[str, int, str]] = {}          # position -> (source, item, whole eval text) of rows that contain a whole eval item
    n_rows = n_short = rows_any = 0
    tick = _Ticker(progress, "decontaminate", progress_every)
    for pos, row in enumerate(rows):
        tick.tick()
        n_rows += 1
        text = text_fn(row)
        norm = normalize_for_overlap(text)
        if len(norm) < ng:
            n_short += 1
            continue
        hset = {hash(norm[i:i + ng]) for i in range(len(norm) - ng + 1)}
        inter = hset & ev
        if not inter:
            continue
        rows_any += 1
        first: Dict[int, int] = {}
        for i in range(len(norm) - ng + 1):
            h = hash(norm[i:i + ng])
            if h in inter and h not in first:
                first[h] = i
        real = []
        for h in inter:
            df[h] += 1
            if h not in shingle_text:
                shingle_text[h] = norm[first[h]:first[h] + ng]
            if eval_df.get(h, 1) < boilerplate_eval_df:
                real.append(h)
        whole = index.whole_matches(norm, hset, inter)
        if whole:
            whole_pos[pos] = whole[0]
        if real or whole:
            hits[pos] = (id_fn(row, pos), tuple(sorted(real, key=lambda h: shingle_text[h])), text[:row_prefix_chars])
    tick.done()
    boiler = {h for h in df if eval_df.get(h, 1) >= boilerplate_eval_df}
    flagged: Dict[int, List[int]] = {}
    for pos, (rid, hs, prefix) in hits.items():
        if len(hs) >= min_shared or pos in whole_pos:
            flagged[pos] = list(hs)
    any_real = len(hits)
    hist = collections.Counter()
    for pos, real in flagged.items():
        c = len(real)
        hist["whole_item_only" if c == 0 else ("1" if c == 1 else ("2-4" if c <= 4 else ("5-19" if c <= 19 else "20+")))] += 1
    examples = []
    for pos in sorted(flagged, key=lambda p: (-len(flagged[p]), p))[:max_examples]:
        rid, _, prefix = hits[pos]
        shingles = sorted(shingle_text[h] for h in flagged[pos])
        whole_text = whole_pos[pos][2] if pos in whole_pos else None
        probe = shingles[0] if shingles else whole_text[:ng]
        examples.append({"position": pos, "id": rid, "n_shared_shingles": len(shingles), "shared_shingles": shingles[:3],
                         "contains_whole_eval_item": pos in whole_pos, "row_prefix": prefix, "eval_matches": index.find_sources(probe)})
    top = sorted(((shingle_text[h], eval_df.get(h, 1), df[h]) for h in boiler), key=lambda t: (-t[2], t[0]))
    real_top = sorted(((shingle_text[h], eval_df.get(h, 1), d) for h, d in df.items() if h not in boiler), key=lambda t: (-t[2], t[0]))
    return {"n_rows": n_rows, "n_rows_shorter_than_ngram": n_short, "ngram": ng, "boilerplate_eval_df": boilerplate_eval_df, "min_shared": min_shared,
            "rows_with_any_shared_shingle": rows_any, "rows_boilerplate_only": rows_any - any_real,
            "rows_with_real_overlap": any_real, "rows_flagged": len(flagged), "flagged_positions": sorted(flagged),
            "rows_flagged_whole_item": len(whole_pos), "whole_item_positions": sorted(whole_pos), "whole_item_min_chars": index.whole_min_chars,
            "shared_count_hist": {k: hist[k] for k in ("1", "2-4", "5-19", "20+", "whole_item_only")},
            "boilerplate": {"n_shingles": len(boiler), "top": [{"shingle": sh, "eval_items": e, "rows": d} for sh, e, d in top[:20]]},
            "real_shingles_top": [{"shingle": sh, "eval_items": e, "rows": d} for sh, e, d in real_top[:10]],
            "examples": examples, "eval_index": index.stats(),
            "limits": "exact digit-stripped 13-gram overlap only; paraphrases/translations not detected"}


def drop_flagged(rows: Iterable[Any], flagged_positions: Iterable[int]) -> Iterator[Any]:
    """Second streaming pass: the rows whose stream position is not in flagged_positions."""
    drop = set(flagged_positions)
    return (r for i, r in enumerate(rows) if i not in drop)


# =====================================================================================================================
# Semantic look-alike hook (character bigram TF-IDF cosine, blocked by rare 4-grams) - for human reading, NOT exhaustive
# =====================================================================================================================
def _grams(s: str, n: int) -> List[str]:
    return [s[i:i + n] for i in range(len(s) - n + 1)] if len(s) >= n else ([s] if s else [])


def _tfidf_vec(s: str, idf: Dict[str, float], default_idf: float) -> Dict[str, float]:
    cnt = collections.Counter(_grams(s, 2))
    vec = {g: (1.0 + math.log(c)) * idf.get(g, default_idf) for g, c in cnt.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    return {g: v / norm for g, v in vec.items()} if norm else {}


def semantic_neighbors(queries: Sequence[str], rows: Iterable[Any], *, text_fn: Optional[Callable[[Any], str]] = None,
                       id_fn: Optional[Callable[[Any, int], str]] = None, top_n: int = 5, min_cos: float = 0.30, block_n: int = 4,
                       max_block_df: int = 30, min_shared_blocks: int = 2, idf_buffer_rows: int = 5000, snippet_chars: int = 200,
                       progress: Optional[ProgressCB] = None, progress_every: int = 20000, deadline: Optional[float] = None,
                       info: Optional[dict] = None) -> List[dict]:
    """For a SMALL list of eval questions find the train rows most similar by character-bigram TF-IDF cosine (digits and spaces
    stripped, NFKC).  Candidate rows are those sharing >= min_shared_blocks rare 4-grams (a 4-gram is a blocking key only if it
    occurs in <= max_block_df of the queries); only candidates get a cosine.  One streaming pass over `rows`.  Returns, per
    query, {"query", "matches": [{cosine, position, id, shared_blocks, snippet}]} (best first) - the numbers are for a human to read,
    not a verdict.  LIMITS: lexical only (no embeddings); paraphrases that change the wording, translations and re-ordered
    sentences are missed; IDF comes from the queries + the first `idf_buffer_rows` rows; recall is unknown, nothing is exhaustive.
    deadline = an absolute time.time() value: when it passes the pass over `rows` stops and info["truncated"] / info["rows_scanned"]
    (a dict passed by the caller) say so; the matches found so far are returned."""
    text_fn, id_fn = text_fn or default_text, id_fn or default_id
    norm = lambda t: _STRIP_DIGITS_WS.sub("", unicodedata.normalize("NFKC", t).casefold())   # noqa: E731
    qs = [norm(q) for q in queries]
    q_blocks = [set(_grams(q, block_n)) if len(q) >= block_n else set() for q in qs]
    dfq: collections.Counter = collections.Counter()
    for s in q_blocks:
        dfq.update(s)
    block_to_q: Dict[str, List[int]] = collections.defaultdict(list)
    for qi, s in enumerate(q_blocks):
        for g in sorted(s):
            if dfq[g] <= max_block_df:
                block_to_q[g].append(qi)
    it = iter(rows)
    df2: collections.Counter = collections.Counter()
    n_docs = len(qs)
    for q in qs:
        df2.update(set(_grams(q, 2)))
    buffered: List[Tuple[Any, str]] = []
    for row in itertools.islice(it, idf_buffer_rows):
        s = norm(text_fn(row))
        buffered.append((row, s))
        df2.update(set(_grams(s, 2)))
        n_docs += 1
    idf = {g: math.log((n_docs + 1) / (c + 1)) + 1.0 for g, c in df2.items()}
    default_idf = math.log(n_docs + 1) + 1.0
    qvecs = [_tfidf_vec(q, idf, default_idf) for q in qs]
    best: List[list] = [[] for _ in qs]
    tick = _Ticker(progress, "semantic_neighbors", progress_every)

    def handle(pos: int, row: Any, s: str) -> None:
        tick.tick()
        if len(s) < block_n:
            return
        shared: collections.Counter = collections.Counter()
        for g in {s[i:i + block_n] for i in range(len(s) - block_n + 1)}:
            lst = block_to_q.get(g)
            if lst:
                for qi in lst:
                    shared[qi] += 1
        vec = None
        for qi, cnt in shared.items():
            if cnt < min_shared_blocks or not qvecs[qi]:
                continue
            if vec is None:
                vec = _tfidf_vec(s, idf, default_idf)
            a, b = (vec, qvecs[qi]) if len(vec) <= len(qvecs[qi]) else (qvecs[qi], vec)
            cos = sum(v * b.get(g, 0.0) for g, v in a.items())
            if cos >= min_cos:
                item = (cos, -pos, id_fn(row, pos), text_fn(row)[:snippet_chars], cnt)
                if len(best[qi]) < top_n:
                    heapq.heappush(best[qi], item)
                else:
                    heapq.heappushpop(best[qi], item)
    scanned, stopped = 0, False
    for pos, (row, s) in enumerate(buffered):
        if deadline is not None and pos % 256 == 0 and time.time() > deadline:
            stopped = True
            break
        handle(pos, row, s)
        scanned += 1
    if not stopped:
        for pos, row in enumerate(it, start=len(buffered)):
            if deadline is not None and pos % 256 == 0 and time.time() > deadline:
                stopped = True
                break
            handle(pos, row, norm(text_fn(row)))
            scanned += 1
    if stopped and info is not None:
        info["truncated"] = True
    if info is not None:
        info["rows_scanned"] = scanned
    tick.done()
    return [{"query": q, "matches": [{"cosine": round(c, 4), "position": -negpos, "id": rid, "shared_blocks": cnt, "snippet": snip}
                                      for c, negpos, rid, snip, cnt in sorted(best[i], reverse=True)]} for i, q in enumerate(queries)]


def format_semantic_report(results: Sequence[dict]) -> str:
    out = ["semantic look-alikes (character-bigram TF-IDF cosine; lexical only, not exhaustive)"]
    for i, r in enumerate(results, 1):
        out.append(f"\n[{i}] QUERY: {r['query']}")
        if not r["matches"]:
            out.append("    (no candidate above the cosine floor)")
        for m in r["matches"]:
            out.append(f"    cos={m['cosine']:.3f} pos={m['position']} id={m['id']} shared_4grams={m['shared_blocks']}\n      {m['snippet']!r}")
    return "\n".join(out) + "\n"


# =====================================================================================================================
# Code blocks + sandbox
# =====================================================================================================================
_FENCE_OPEN = re.compile(r"^([ \t]{0,3})(`{3,}|~{3,})([^\n]*)$")
_FENCE_CLOSE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})[ \t]*$")
_LLM_CODE = re.compile(r"<llm-code>(.*?)</llm-code>", re.S)
_PY_LANGS = {"python", "py", "python3", "py3", "ipython", "python2"}


@dataclasses.dataclass
class CodeBlock:
    lang: str
    code: str
    start_line: int
    closed: bool


def extract_code_blocks(text: str, *, include_unclosed: bool = False) -> List[CodeBlock]:
    """Fenced code blocks (``` or ~~~, info string = language, up to 3 spaces of indent) of a markdown-ish text, in order."""
    blocks: List[CodeBlock] = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    i = 0
    while i < len(lines):
        m = _FENCE_OPEN.match(lines[i])
        if not m or (m.group(2)[0] == "`" and "`" in m.group(3)):
            i += 1
            continue
        indent, fence, info = len(m.group(1)), m.group(2), m.group(3).strip()
        lang = (info.split()[0] if info else "").strip("{}.").lower()
        j, body, closed = i + 1, [], False
        while j < len(lines):
            c = _FENCE_CLOSE.match(lines[j])
            if c and c.group(1)[0] == fence[0] and len(c.group(1)) >= len(fence):
                closed = True
                break
            ln = lines[j]
            body.append(ln[indent:] if ln[:indent].strip() == "" else ln)
            j += 1
        if not closed:                                    # a truncated answer: drop the blank line that the final newline leaves
            while body and not body[-1].strip():
                body.pop()
        if closed or include_unclosed:
            blocks.append(CodeBlock(lang, "\n".join(body), i + 1, closed))
        i = j + 1
    return blocks


def extract_python_blocks(text: str, *, include_unlabeled: bool = False, include_llm_code_tags: bool = True,
                          include_unclosed: bool = False) -> List[str]:
    """The Python code of a text, in order: fenced blocks labelled python / py / python3 / ipython (and unlabeled fences when
    include_unlabeled), plus <llm-code>...</llm-code> tags (AcroYAMALEX format).  A block that is never closed (a truncated
    answer) is skipped unless include_unclosed.  ```output / ```text / other languages are ignored."""
    found: List[Tuple[int, str]] = []
    for b in extract_code_blocks(text, include_unclosed=include_unclosed):
        if b.lang in _PY_LANGS or (include_unlabeled and b.lang == ""):
            found.append((b.start_line, b.code))
    if include_llm_code_tags:
        for m in _LLM_CODE.finditer(text):
            found.append((text.count("\n", 0, m.start()) + 1, m.group(1).strip("\n")))
    found.sort(key=lambda t: t[0])
    return [c for _, c in found]


# --- the sandbox.  WHAT IT IS AND IS NOT.
# The static rules are those of the GeoCPU math worker (executor_worker_v44r1.py in the r6 bundle: imports only from math cmath
# decimal fractions statistics itertools functools operator collections, no classes, no attribute assignment, no underscore
# attributes, no open/eval/exec/compile/getattr/globals ...) PLUS extra hardening found in review (import-from names, frame and
# code attributes, str.format only on a literal with plain fields, sanitised module namespaces).  A row that passes here is
# therefore also accepted by the model's own executor, and stays inside what the model can really call.
#
# The Python-level layers (static rules, restricted builtins/imports, an audit hook) are a FILTER, not a security boundary:
# CPython sandboxes in pure Python always have escapes.  The boundary is the operating system: on the Colab runtime the child
# runs as the unprivileged user 'nobody' (no supplementary groups, cannot read /content/drive, cannot write anywhere but its own
# output files), under RLIMIT_CPU / AS / FSIZE / NPROC / NOFILE, in its own session that is killed on timeout, and the notebook
# unmounts Drive while any untrusted code runs.  sandbox_selftest() proves these properties on the machine that runs the job.
# There is NO weaker fallback: if the sandbox cannot start, the caller must stop (convert.Checks raises).
EXECUTOR_ROOTS = ("math", "cmath", "decimal", "fractions", "statistics", "itertools", "functools", "operator", "collections")
THIRD_PARTY_ROOTS = ("sympy", "numpy", "pandas")                     # additional roots of the executor; off by default
SANDBOX_PROFILES = {
    "executor": {"roots": EXECUTOR_ROOTS, "allow_classes": False, "allow_attr_store": False, "allow_private_attrs": False, "dunder_names": ()},
}
SANDBOX_BACKEND = "checks.code_sandbox_check(profile=executor)"


class SandboxError(RuntimeError):
    """The sandbox cannot be started the way it must be (no weaker fallback exists)."""


_VALIDATE_SRC = r'''
import ast, re, string
class StaticReject(Exception):
    pass
DENIED_NAMES = {'eval','exec','compile','open','input','help','dir','vars','globals','locals','getattr','setattr','delattr','breakpoint','memoryview','exit','quit','type','object','super','classmethod','staticmethod','property'}
DENIED_ATTRS = {'save','savez','savez_compressed','load','loads','dump','fromfile','tofile','memmap','ctypes','ctypeslib','system','popen','fork','spawn','get_handle','urlopen','request','write','writelines','open','codegen','autowrap','ufuncify','os','sys','builtins','subprocess','socket','environ','attrgetter','methodcaller','importlib','import_module','gi_frame','gi_code','gi_yieldfrom','cr_frame','cr_code','cr_await','ag_frame','ag_code','ag_await','f_back','f_globals','f_locals','f_builtins','f_code','f_trace','tb_frame','tb_next','func_globals','func_code','func_closure','cell_contents','co_code','co_consts','co_names','mro','with_traceback'}
IMPORT_DENIED = DENIED_NAMES | DENIED_ATTRS | {'call'}
ALLOWED_DOTTED = {'collections.abc'}
_READ_PREFIXES = ('read_','to_csv','to_excel','to_json','to_pickle','to_parquet','to_hdf','to_sql','to_feather','to_orc')
_STR_FORMATTERS = ('format','format_map')
MATCH_CLASS_OK = ('int','float','str','bool','list','tuple','dict','set','frozenset','complex','bytes')
_FMT_TRAVERSAL = re.compile(r"\{[^{}]*__")
_FIELD_OK = re.compile(r"[0-9]+|[A-Za-z][A-Za-z0-9_]*")
def _check_format(s):
    try:
        parts = list(string.Formatter().parse(s))
    except ValueError:
        raise StaticReject("format_string|malformed format string")
    for _lit, field, spec, _conv in parts:
        if field and not _FIELD_OK.fullmatch(field):
            raise StaticReject("format_string|only plain {}, {0} and {name} fields are allowed (no attribute or index access)")
        if spec:
            _check_format(spec)
def _attr_bad(a, cfg):
    priv = a.startswith("__") if cfg["allow_private_attrs"] else a.startswith("_")
    return priv or a in DENIED_ATTRS or a.startswith(_READ_PREFIXES)
def _name_bad(n, cfg):
    return (n.startswith("__") and n not in cfg["dunder_names"]) or n in DENIED_NAMES
def _strname_bad(a, cfg):
    # a name that is carried as a plain STRING in the tree but is looked up as an ATTRIBUTE at run time (from-import names): the attribute
    # rules plus the two str formatters, which are only allowed on a string literal (ast.Attribute branch below)
    return _attr_bad(a, cfg) or a in _STR_FORMATTERS
_MATCH_CLASS_NODE = getattr(ast, 'MatchClass', ())
def validate(code, cfg):
    roots = set(cfg["roots"])
    if not isinstance(code, str):
        raise StaticReject("type|code must be a str")
    if len(code) > cfg.get("max_code_chars", 32768):
        raise StaticReject("too_long|%d characters" % len(code))
    if "\x00" in code:
        raise StaticReject("syntax|null byte")
    try:
        tree = ast.parse(code, filename="<row-code>", mode="exec")
    except SyntaxError as e:
        raise StaticReject("syntax|%s" % e.msg)
    except (ValueError, RecursionError, MemoryError) as e:
        raise StaticReject("syntax|%s" % type(e).__name__)
    bound, class_pattern_names = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            raise StaticReject("global_nonlocal|global/nonlocal declarations are disabled")
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise StaticReject("async|async code is disabled")
        if isinstance(node, ast.ClassDef) and not cfg["allow_classes"]:
            raise StaticReject("class_def|class definitions are disabled")
        if isinstance(node, ast.Name) and _name_bad(node.id, cfg):
            raise StaticReject("name|" + node.id)
        # Names that are carried as plain STRINGS instead of ast.Name / ast.Attribute nodes.  Two kinds:
        # (1) names that are LOOKED UP as an attribute at run time: ast.MatchClass.kwd_attrs (`case str(format=f)` is getattr(subject, 'format')
        #     - it handed out str.format on a string built at run time) and the names of `from m import n`.  Patching one forbidden name at a
        #     time failed twice (__class__, then format), so keyword attributes are DENY BY DEFAULT: a class pattern may not have any, and a
        #     class pattern may only name a builtin type.  With no keyword attribute a positional pattern on int / str / ... matches the
        #     subject itself (no attribute is read at all).  Honest math code essentially never needs a class pattern with attributes.
        # (2) names that are only BOUND (as-patterns, star / mapping captures, except-as, import-as, def names, parameters, type
        #     parameters, call keywords such as dict(format=1)): no attribute is ever read from them, so they only get the name rules below.
        if getattr(node, "kwd_attrs", None):                                 # ast.MatchClass (Python 3.10+)
            raise StaticReject("match_class|keyword attributes in class patterns are disabled: " + ", ".join(str(a) for a in node.kwd_attrs))
        if isinstance(node, _MATCH_CLASS_NODE):
            if not (isinstance(node.cls, ast.Name) and node.cls.id in MATCH_CLASS_OK):
                raise StaticReject("match_class|class patterns are only allowed on these builtin types: " + " ".join(MATCH_CLASS_OK))
            class_pattern_names.add(node.cls.id)
        for _fld in ("name", "rest", "asname"):
            _bound = getattr(node, _fld, None)
            if isinstance(_bound, str) and not (_fld == "name" and isinstance(node, ast.alias)) and _name_bad(_bound, cfg):
                raise StaticReject("name|" + _bound)
            if isinstance(_bound, str) and not isinstance(node, ast.alias):
                bound.add(_bound)
        if isinstance(node, ast.arg):
            if _name_bad(node.arg, cfg):
                raise StaticReject("name|" + node.arg)
            bound.add(node.arg)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        if isinstance(node, ast.alias):
            bound.add(node.asname or node.name.split(".")[0])
        if isinstance(node, ast.Attribute):
            a = node.attr
            if _attr_bad(a, cfg):
                raise StaticReject("attr|" + a)
            if isinstance(node.ctx, (ast.Store, ast.Del)) and not cfg["allow_attr_store"]:
                raise StaticReject("attr_store|attribute assignment is disabled: ." + a)
            if a in ("format", "format_map"):
                v = node.value
                if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                    raise StaticReject("format_string|str.format / format_map are only allowed on a string literal")
                _check_format(v.value)
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name.split(".")[0] not in roots or ("." in n.name and n.name not in ALLOWED_DOTTED):
                    raise StaticReject("import|" + n.name)
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level or not mod or mod.split(".")[0] not in roots or ("." in mod and mod not in ALLOWED_DOTTED):
                raise StaticReject("import|" + str(node.module))
            for a in node.names:
                if a.name == "*" or a.name.startswith("_"):
                    raise StaticReject("wildcard_import|wildcard or private import")
                if a.name in IMPORT_DENIED or _strname_bad(a.name, cfg):
                    raise StaticReject("import_name|" + a.name)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and _FMT_TRAVERSAL.search(node.value):
            raise StaticReject("format_string|attribute traversal inside a format string")
    clash = sorted(class_pattern_names & bound)                  # `str = namedtuple(...)` would turn `case str(x)` into a pattern on another class
    if clash:
        raise StaticReject("match_class|a name that a class pattern uses is rebound in this code: " + ", ".join(clash))
    return tree
'''

_CHILD_SRC = _VALIDATE_SRC + r'''
import builtins, contextlib, importlib, io, json, os, sys, types
cfg = json.loads(sys.argv[1])
with open(sys.argv[2], encoding="utf-8") as _f:
    code = _f.read()
try:
    import resource
except Exception:
    resource = None
if resource is not None:
    def _lim(name, val):
        try:
            which = getattr(resource, name)
            hard = resource.getrlimit(which)[1]
            if hard != resource.RLIM_INFINITY:
                val = min(val, hard)
            resource.setrlimit(which, (val, val))
        except Exception:
            pass
    _lim("RLIMIT_CPU", int(cfg["cpu"]))
    if cfg.get("mem_mb"):
        _lim("RLIMIT_AS", int(cfg["mem_mb"]) << 20)
    _lim("RLIMIT_FSIZE", 1 << 20)
    _lim("RLIMIT_CORE", 0)
    _lim("RLIMIT_NPROC", 64)
    _lim("RLIMIT_NOFILE", 64)
ROOTS = set(cfg["roots"])
_HIDE = {"attrgetter", "methodcaller", "call"}
def _public(mod):
    # a namespace of the PUBLIC, NON-MODULE attributes of a module: nothing like random._os, fractions.sys or dataclasses.inspect
    return types.SimpleNamespace(**{k: v for k, v in vars(mod).items() if not k.startswith("_") and not isinstance(v, types.ModuleType) and k not in _HIDE})
NS = {}
for r in sorted(ROOTS):
    try:
        NS[r] = _public(importlib.import_module(r))
    except Exception:
        pass
DOTTED = {}
if "collections" in NS:
    try:
        DOTTED["collections.abc"] = _public(importlib.import_module("collections.abc"))
        NS["collections"].abc = DOTTED["collections.abc"]
    except Exception:
        pass
LIB = [os.path.realpath(os.path.dirname(os.__file__)), os.path.realpath(os.path.join(sys.base_prefix, "DLLs"))]
if cfg.get("third_party"):
    for _n in NS:
        _fn = getattr(sys.modules.get(_n), "__file__", None)
        if _fn:
            LIB.append(os.path.realpath(os.path.dirname(_fn)))
    LIB += [os.path.realpath(p) for p in sys.path if p.endswith(("site-packages", "dist-packages"))]
def _in_lib(p):
    try:
        p = os.path.realpath(p)
    except Exception:
        return False
    return any(p == r or p.startswith(r + os.sep) for r in LIB)
_BLOCK_PREFIX = ("socket.", "subprocess.", "ctypes.", "urllib.", "http.client.", "ftplib.", "smtplib.", "imaplib.", "poplib.", "telnetlib.", "webbrowser.", "shutil.", "pty.", "_posixsubprocess.")
_BLOCK_EVENTS = {"os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty", "os.kill", "os.killpg", "os.remove", "os.rmdir", "os.rename", "os.mkdir", "os.chdir", "os.chmod", "os.chown", "os.link", "os.symlink", "os.truncate", "os.putenv", "os.unsetenv", "os.utime", "os.setxattr", "os.removexattr", "os.listxattr", "os.chroot", "os.mkfifo", "os.mknod"}
def audit(event, args):
    if event.startswith(_BLOCK_PREFIX) or event in _BLOCK_EVENTS:
        raise PermissionError("operation disabled in the sandbox: " + event)
    if event in ("os.listdir", "os.scandir"):
        if not args or not isinstance(args[0], (str, bytes, os.PathLike)) or not _in_lib(os.fsdecode(args[0])):
            raise PermissionError("filesystem enumeration disabled")
    if event == "open":
        f, mode, flags = args
        if not isinstance(f, (str, bytes, os.PathLike)):
            raise PermissionError("file descriptor access disabled")
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError("filesystem writes disabled")
        if not _in_lib(os.fsdecode(f)):
            raise PermissionError("filesystem reads outside the python library disabled")
sys.addaudithook(audit)
_NAMES = ('abs','all','any','bin','bool','bytes','bytearray','chr','complex','dict','divmod','enumerate','filter','float','format','frozenset','hash','hex','int','isinstance','issubclass','iter','len','list','map','max','min','next','oct','ord','pow','print','range','repr','reversed','round','set','slice','sorted','str','sum','tuple','zip','Exception','ValueError','TypeError','ZeroDivisionError','RuntimeError','AssertionError','NameError','IndexError','KeyError','StopIteration','OverflowError','ArithmeticError','LookupError','True','False','None')
SAFE = {n: getattr(builtins, n) for n in _NAMES if hasattr(builtins, n)}
def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise PermissionError("import disabled: relative import")
    if name in DOTTED:
        return DOTTED[name] if fromlist else NS[name.split(".")[0]]
    if "." in name or name.split(".")[0] not in ROOTS:
        raise PermissionError("import disabled: " + name)
    if name not in NS:
        raise ImportError("module unavailable: " + name)
    return NS[name]
SAFE["__import__"] = safe_import
class OutputLimit(RuntimeError):
    pass
class Capped(io.TextIOBase):
    def __init__(self, cap):
        self.cap, self.parts, self.size = cap, [], 0
    def write(self, text):
        text = str(text)
        room = max(0, self.cap - self.size)
        if room:
            self.parts.append(text[:room])
            self.size += min(len(text), room)
        if len(text) > room:
            raise OutputLimit("captured output exceeded %d characters" % self.cap)
        return len(text)
    def flush(self):
        pass
    def getvalue(self):
        return "".join(self.parts)
out, err = Capped(int(cfg["cap"])), Capped(int(cfg["cap"]))
res = {"ok": False, "stdout": "", "error": ""}
ns = {"__builtins__": SAFE, "__name__": "__main__" if cfg["dunder_names"] else "tir_math"}
try:
    tree = validate(code, cfg)
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        exec(compile(tree, "<row-code>", "exec"), ns, ns)
    res["ok"] = True
except StaticReject as e:
    res["error"] = "static:" + str(e).replace("|", ":", 1)
except OutputLimit as e:
    res["error"] = "output_limit:" + str(e)
except BaseException as e:
    res["error"] = "exception:" + type(e).__name__ + ": " + str(e)[:300]
res["stdout"] = out.getvalue()
sys.__stdout__.write(cfg["marker"] + json.dumps(res) + "\n")
sys.__stdout__.flush()
'''

_VALIDATE_NS: Dict[str, Any] = {}
exec(compile(_VALIDATE_SRC, "<checks-validate>", "exec"), _VALIDATE_NS)       # same text the child runs


def _profile_cfg(profile: str, allowed_modules: Optional[Iterable[str]], allow_third_party: bool) -> dict:
    if profile not in SANDBOX_PROFILES:
        raise ValueError(f"unknown sandbox profile {profile!r} (use {sorted(SANDBOX_PROFILES)})")
    cfg = dict(SANDBOX_PROFILES[profile])
    roots = list(allowed_modules) if allowed_modules is not None else list(cfg["roots"])
    if allow_third_party:
        roots += [r for r in THIRD_PARTY_ROOTS if r not in roots]
    cfg["roots"] = sorted(set(roots))
    cfg["dunder_names"] = list(cfg["dunder_names"])
    cfg["third_party"] = bool(allow_third_party)
    return cfg


def static_check(code: str, *, profile: str = "executor", allowed_modules: Optional[Iterable[str]] = None,
                 allow_third_party: bool = False) -> str:
    """'' when the code passes the static rules, else '<category>:<detail>' (syntax, too_long, global_nonlocal, async, class_def, name,
    attr, attr_store, import, import_name, wildcard_import, format_string, match_class).  Profile 'executor' = the GeoCPU math worker's rules
    (imports only from math cmath decimal fractions statistics itertools functools operator collections; no classes, no attribute
    assignment, no underscore attributes, no open/eval/exec/compile/getattr/globals..., no 'os'/'sys'/'subprocess'/'socket'
    attributes, no frame / code introspection attributes, str.format only on a literal with plain fields, no keyword attribute in a
    match class pattern and class patterns only on the builtin types int float str bool list tuple dict set frozenset complex bytes).
    This is the ONLY profile:
    the earlier 'wide' profile (classes, random, re, json ...) was removed because the model's own executor cannot run such code."""
    cfg = _profile_cfg(profile, allowed_modules, allow_third_party)
    try:
        _VALIDATE_NS["validate"](code, cfg)
    except _VALIDATE_NS["StaticReject"] as e:
        return str(e).replace("|", ":", 1)
    return ""


def _resolve_run_as(run_as: Union[None, bool, str]) -> Optional[Tuple[int, int, str]]:
    """None (no user switch) or (uid, gid, name) of the unprivileged account the child runs as.  'auto' = 'nobody' when this process is
    root on POSIX (Colab), nothing otherwise (a local test run).  An explicit name that cannot be used raises SandboxError."""
    if run_as in (None, False):
        return None
    if os.name != "posix":
        if run_as == "auto":
            return None
        raise SandboxError(f"run_as={run_as!r}: user switching needs POSIX")
    import pwd
    if run_as == "auto":
        if os.geteuid() != 0:
            return None
        run_as = "nobody"
    elif os.geteuid() != 0:
        raise SandboxError(f"run_as={run_as!r} needs root to switch users")
    try:
        pw = pwd.getpwnam(str(run_as))
    except KeyError as e:
        raise SandboxError(f"user {run_as!r} does not exist on this machine") from e
    if pw.pw_uid == 0:
        raise SandboxError("refusing to run the sandbox child as root")
    return pw.pw_uid, pw.pw_gid, str(run_as)


_POSIX = os.name == "posix"
_LIVE_CHILDREN: Dict[int, bool] = {}                 # pid (= process-group id: every child is a session leader) of the sandbox children running right now
_LIVE_LOCK = threading.RLock()                     # re-entrant on purpose: convert.py's SIGTERM handler runs in the main thread, possibly while _track_child holds the lock


def _track_child(pid: int, running: bool) -> None:
    with _LIVE_LOCK:
        if running:
            _LIVE_CHILDREN[pid] = True
        else:
            _LIVE_CHILDREN.pop(pid, None)


def kill_live_sandbox_children() -> int:
    """SIGKILL the process group of every sandbox child that is running right now (POSIX only; returns how many groups were signalled).
    Each child is a session leader (start_new_session), so a signal sent to the CONVERTER's process group never reaches it: convert.py calls
    this from its SIGTERM / SIGINT handler and at exit, so that stopping a converter also stops the untrusted code it started."""
    if not _POSIX:
        return 0
    with _LIVE_LOCK:
        pids = list(_LIVE_CHILDREN)
    n = 0
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGKILL)
            n += 1
        except OSError:
            pass                                       # already gone
    return n


def _file_tail(path: str, nbytes: int) -> bytes:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            return f.read(nbytes)
    except OSError:
        return b""


def _spawn(script_text: str, args: Sequence[str], *, timeout: float, flags: Sequence[str], python: Optional[str],
           acct: Optional[Tuple[int, int, str]], files: Optional[Mapping[str, str]] = None) -> dict:
    """Run `python <flags> _run.py <args>` once in a fresh temp dir: cwd = that dir, stdin closed, minimal environment, stdout/stderr
    to files (never pipes, so a child cannot fill this process's memory; only the tail is read), own session killed on timeout, and
    on POSIX as the unprivileged `acct` with no supplementary groups.  Returns {returncode, out, err, timed_out, spawn_error}."""
    exe = python or sys.executable
    env = {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    if os.name == "nt":
        for k in ("SYSTEMROOT", "SystemRoot", "windir", "TEMP", "TMP"):
            if k in os.environ:
                env[k] = os.environ[k]
        env["PATH"] = os.path.dirname(exe)
    with tempfile.TemporaryDirectory(prefix="v49sbx_", ignore_cleanup_errors=True) as d:
        script, out_p, err_p = os.path.join(d, "_run.py"), os.path.join(d, "out.bin"), os.path.join(d, "err.bin")
        payload = {"_run.py": script_text}
        payload.update(files or {})
        for name, text in payload.items():
            with open(os.path.join(d, name), "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
        kw: Dict[str, Any] = {}
        if acct is not None:                                      # the unprivileged child must be able to read its script
            os.chmod(d, 0o755)
            for name in payload:
                os.chmod(os.path.join(d, name), 0o644)
            kw.update(user=acct[0], group=acct[1], extra_groups=[])
        timed_out, rc, spawn_error = False, None, ""
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            try:
                proc = subprocess.Popen([exe] + list(flags) + [script] + list(args), stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, cwd=d,
                                        env=env, start_new_session=(os.name == "posix"), **kw)
            except (OSError, ValueError) as e:
                proc, spawn_error = None, f"{type(e).__name__}: {e}"
            if proc is not None:
                if _POSIX:
                    _track_child(proc.pid, True)
                try:
                    try:
                        proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        try:
                            if os.name == "posix":
                                os.killpg(proc.pid, signal.SIGKILL)
                            else:
                                proc.kill()
                        except OSError:
                            pass
                        proc.wait()
                finally:
                    if _POSIX:
                        _track_child(proc.pid, False)
                rc = proc.returncode
        raw_out, raw_err = _file_tail(out_p, 1 << 18), _file_tail(err_p, 4096)
    return {"returncode": rc, "out": raw_out, "err": raw_err, "timed_out": timed_out, "spawn_error": spawn_error}


def code_sandbox_run(code: str, timeout: float = 5.0, *, profile: str = "executor", allowed_modules: Optional[Iterable[str]] = None,
                     allow_third_party: bool = False, max_output_chars: int = 8192, mem_mb: Optional[int] = None,
                     python: Optional[str] = None, run_as: Union[None, bool, str] = "auto") -> dict:
    """Run `code` in a fresh python subprocess.  Returns {ok, stdout, error, category, seconds}.
    category: '' | static | timeout | output_limit | exception | crash.  error is '<category>:<detail>'.
    Defence in depth (see the comment above SANDBOX_PROFILES: the Python layers are a filter, the OS is the boundary): only code that
    passes the static rules is ever executed and the child checks it again; python -S -s -B -X utf8 (no site / user site / .pyc;
    -I is not used because it would ignore PYTHONHASHSEED=0, which makes set order of strings reproducible); cwd = a fresh temp
    dir; stdin closed; minimal environment; restricted builtins; modules are handed out as namespaces of their public non-module
    attributes; an audit hook refuses sockets / subprocesses / ctypes / file writes / reads outside the python library; output is
    capped at max_output_chars; wall-clock `timeout` (the whole process group is killed).  On POSIX: RLIMIT_CPU / AS (mem_mb, default
    1024; 4096 with third party) / FSIZE / CORE / NPROC / NOFILE and, when this process is root (Colab), the child runs as the
    unprivileged user 'nobody' (run_as='auto'; None disables; a name forces it).  Windows has no `resource`: only the wall-clock
    timeout applies there (local unit tests only)."""
    t0 = time.time()
    cfg = _profile_cfg(profile, allowed_modules, allow_third_party)
    rej = static_check(code, profile=profile, allowed_modules=allowed_modules, allow_third_party=allow_third_party)
    if rej:
        return {"ok": False, "stdout": "", "error": "static:" + rej, "category": "static", "seconds": time.time() - t0}
    acct = _resolve_run_as(run_as)
    marker = "@@CHECKS-RESULT-" + secrets.token_hex(8) + "@@"
    cfg.update({"cpu": int(math.ceil(timeout)) + 1, "cap": int(max_output_chars), "marker": marker,
                "mem_mb": mem_mb if mem_mb is not None else (4096 if allow_third_party else 1024)})
    flags = ["-s", "-B", "-X", "utf8"] + ([] if allow_third_party else ["-S"])
    r = _spawn(_CHILD_SRC, [json.dumps(cfg), "code.py"], timeout=timeout, flags=flags, python=python, acct=acct, files={"code.py": code})
    secs = time.time() - t0
    if r["spawn_error"]:
        return {"ok": False, "stdout": "", "error": "crash:spawn:" + r["spawn_error"][:200], "category": "crash", "seconds": secs}
    if r["timed_out"]:
        return {"ok": False, "stdout": "", "error": f"timeout:{timeout}s", "category": "timeout", "seconds": secs}
    result = None
    for line in reversed(r["out"].decode("utf-8", "replace").splitlines()):
        if line.startswith(marker):
            try:
                result = json.loads(line[len(marker):])
            except ValueError:
                result = None
            break
    if result is None:
        tail = r["err"].decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
        return {"ok": False, "stdout": "", "error": f"crash:returncode={r['returncode']}:{tail[0][:200]}", "category": "crash", "seconds": secs}
    cat = "" if result["ok"] else result["error"].split(":", 1)[0]
    return {"ok": bool(result["ok"]), "stdout": result["stdout"], "error": result["error"], "category": cat, "seconds": secs}


def code_sandbox_check(code: str, timeout: float = 5.0, **kw: Any) -> Tuple[bool, str, str]:
    """(ok, stdout, error) - see code_sandbox_run for the flags, profile and limits.  error is '' when ok, else e.g.
    'static:import:os', 'static:name:open', 'static:class_def:...', 'timeout:5s', 'exception:ZeroDivisionError: ...',
    'output_limit:...', 'crash:returncode=...'.  `print(1 + 1)` -> (True, '2\\n', '')."""
    r = code_sandbox_run(code, timeout, **kw)
    return r["ok"], r["stdout"], r["error"]


# payloads that the static rules must refuse (each is harmless if it ever ran: it only prints or raises)
SANDBOX_MUST_REJECT = (
    ("import os", "import os\nprint(os.getcwd())\n"),
    ("operator.attrgetter walk", "from operator import attrgetter, methodcaller\nprint(methodcaller('__subclasses__')(attrgetter('__class__.__base__')(0)))\n"),
    ("format-string traversal (literal)", "print('{0.__class__}'.format(1))\n"),
    ("format-string traversal (built)", "s = '{0._' + '_class__}'\nprint(s.format(1))\n"),
    ("generator frame walk", "g = (g.gi_frame.f_back for _ in [1])\nprint(next(g))\n"),
    ("class definition", "class A:\n    pass\n"),
    ("import re", "import re\nprint(re.sub('a', 'b', 'a'))\n"),
    ("import random", "import random\nprint(random.random())\n"),
    ("open()", "print(open('x'))\n"),
    ("getattr()", "print(getattr(1, 'real'))\n"),
    ("match class pattern reads __class__", "match 0:\n    case int(__class__=c):\n        print(c)\n"),
    ("match capture named like a dunder", "match 0:\n    case int() as __builtins__:\n        print(1)\n"),
    # keyword attributes in class patterns are refused whatever their name (the format / format_map hole: the str.format rule only existed on ast.Attribute)
    ("match keyword attribute: str.format on a built string", "s = '{0._' + '_class__}'\nmatch s:\n    case str(format=f):\n        print(f(1))\n"),
    ("match keyword attribute: str.format_map", "s = '{a._' + '_class__}'\nmatch s:\n    case str(format_map=f):\n        print(f({'a': 1}))\n"),
    ("match keyword attribute: format walks to __globals__", "def g():\n    pass\ns = '{0._' + '_globals__}'\nmatch s:\n    case str(format=f):\n        print(f(g)[:60])\n"),
    ("match keyword attribute: format_map walks to __globals__", "def g():\n    pass\ns = '{a._' + '_globals__}'\nmatch s:\n    case str(format_map=f):\n        print(f({'a': g})[:60])\n"),
    ("match keyword attribute with a benign-looking name", "match 3:\n    case int(real=r):\n        print(r)\n"),
    ("match class pattern on a name that is not a builtin type", "from collections import Counter\nmatch Counter('a'):\n    case Counter():\n        print(1)\n"),
    ("match class pattern on a rebound builtin name", "from collections import namedtuple\nstr = namedtuple('P', 'x')\nmatch str(1):\n    case str(a):\n        print(a)\n"),
)
_PROBE_SRC = r'''
import json, os
res = {"uid": os.getuid() if hasattr(os, "getuid") else None, "gid": os.getgid() if hasattr(os, "getgid") else None}
def t(label, fn):
    try:
        res[label] = "ok:" + str(fn())
    except BaseException as e:
        res[label] = "error:" + type(e).__name__
t("listdir_drive", lambda: len(os.listdir("/content/drive")))
t("listdir_mydrive", lambda: len(os.listdir("/content/drive/MyDrive")))
t("read_etc_shadow", lambda: len(open("/etc/shadow").read()))
res["can_write_mydrive"] = os.access("/content/drive/MyDrive", os.W_OK)
res["can_write_content"] = os.access("/content", os.W_OK)
res["can_write_cwd"] = os.access(".", os.W_OK)
print(json.dumps(res))
'''


def sandbox_selftest(*, timeout: float = 20.0, python: Optional[str] = None, require_os_containment: bool = False, light: bool = False,
                     run_as: Union[None, bool, str] = "auto", drive_unmounted: bool = False) -> dict:
    """Proves on THIS machine that the sandbox works and refuses what it must.  Returns {ok, backend, run_as, checks: [{name, ok,
    detail[, info]}], failed: [names], n_passed, n_info}.  light=True runs only the cheap start-up checks (convert.Checks uses it); the full run (the notebook,
    step '3d') adds the escape payloads, the runtime-import smoke test, timeout handling and the OS probe.  With
    require_os_containment=True a machine where the child cannot be switched to an unprivileged user (Windows, a non-root Linux
    process) FAILS: that is the setting for the Colab run.  On a root POSIX process the OS probe additionally requires that the
    child is not root, cannot list or write a mounted /content/drive and cannot read /etc/shadow.  drive_unmounted=True says that the
    notebook unmounts Drive while code runs (and verifies it): the two Drive checks are then replaced by ONE information line (info=True,
    never a failure, not counted in n_passed) that only reports what the child could see of the mounted Drive, because the unmount is the
    barrier that matters for Drive (a mounted Drive that happens to be world-readable must not block the run).  This test runs while
    Drive is still mounted: it proves nothing about the unmount, which happens only during 7a / 7b."""
    out: List[dict] = []

    def add(name: str, ok: bool, detail: str = "", info: bool = False) -> bool:
        # info=True: a line that only REPORTS something (it can never fail and is not counted as a passed check)
        out.append({"name": name, "ok": bool(ok), "detail": detail, **({"info": True} if info else {})})
        return bool(ok)

    def finish(acct: Optional[Tuple[int, int, str]]) -> dict:
        failed = [c["name"] for c in out if not c["ok"]]
        return {"ok": not failed, "backend": SANDBOX_BACKEND, "run_as": None if acct is None else acct[2], "light": light, "checks": out,
                "failed": failed, "n_passed": sum(1 for c in out if c["ok"] and not c.get("info")), "n_info": sum(1 for c in out if c.get("info")),
                "platform": sys.platform, "python": sys.version.split()[0]}

    try:
        acct = _resolve_run_as(run_as)
    except SandboxError as e:
        add("run_as", False, str(e))
        return finish(None)                    # the other checks would call code_sandbox_run with the same unusable run_as and raise out of here
    else:
        add("run_as", True, "no user switch (not root / not POSIX)" if acct is None else f"child runs as {acct[2]} (uid {acct[0]})")
    if require_os_containment:
        add("os_containment_available", acct is not None, "the child must run as an unprivileged user" if acct is None else f"uid {acct[0]}")
    r = code_sandbox_run("print(1 + 1)\n", timeout, python=python, run_as=run_as)
    add("print(1 + 1)", r["ok"] and r["stdout"].strip() == "2", r["error"] or r["stdout"].strip())
    r = code_sandbox_run("import os\n", timeout, python=python, run_as=run_as)
    add("import os is refused", (not r["ok"]) and r["category"] == "static", r["error"])
    r = code_sandbox_run("import math, cmath, decimal, fractions, statistics, itertools, functools, operator, collections\nprint('ok')\n", timeout,
                         python=python, run_as=run_as)
    add("every allowed module imports", r["ok"] and r["stdout"].strip() == "ok", r["error"] or r["stdout"].strip())
    if not light:
        for name, code, expect in (
                ("Fraction", "from fractions import Fraction\nprint(Fraction(1, 3) + Fraction(1, 6))\n", "1/2"),
                ("Decimal", "from decimal import Decimal\nprint(Decimal('0.1') + Decimal('0.2'))\n", "0.3"),
                ("statistics", "import statistics\nprint(statistics.mean([1, 2, 3]))\n", "2"),
                ("collections", "import collections\nprint(collections.Counter('aab').most_common(1))\n", "[('a', 2)]"),
                ("collections.abc", "from collections.abc import Iterable\nprint(isinstance([1], Iterable))\n", "True"),
                ("operator", "import operator\nprint(operator.add(1, 2), operator.mul(3, 4))\n", "3 12"),
                ("functools", "from functools import reduce\nprint(reduce(lambda a, b: a * b, [1, 2, 3, 4]))\n", "24"),
                ("itertools", "import itertools\nprint(list(itertools.permutations([1, 2, 3]))[0])\n", "(1, 2, 3)"),
                ("math", "import math\nprint(round(math.sqrt(2), 4), math.gcd(12, 18))\n", "1.4142 6"),
                ("str.format literal", "print('{:>5}|{name}'.format(3, name='x'))\n", "3|x"),
                ("function and loop", "def f(n):\n    return sum(i * i for i in range(n))\nprint(f(10))\n", "285"),
                ("match without a class pattern", "match (3, 4):\n    case (a, b) if a < b:\n        print(a + b)\n    case _:\n        print(0)\n", "7"),
                ("match with a positional builtin class pattern", "match 5:\n    case int(n):\n        print(n + 1)\n", "6")):
            r = code_sandbox_run(code, timeout, python=python, run_as=run_as)
            add("runs: " + name, r["ok"] and r["stdout"].strip() == expect, r["error"] or r["stdout"].strip())
        for name, code in SANDBOX_MUST_REJECT:
            r = code_sandbox_run(code, timeout, python=python, run_as=run_as)
            add("refused: " + name, (not r["ok"]) and r["category"] == "static", r["error"] or ("RAN: " + r["stdout"].strip()[:80]))
        r = code_sandbox_run("import statistics\nprint(statistics.random)\n", timeout, python=python, run_as=run_as)
        add("module namespaces hide submodules (statistics.random)", (not r["ok"]) and ("AttributeError" in r["error"] or r["category"] == "static"),
            r["error"] or ("RAN: " + r["stdout"].strip()[:80]))
        r = code_sandbox_run("while True:\n    pass\n", 1.0, python=python, run_as=run_as)
        add("endless loop is stopped", (not r["ok"]) and r["category"] in ("timeout", "exception", "crash"), r["error"])
        r = code_sandbox_run("print('x' * 100000)\n", timeout, python=python, run_as=run_as, max_output_chars=1000)
        add("output is capped", (not r["ok"]) and r["category"] == "output_limit", r["error"])
        probe = _spawn(_PROBE_SRC, [], timeout=timeout, flags=["-s", "-B", "-X", "utf8", "-S"], python=python, acct=acct)
        info: dict = {}
        try:
            info = json.loads(probe["out"].decode("utf-8", "replace").strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        if not info:
            add("os probe ran", False, probe["spawn_error"] or probe["err"].decode("utf-8", "replace")[-200:])
        else:
            add("os probe ran", True, json.dumps(info))
            if acct is not None:
                add("child is not root", info.get("uid") not in (0, None), f"uid {info.get('uid')}")
                sees = str(info.get("listdir_drive", "")).startswith("ok:") or str(info.get("listdir_mydrive", "")).startswith("ok:")
                writes = bool(info.get("can_write_mydrive"))
                if drive_unmounted:
                    # NOT a proof that Drive is unmounted: 3d runs with Drive mounted; the unmount happens only during 7a / 7b and is enforced
                    # by the notebook's _detach (which verifies it).  This line only reports what a child could see of a mounted Drive.
                    add("info: what the child could see of Drive at 3d time (Drive is unmounted only during 7a/7b, enforced by the notebook _detach)", True,
                        ("it could list it" if sees else "it saw nothing") + ("; it could write it" if writes else "")
                        + f" [{info.get('listdir_drive')} / {info.get('listdir_mydrive')}]", info=True)
                else:
                    add("child cannot list /content/drive", not sees, f"{info.get('listdir_drive')} / {info.get('listdir_mydrive')}")
                    add("child cannot write Drive", not writes, str(info.get("can_write_mydrive")))
                add("child cannot read /etc/shadow", not str(info.get("read_etc_shadow", "")).startswith("ok:"), str(info.get("read_etc_shadow")))
    return finish(acct)


def code_sandbox_check_many(codes: Sequence[str], timeout: float = 5.0, *, workers: int = 2, progress: Optional[ProgressCB] = None,
                            progress_every: int = 200, **kw: Any) -> List[Tuple[bool, str, str]]:
    """code_sandbox_check over many snippets with `workers` parallel subprocesses (Colab has 2 vCPUs); order is preserved."""
    tick = _Ticker(progress, "sandbox", progress_every)

    def one(c: str) -> Tuple[bool, str, str]:
        r = code_sandbox_check(c, timeout, **kw)
        tick.tick()
        return r
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        out = list(ex.map(one, codes))
    tick.done()
    return out


_NUM_COMMA = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


def normalize_answer(s: Any) -> str:
    s = unicodedata.normalize("NFKC", str(s)).strip().replace(" ", "").replace("\u3000", "").lower()
    s = s.rstrip("。.")
    return s.replace(",", "") if _NUM_COMMA.match(s) else s


def _as_number(s: str) -> Optional[float]:
    try:
        if "/" in s and s.count("/") == 1:
            a, b = s.split("/")
            return float(a) / float(b)
        return float(s)
    except (ValueError, ZeroDivisionError):
        return None


def answers_equal(output: Any, reference: Any, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    """Output of the sandbox == reference answer: equal after NFKC / whitespace / trailing '。' / thousands-comma normalisation, or
    equal as numbers (int, decimal, a/b fraction) within rel_tol."""
    a, b = normalize_answer(output), normalize_answer(reference)
    if a == b:
        return True
    x, y = _as_number(a), _as_number(b)
    return x is not None and y is not None and math.isclose(x, y, rel_tol=rel_tol, abs_tol=abs_tol)


# =====================================================================================================================
# The by-eye sample
# =====================================================================================================================
def sample_for_reading(rows: Iterable[Any], n: int = 100, seed: int = 0, out_path: Optional[str] = None, *,
                       text_fn: Optional[Callable[[Any], str]] = None, title: str = "", vocab: Any = None,
                       max_chars: int = 4000, progress: Optional[ProgressCB] = None) -> List[Tuple[int, Any]]:
    """Seeded sample of n rows (priority sampling: the n rows with the smallest sha256(seed:position); deterministic across
    Python versions and independent of the stream length), returned as [(position, row)] in stream order and, with out_path,
    written as a UTF-8 text file for a human: position, id, length, OOV share and characters (when `vocab` is given), prompt and
    target (or the JSON of the row).  Refuses to overwrite a different existing file (write_new_file)."""
    text_fn = text_fn or default_text
    heap: List[Tuple[int, int, Any]] = []                  # max-heap on priority via negation
    total = 0
    tick = _Ticker(progress, "sample_for_reading", 100000)
    for pos, row in enumerate(rows):
        total += 1
        pri = int.from_bytes(hashlib.sha256(f"{seed}:{pos}".encode()).digest()[:8], "big")
        if len(heap) < n:
            heapq.heappush(heap, (-pri, pos, row))
        elif pri < -heap[0][0]:
            heapq.heapreplace(heap, (-pri, pos, row))
        tick.tick()
    tick.done()
    picked = sorted(((pos, row) for _, pos, row in heap), key=lambda t: t[0])
    if out_path:
        chars = _chars_of(vocab) if vocab is not None else None
        out = [f"# {title or 'sample'}  seed={seed}  n={len(picked)} of {total} rows (priority sampling; read every row by eye)\n"]
        for k, (pos, row) in enumerate(picked, 1):
            text = text_fn(row)
            norm = text.translate(RUNTIME_NORMALIZE)
            head = f"==== [{k}/{len(picked)}] position={pos} id={default_id(row, pos)}  len(prompt+target)+1={row_length(row, None)}"
            if chars is not None:
                frac, nb = oov_fraction(norm, chars)
                bad = sorted(set(norm) - chars)
                head += f"  oov={frac:.4f}" + (f" oov_chars={[f'U+{ord(c):04X}' for c in bad][:12]}" if bad else "")
            out.append(head + " ====\n")
            if isinstance(row, dict) and isinstance(row.get("prompt"), str) and isinstance(row.get("target"), str):
                for key in ("source", "family", "split", "answer_type", "reference_final_answer", "license"):
                    if row.get(key) not in (None, ""):
                        out.append(f"{key}: {row[key]}\n")
                out.append("--- prompt ---\n" + row["prompt"][:max_chars] + "\n--- target ---\n" + row["target"][:max_chars] + "\n")
                steps = row.get("tool_steps")
                if steps:
                    out.append(f"--- tool_steps ({len(steps)}), loss_mask_spans={row.get('loss_mask_spans')} ---\n")
            else:
                out.append(json.dumps(row, ensure_ascii=False, indent=1)[:max_chars] + "\n")
            out.append("\n")
        write_new_file(out_path, "".join(out))
    return picked


__all__ = [n for n in dir() if not n.startswith("_") and n not in {
    "annotations", "array", "ast", "bisect", "collections", "dataclasses", "gzip", "hashlib", "heapq", "io", "itertools", "json", "math",
    "operator", "os", "re", "secrets", "signal", "subprocess", "sys", "tempfile", "time", "unicodedata", "zipfile", "zlib",
    "ThreadPoolExecutor", "Any", "Callable", "Dict", "Iterable", "Iterator", "List", "Mapping", "Optional", "Sequence", "Tuple", "Union"}]
