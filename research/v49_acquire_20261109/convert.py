#!/usr/bin/env python3
"""GeoCPU v49 acquisition converters: raw Hugging Face files -> GeoCPU training rows (one converter per dataset).

Runs on a Google Colab CPU runtime (12 GB RAM) over files that were downloaded from huggingface.co.  Everything is
streaming: a 1 GB JSONL, a 330 MB parquet and a 37 MB JSON array are read record by record.  Nothing here needs a secret.

    python convert.py --dataset jaquad|coding|tengentoppa|hotpot|toolace|wizard|all --raw-dir DIR --out-dir DIR [--limit N]
                      [--vocab base_metadata.json] [--max-chars 1900] [--sample-dump 100] [--no-sandbox]

Row contract = the existing families (see families/jglue_human/build_jglue_rows.py, families/python_a/common.py):
    id, source, license, split, training_category, family, structural_template, difficulty, _v46_group,
    prompt, target, full_text (== prompt + target exactly), loss_mask_spans, tool_steps, reference_final_answer, answer_type
  + attribution, adoption, and family specific extras.  Prompt style "問題: ...\\n条件と根拠を順に考えてください。",
  target "\\n根拠: 「verbatim quote」\\n答え: X。".  A row needs len(full_text)+1 <= 2048; the cap used here is --max-chars (1900).

Rules every converter obeys (all re-checked by validate_row() on every emitted row):
  * every input record ends as exactly ONE of: an emitted row, or a counted reject with a reason  (n_in == n_out + sum(rejects))
  * rows with injected tool observations (python stdout, tool results) list the observation text in loss_mask_spans,
    and tool_steps[k]["observation"] == full_text[a:b]; such rows must be 100% in-vocabulary and unchanged by the runtime's
    symbol normaliser (otherwise the training runtime would discard the mask)
  * a row with more than 1 % characters outside the model vocabulary is dropped
  * validation / dev / test splits of the sources are never read (file finder + id guards)
  * splits are by GROUP (hash of the group key; datasets without natural groups hash the row id): heldout 3 %, dev 1 %, train 96 %

Nothing is fetched here; --raw-dir holds files that were downloaded elsewhere.  Output files are only ever created, never
overwritten: if a target exists with a different sha256 the new content is kept under a .new-<sha8> name and the run fails (exit 3).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import heapq
import itertools
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import unicodedata
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, replace as dc_replace
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Iterator

HERE = Path(__file__).resolve().parent

MAX_LEN = 2048                    # model context (characters); a row needs len(full_text) + 1 <= MAX_LEN
DEFAULT_CAP = 1900                # rows are kept at <= 1,900 characters
OOV_MAX_FRAC = 0.01               # rows with more than 1 % characters outside the vocabulary are dropped
INSTR = "条件と根拠を順に考えてください。"
FENCE = chr(96) * 3
SPLIT_PCT = {"heldout": 3, "dev": 1}      # share of GROUPS; the rest (96 %) is train

REQUIRED_FIELDS = ("id", "source", "license", "split", "training_category", "family", "structural_template", "difficulty",
                   "_v46_group", "prompt", "target", "full_text", "loss_mask_spans", "tool_steps",
                   "reference_final_answer", "answer_type", "attribution", "adoption", "adoption_note")


# ====================================================================================================== small helpers
def h(*parts) -> int:
    return int(hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12], 16)


def split_of(group_key: str) -> str:
    v = h("split-v1", group_key) % 100
    if v < SPLIT_PCT["heldout"]:
        return "heldout"
    if v < SPLIT_PCT["heldout"] + SPLIT_PCT["dev"]:
        return "dev"
    return "train"


def pick(options, *key):
    return options[h(*key) % len(options)]


def clean_text(s) -> str:
    """Universal newlines; nothing else is touched (code and quotes must stay verbatim)."""
    return str(s).replace("\r\n", "\n").replace("\r", "\n")


def clean_ws(s) -> str:
    return " ".join(str(s).split())


_KANA = re.compile(r"[\u3040-\u30ff]")
_HAN = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")


def detect_lang(s: str) -> str:
    kana, han, latin = len(_KANA.findall(s)), len(_HAN.findall(s)), len(_LATIN.findall(s))
    letters = kana + han + latin
    if letters == 0:
        return "other"
    if kana and kana / letters > 0.03:
        return "ja"
    if han / letters > 0.3:
        return "zh" if not kana else "ja"
    if latin / letters > 0.5:
        return "en"
    return "other"


class Reject(Exception):
    """One input record cannot become a row; .reason is the (counted) why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# ====================================================================================================== vocabulary
# Same map as qwenputer_controller_v30._BASE_NORM_DICT (+ the full-width ASCII loop): the training runtime applies it to every
# row before tokenising; test_convert.py compares this copy with the controller source when that file is reachable.
_NORM = {
    "ϵ": "ε", "ϕ": "φ", "ϑ": "θ", "ϰ": "κ", "ϱ": "ρ", "ς": "σ", "ο": "o",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P",
    "Τ": "T", "Υ": "Y", "Χ": "X", "＞": ">", "＜": "<", "⇔": "↔", "↦": "→", "⊨": "⊢", "⊤": "T", "∅": "φ", "≢": "≠",
    "÷": "/", "⋅": "・", "∙": "・", "⟨": "(", "⟩": ")", "⟪": "(", "⟫": ")", "⟦": "[", "⟧": "]", "⌈": "[", "⌉": "]",
    "⌊": "[", "⌋": "]", "−": "-",
}
for _c in range(0xFF01, 0xFF5F):
    if chr(_c) not in ("（", "）", "！", "？", "：", "～"):
        _NORM[chr(_c)] = chr(_c - 0xFEE0)
NORMALIZE = str.maketrans(_NORM)


def load_vocab(path=None, drive_root=None):
    """(frozenset of single characters, source).  Search order: explicit path, $GEOCPU_VOCAB, next to this file,
    src/v44fix/base_metadata.json in any parent directory, then checks.load_vocab(drive_root) (finds base_metadata.json inside the
    bundle zip on Drive; never derives a vocabulary from the corpus).  base_metadata.json["vocab"] maps token -> id."""
    cands = []
    if path:
        cands.append(Path(path))
    if os.environ.get("GEOCPU_VOCAB"):
        cands.append(Path(os.environ["GEOCPU_VOCAB"]))
    cands.append(HERE / "base_metadata.json")
    cands.extend(up / "src" / "v44fix" / "base_metadata.json" for up in HERE.parents)
    for p in cands:
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            v = data["vocab"] if isinstance(data, dict) and "vocab" in data else data
            keys = v.keys() if isinstance(v, dict) else v
            return frozenset(k for k in keys if isinstance(k, str) and len(k) == 1), str(p)
    ext = _load_external_checks()
    if ext is not None and callable(getattr(ext, "load_vocab", None)):
        try:
            root = drive_root or os.environ.get("GEOCPU_DRIVE_ROOT") or getattr(ext, "DRIVE_ROOT", "")
            info = ext.load_vocab(root, allow_derive=False)
            if len(info.chars) >= 1000 and not getattr(info, "derived", False):
                return frozenset(info.chars), "checks.load_vocab:" + str(info.source)
        except Exception:                                        # noqa: BLE001
            pass
    raise FileNotFoundError("model vocabulary not found: pass --vocab base_metadata.json (src/v44fix/base_metadata.json) or --drive-root")


# ====================================================================================================== code sandbox (builtin)
ALLOWED_MODULES = frozenset("""math cmath itertools functools collections re string json heapq bisect statistics fractions decimal
typing dataclasses enum copy operator textwrap abc numbers array queue contextlib unicodedata difflib pprint random datetime
calendar hashlib base64 binascii struct zlib csv io""".split())
FORBIDDEN_CALLS = frozenset("open exec eval compile input __import__ breakpoint exit quit help".split())
FORBIDDEN_ATTRS = frozenset("""open system popen __subclasses__ __globals__ __builtins__ __code__ __bases__ __mro__ __import__
__loader__ __closure__ __getattribute__ exec eval""".split())
MAX_CODE_CHARS = 6000


def static_code_check(code: str):
    """(ok, reason).  Allow-list: stdlib modules that cannot touch files / processes / network, no input(), no eval/exec/open,
    no dunder escapes, `random` only with a seed, `datetime` only without now()/today()."""
    if len(code) > MAX_CODE_CHARS:
        return False, "code_too_long"
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return False, "syntax_error"
    uses_random = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                root = a.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    return False, "import_not_allowed:" + root
                uses_random |= root == "random"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_MODULES:
                return False, "import_not_allowed:" + (root or "relative")
            uses_random |= root == "random"
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in FORBIDDEN_CALLS:
                return False, "uses_input" if f.id == "input" else ("uses_file_io" if f.id == "open" else "forbidden_call:" + f.id)
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                return False, "forbidden_attr:" + node.attr
            if node.attr in ("now", "today", "utcnow", "fromtimestamp"):
                return False, "nondeterministic:datetime"
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "__import__"):
            return False, "forbidden_name:" + node.id
    if uses_random and "seed(" not in code:
        return False, "nondeterministic:random"
    return True, ""


def run_python_sandbox(code: str, timeout: float = 5.0, max_out: int = 20000) -> dict:
    """Run `code` as a script in a throw-away directory, stdin closed, minimal environment, fixed hash seed, wall/cpu/memory/file
    size limits (POSIX).  Returns {ok, stdout, stderr, reason}.  The caller has already applied static_code_check()."""
    if not code.endswith("\n"):
        code += "\n"
    with tempfile.TemporaryDirectory(prefix="v49sbx_") as d:
        script, out_p, err_p = (os.path.join(d, n) for n in ("main.py", "out.bin", "err.bin"))
        with open(script, "w", encoding="utf-8", newline="\n") as f:
            f.write(code)
        env = {"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1",
               "LC_ALL": "C.UTF-8"}
        for k in ("PATH", "SYSTEMROOT", "SystemRoot", "TEMP", "TMP"):
            if k in os.environ:
                env[k] = os.environ[k]
        kw = {}
        if os.name == "posix":
            def _limits():                                     # pragma: no cover - runs in the child
                import resource
                cpu = int(timeout) + 1
                resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
                resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
                resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 21, 1 << 21))
            kw["preexec_fn"] = _limits
        rc, timed_out = None, False
        with open(out_p, "wb") as fo, open(err_p, "wb") as fe:
            try:
                rc = subprocess.run([sys.executable, "-S", "-B", script], stdin=subprocess.DEVNULL, stdout=fo, stderr=fe,
                                    cwd=d, env=env, timeout=timeout, **kw).returncode
            except subprocess.TimeoutExpired:
                timed_out = True
        with open(out_p, "rb") as fo:
            raw = fo.read(max_out + 1)
        with open(err_p, "rb") as fe:
            err = fe.read(4000).decode("utf-8", "replace")
    stdout = raw[:max_out].decode("utf-8", "replace").replace("\r\n", "\n")
    if timed_out:
        return {"ok": False, "stdout": stdout, "stderr": err, "reason": "timeout"}
    if len(raw) > max_out:
        return {"ok": False, "stdout": stdout, "stderr": err, "reason": "output_too_large"}
    if rc != 0:
        last = err.strip().splitlines()[-1] if err.strip() else ""
        m = re.match(r"([A-Za-z_][\w.]*)", last)
        return {"ok": False, "stdout": stdout, "stderr": err, "reason": "runtime_error:" + (m.group(1) if m else f"rc{rc}")}
    return {"ok": True, "stdout": stdout, "stderr": err, "reason": ""}


# ====================================================================================================== checks (builtin + optional checks.py)
def _load_external_checks():
    try:
        if str(HERE) not in sys.path:
            sys.path.insert(0, str(HERE))
        import checks as ext                                     # sibling module written by the pipeline's checks stage
        return ext
    except Exception:
        return None


def _coerce_sandbox(res) -> dict:
    if isinstance(res, (tuple, list)) and len(res) >= 2:
        return {"ok": bool(res[0]), "stdout": str(res[1]), "reason": str(res[2]) if len(res) > 2 else ""}
    get = res.get if isinstance(res, dict) else (lambda k, d=None: getattr(res, k, d))
    ok = next((get(k) for k in ("ok", "passed", "success") if get(k) is not None), None)
    out = next((get(k) for k in ("stdout", "output", "out") if get(k) is not None), None)
    if ok is None or out is None:
        raise ValueError("unrecognised checks.code_sandbox_check result")
    reason = next((get(k) for k in ("reason", "error", "why") if get(k)), "")
    return {"ok": bool(ok), "stdout": str(out), "reason": str(reason)}


class Checks:
    """Vocabulary / OOV / sandbox services.  `use_external=True` runs python through checks.code_sandbox_check(profile="wide") when a
    sibling checks.py provides it AND it passes a start-up self-test (otherwise the builtin sandbox is used).  OOV is always measured per
    row here with the runtime normaliser (checks.oov_stats is a corpus-level report over rows: run it over the output files).
    The backend actually used is recorded in .backends (and in every summary)."""

    def __init__(self, vocab=None, vocab_path=None, max_chars=DEFAULT_CAP, run_code=True, sandbox_timeout=5.0, use_external=True,
                 python_require_stored_output=False, tengentoppa_steps=True, drive_root=None):
        if vocab is None:
            vocab, vocab_path = load_vocab(vocab_path, drive_root)
        self.vocab = frozenset(vocab)
        self.vocab_path = vocab_path
        self.max_chars = min(int(max_chars), MAX_LEN - 1)
        self.run_code = run_code
        self.sandbox_timeout = sandbox_timeout
        self.python_require_stored_output = python_require_stored_output   # keep python rows only if the answer's own expected output matches
        self.tengentoppa_steps = tengentoppa_steps                        # include the 'reasoning' steps when the row fits
        self.backends = {"oov": "builtin", "sandbox": "builtin" if run_code else "disabled", "static_check": "builtin"}
        self._ext_sandbox = None
        if use_external:
            ext = _load_external_checks()
            if ext is not None and run_code and callable(getattr(ext, "code_sandbox_check", None)):
                # checks.code_sandbox_check(code, timeout, profile=...) -> (ok, stdout, error); profile "wide" = stdlib allow-list WITH classes
                # (the default "executor" profile is the model's own math worker: no classes, no attribute assignment)
                def _ext(code, _fn=ext.code_sandbox_check, _t=sandbox_timeout):
                    return _fn(code, _t, profile="wide")
                try:
                    r = _coerce_sandbox(_ext("print(1 + 1)\n"))
                    if r["ok"] and r["stdout"].strip() == "2":
                        self._ext_sandbox = _ext
                        self.backends["sandbox"] = "checks.code_sandbox_check(profile=wide)"
                    else:
                        self.backends["sandbox"] = "builtin (checks.code_sandbox_check failed self-test)"
                except Exception as e:                           # noqa: BLE001
                    self.backends["sandbox"] = f"builtin (checks.code_sandbox_check unusable: {type(e).__name__})"
            if ext is not None and getattr(ext, "RUNTIME_NORMALIZE", None) == NORMALIZE:
                self.backends["oov"] = "builtin per-row (same runtime normaliser as checks.RUNTIME_NORMALIZE; checks.oov_stats is corpus level)"

    # --- vocabulary
    def oov_list(self, text: str):
        return [c for c in text.translate(NORMALIZE) if c not in self.vocab]

    def _oov(self, text: str):
        t = text.translate(NORMALIZE)                             # 1:1 character map, so len() is unchanged
        bad = set(t) - self.vocab                                 # set arithmetic runs at C speed; counting only when something is OOV
        return (sum(t.count(c) for c in bad) if bad else 0), bad

    def oov_stats(self, text: str) -> dict:
        n_oov, bad = self._oov(text)
        return {"n_chars": len(text), "n_oov": n_oov, "frac": n_oov / max(1, len(text)), "bad": sorted(bad)[:20]}

    def oov_count(self, text: str) -> int:
        return self._oov(text)[0]

    def normalizer_changes(self, text: str) -> bool:
        return text.translate(NORMALIZE) != text

    # --- code
    def run_python(self, code: str) -> dict:
        if not self.run_code:
            return {"ok": False, "stdout": "", "reason": "sandbox_disabled"}
        if self._ext_sandbox is not None:
            try:
                return _coerce_sandbox(self._ext_sandbox(code))
            except Exception as e:                               # noqa: BLE001
                return {"ok": False, "stdout": "", "reason": "checks_error:" + type(e).__name__}
        return run_python_sandbox(code, timeout=self.sandbox_timeout)


# ====================================================================================================== rows, stats, validation
@dataclass
class Spec:
    key: str
    hf: str
    license: str
    source: str
    attribution: str
    adoption: str
    adoption_note: str
    find: Callable
    reader: Callable
    convert: Callable


class Stats:
    """Per-dataset counters.  Invariant: n_in == n_out + sum(rejects)."""

    MAX_EXAMPLES = 5

    def __init__(self, key: str):
        self.key = key
        self.n_in = 0
        self.n_out = 0
        self.rejects: Counter = Counter()
        self.families: Counter = Counter()
        self.splits: Counter = Counter()
        self.family_splits: dict = defaultdict(Counter)
        self.extra: Counter = Counter()
        self.reject_examples: dict = {}

    def reject(self, reason: str, ident: str = "", rec=None):
        self.rejects[reason] += 1
        ex = self.reject_examples.setdefault(reason, []) if (reason in self.reject_examples or len(self.reject_examples) < 200) else None
        if ex is not None and len(ex) < self.MAX_EXAMPLES:
            ex.append({"id": ident, "excerpt": _excerpt(rec)})

    ACCEPT_FIELDS = (("verification", "verification_"), ("code_lang", "code_text_lang_"), ("answer_lang", "code_text_answer_lang_"),
                     ("trim_level", "trim_level_"), ("user_lang", "accepted_user_lang_"), ("answer_format", "answer_format_"),
                     ("has_python_block", "has_python_block_"))

    def accept(self, row: dict):
        self.n_out += 1
        self.families[row["family"]] += 1
        self.splits[row["split"]] += 1
        self.family_splits[row["family"]][row["split"]] += 1
        for field, prefix in self.ACCEPT_FIELDS:                  # counters over EMITTED rows only
            if field in row:
                self.extra[prefix + str(row[field])] += 1


def build_row(*, id, family, template, group, category, prompt, target, ref, answer_type, difficulty, spec: Spec, split,
              spans=None, steps=None, **extra) -> dict:
    row = {
        "id": id, "source": spec.source, "license": spec.license, "split": split, "training_category": category,
        "family": family, "structural_template": template, "difficulty": difficulty, "_v46_group": group,
        "prompt": prompt, "target": target, "full_text": prompt + target,
        "loss_mask_spans": [list(s) for s in (spans or [])], "tool_steps": steps or [],
        "reference_final_answer": ref, "answer_type": answer_type, "attribution": spec.attribution,
        "adoption": spec.adoption, "adoption_note": spec.adoption_note,
    }
    row.update(extra)
    return row


def validate_row(row: dict, ck: Checks):
    """None if the row is a valid GeoCPU row, else a short reason."""
    for k in REQUIRED_FIELDS:
        if k not in row:
            return "missing:" + k
    p, t, f = row["prompt"], row["target"], row["full_text"]
    if not all(isinstance(x, str) for x in (p, t, f)) or not p or not t:
        return "bad_text"
    if f != p + t:
        return "full_text_mismatch"
    if len(f) > ck.max_chars:
        return "too_long"
    if len(f) + 1 > MAX_LEN:
        return "too_long_ctx"
    if any(0xD800 <= ord(c) <= 0xDFFF for c in f):
        return "surrogate"
    if row["split"] not in ("train", "dev", "heldout"):
        return "bad_split"
    spans, steps = row["loss_mask_spans"], row["tool_steps"]
    prev = len(p)
    for s in spans:
        if not (isinstance(s, (list, tuple)) and len(s) == 2 and all(isinstance(x, int) for x in s)):
            return "bad_span"
        a, b = s
        if not (prev <= a < b <= len(f)):
            return "span_out_of_order_or_range"
        prev = b
    if spans:
        if len(steps) != len(spans) or any(f[a:b] != st.get("observation") for (a, b), st in zip(spans, steps)):
            return "mask_does_not_cover_observation"
        if ck.normalizer_changes(f) or ck.oov_count(f):
            return "masked_row_not_clean"
    if ck.oov_stats(f)["frac"] > OOV_MAX_FRAC:
        return "oov_frac"
    for q in row.get("quotes", []) or []:
        if q not in p:
            return "quote_not_in_prompt"
    return None


def run_converter(records: Iterable, stats: Stats, fn: Callable, ck: Checks) -> Iterator[dict]:
    """fn(idx, rec) -> row | raises Reject.  Every record ends as a validated row or one counted reject."""
    for idx, rec in records:
        stats.n_in += 1
        ident = f"{stats.key}:{idx}"
        try:
            if not isinstance(rec, dict):
                raise Reject("not_an_object")
            row = fn(idx, rec)
            why = validate_row(row, ck)
            if why:
                raise Reject("invalid_row:" + why)
            row["n_oov"] = ck.oov_count(row["full_text"])
        except Reject as r:
            stats.reject(r.reason, ident, rec)
            continue
        except Exception as e:                                   # noqa: BLE001  one odd record must not kill a 1 GB run
            stats.reject("exception:" + type(e).__name__, ident, rec)
            continue
        stats.accept(row)
        yield row


def _excerpt(rec) -> str:
    try:
        return json.dumps(rec, ensure_ascii=False)[:160]
    except Exception:                                            # noqa: BLE001
        return repr(rec)[:160]


# ====================================================================================================== streaming readers
def iter_jsonl(path) -> Iterator[dict]:
    with open(path, encoding="utf-8", errors="replace", newline="\n") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj = json.loads(line)
                except ValueError:
                    obj = {"__bad_json__": line[:80]}
                yield obj


def iter_json_array(path, chunk: int = 1 << 20) -> Iterator[dict]:
    """Objects of a top-level JSON array, one at a time (falls back to JSONL when the file starts with '{')."""
    dec = json.JSONDecoder()
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.read(1)
        while first and first.isspace():
            first = f.read(1)
        if not first:
            return
        if first == "{":
            f.seek(0)
            yield from iter_jsonl(path)
            return
        if first != "[":
            raise ValueError("not a JSON array: " + str(path))
        buf, i, eof = "", 0, False
        while True:
            while True:
                while i < len(buf) and (buf[i].isspace() or buf[i] == ","):
                    i += 1
                if i < len(buf):
                    break
                if eof:
                    return
                data = f.read(chunk)
                buf, i = data, 0
                eof = not data
            if buf[i] == "]":
                return
            while True:
                try:
                    obj, j = dec.raw_decode(buf, i)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    data = f.read(chunk)
                    if not data:
                        eof = True
                    buf, i = buf[i:] + data, 0
            yield obj
            i = j


def iter_parquet(path, columns=None, batch_size: int = 256) -> Iterator[dict]:
    import pyarrow.parquet as pq                                 # present on Colab; pip install -q pyarrow otherwise
    pf = pq.ParquetFile(str(path))
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield from batch.to_pylist()


def _find(raw_dir, patterns, root_only=()):
    root = Path(raw_dir)
    found = set()
    for pat in patterns:
        found.update(p for p in root.glob(pat) if p.is_file())
    out = []
    for p in sorted(found):
        rel = "/".join(p.relative_to(root).parts).lower()
        if any(part.startswith("do_not_train") for part in p.relative_to(root).parts):
            continue                                              # acquire.py keeps dev / validation files there
        if re.search(r"(?:^|[/_.-])(dev|validation|val|test)(?:[/_.-]|$)", rel):
            continue                                              # never train on validation / dev / test files
        out.append(p)
    return out


def _limited(it: Iterable, limit):
    return itertools.islice(enumerate(it), limit) if limit is not None else enumerate(it)


def find_jaquad(d):
    return _find(d, ["**/jaquad_train_*.json"])


def find_coding(d):
    return _find(d, ["**/train-*.parquet"])


def find_tengentoppa(d):
    return _find(d, ["**/completed_sets.jsonl"])


def find_hotpot(d):
    files = [p for p in _find(d, ["**/distractor/train-*.parquet"])]
    if not files:                                                 # flat folder: accept only if it is clearly not fullwiki
        files = [p for p in _find(d, ["train-*.parquet", "**/train-*.parquet"]) if "fullwiki" not in str(p).lower()]
    return files


def find_toolace(d):
    return _find(d, ["**/data.json"])


def find_wizard(d):
    return _find(d, ["**/train-*.parquet"])


def read_jaquad(files, limit=None):
    n = 0
    for fp in files:
        d = json.loads(Path(fp).read_text(encoding="utf-8"))
        for art in d.get("data", []):
            title = str(art.get("title", "")).strip()
            for para in art.get("paragraphs", []):
                ctx = para.get("context", "")
                for qa in para.get("qas", []):
                    if limit is not None and n >= limit:
                        return
                    yield n, {"id": qa.get("id"), "title": title, "context": ctx, "question": qa.get("question", ""),
                              "question_type": qa.get("question_type", ""), "answers": qa.get("answers", [])}
                    n += 1


def _chain(files, reader):
    for fp in files:
        yield from reader(fp)


def read_messages_parquet(files, limit=None):
    return _limited(_chain(files, lambda p: iter_parquet(p, columns=["messages"])), limit)


def read_tengentoppa(files, limit=None):
    return _limited(_chain(files, iter_jsonl), limit)


def read_hotpot(files, limit=None):
    cols = ["id", "question", "answer", "type", "level", "supporting_facts", "context"]
    return _limited(_chain(files, lambda p: iter_parquet(p, columns=cols)), limit)


def read_toolace(files, limit=None):
    return _limited(_chain(files, iter_json_array), limit)


# ====================================================================================================== (a) JaQuAD
JAQUAD_MAX_PASSAGE = 700
PASSAGE_HEAD = ["次の文章を読んで質問に答えてください。", "文章を読んで、文章に書かれていることだけで質問に答えてください。",
                "以下の文章を読んで質問に答えてください。"]


def sentence_spans(body: str):
    """[(start, end)]: a sentence ends after '。' (plus closing brackets) or at a line break; blank pieces are dropped."""
    spans, s = [], 0
    for m in re.finditer(r"。[」』）)]*|\n", body):
        spans.append((s, m.end()))
        s = m.end()
    if s < len(body):
        spans.append((s, len(body)))
    return [(a, b) for a, b in spans if body[a:b].strip()]


def _answers_list(a):
    """JaQuAD raw: [{text, answer_start, answer_type}]; datasets-server view: {text: [..], answer_start: [..], answer_type: [..]}."""
    if isinstance(a, dict) and isinstance(a.get("text"), list):
        return [{"text": t, "answer_start": s, "answer_type": (a.get("answer_type") or [""] * len(a["text"]))[i]}
                for i, (t, s) in enumerate(zip(a["text"], a["answer_start"]))]
    if isinstance(a, dict):
        return [a]
    return list(a or [])


def _jaquad_one(idx, rec, st: Stats, ck: Checks, spec: Spec):
    qid = str(rec.get("id") or "")
    if qid[:3] in ("de-", "te-"):
        raise Reject("not_train_id")                             # dev / test ids never enter training
    title = str(rec.get("title") or "").strip()
    ctx = str(rec.get("context") or "")
    q = clean_ws(rec.get("question") or "")
    answers = _answers_list(rec.get("answers"))
    if not qid or not title or not ctx or not q:
        raise Reject("missing_field")
    if not answers:
        raise Reject("no_answer")
    if "\r" in ctx:
        raise Reject("context_has_cr")
    a = answers[0]
    raw_text, start = str(a.get("text", "")), int(a.get("answer_start", -1))
    if not raw_text.strip() or start < 0:
        raise Reject("empty_answer")
    if ctx[start:start + len(raw_text)] != raw_text:
        raise Reject("offset_mismatch")
    lead = len(ctx) - len(ctx.lstrip())
    body = ctx.strip()
    ans = raw_text.strip()
    s0 = start - lead + (len(raw_text) - len(raw_text.lstrip()))
    e0 = s0 + len(ans)
    if s0 < 0 or e0 > len(body):
        raise Reject("answer_outside_body")
    if "\n" in ans:
        raise Reject("answer_has_newline")
    if len(body) > JAQUAD_MAX_PASSAGE:
        raise Reject("passage_too_long")
    hit = [sp for sp in sentence_spans(body) if sp[0] <= s0 and e0 <= sp[1]]
    if len(hit) != 1:
        raise Reject("answer_spans_sentences")
    quote = body[hit[0][0]:hit[0][1]].strip()
    if ans not in quote or len(quote) < 4 or "\n" in quote:
        raise Reject("quote_check")
    if ck.oov_count(quote + ans):
        raise Reject("oov_in_quote_or_answer")
    head = pick(PASSAGE_HEAD, "jaq-head", qid)
    prompt = f"問題: {head}\n文章（{title}）: {body}\n質問: {q}\n{INSTR}"
    target = f"\n根拠: 「{quote}」\n答え: {ans}。"
    gk = "jaquad:" + title
    return build_row(id=f"v49acq_jaquad_{qid}", family="jaquad_human", template="acq_jaquad_span", group="v49_jaquad_human",
                     category="evidence", prompt=prompt, target=target, ref=ans, answer_type="span", difficulty=2, spec=spec,
                     split=split_of(gk), quotes=[quote], jaquad_question_type=str(rec.get("question_type") or ""),
                     jaquad_answer_type=str(a.get("answer_type") or ""), group_key=gk, quality_flag="human annotated (Skelter Labs)")


def convert_jaquad(records, stats: Stats, ck: Checks, spec: Spec):
    return run_converter(records, stats, lambda i, r: _jaquad_one(i, r, stats, ck, spec), ck)


# ====================================================================================================== fenced code parsing
LANG_ALIAS = {"py": "python", "python3": "python", "py3": "python", "c++": "cpp", "cxx": "cpp", "cc": "cpp", "js": "javascript",
              "node": "javascript", "ts": "typescript", "sh": "bash", "shell": "bash", "zsh": "bash", "c#": "csharp",
              "cs": "csharp", "rs": "rust", "golang": "go", "rb": "ruby", "kt": "kotlin"}
OUTPUT_LANGS = frozenset("output console stdout result results".split())
CODE_LANGS = frozenset("""cpp c rust java javascript typescript go csharp ruby php swift kotlin scala sql bash r perl lua haskell dart
objectivec matlab julia fortran cobol elixir clojure groovy powershell vb vbnet assembly asm verilog""".split())


def _norm_lang(tag: str) -> str:
    tag = tag.strip().strip("{}.").lower()
    tag = tag.split()[0] if tag else ""
    tag = tag.strip("{}.")
    return LANG_ALIAS.get(tag, tag)


def parse_fences(text: str):
    """[{lang, code, start, end}] for the fenced blocks of `text`, or None when the fences are unbalanced / malformed."""
    lines = text.split("\n")
    offs, o = [], 0
    for ln in lines:
        offs.append(o)
        o += len(ln) + 1
    blocks, open_i, lang, indent = [], None, "", ""
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith(FENCE):
            continue
        if open_i is None:
            tag = s[3:].strip()
            if "`" in tag:
                return None
            open_i, lang, indent = i, _norm_lang(tag), ln[:len(ln) - len(ln.lstrip())]
        else:
            if s != FENCE:
                return None
            code = textwrap.dedent("\n".join(lines[open_i + 1:i]))
            blocks.append({"lang": lang, "code": code, "start": offs[open_i], "end": offs[i] + len(ln)})
            open_i = None
    return None if open_i is not None else blocks


def _numbers(s: str):
    out = []
    for m in re.findall(r"-?\d[\d,]*(?:\.\d+)?", s):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _stored_outputs(asst: str, blocks, py):
    """Stored outputs of the answer that follow the python block: blocks tagged output/console/stdout/result, or untagged/text
    blocks directly after a label such as 出力: / Output:."""
    res = []
    for b in blocks:
        if b["start"] < py["end"] or b is py:
            continue
        if b["lang"] in OUTPUT_LANGS:
            res.append(b["code"])
        elif b["lang"] in ("", "text", "plaintext", "txt"):
            before = asst[max(0, b["start"] - 80):b["start"]]
            if re.search(r"(出力|実行結果|結果|output|result|expected)[^\n]{0,40}[:：]?\s*$", before, re.I):
                res.append(b["code"])
    return res


def _same_output(a: str, b: str) -> bool:
    na = [ln.rstrip() for ln in a.strip().splitlines()]
    nb = [ln.rstrip() for ln in b.strip().splitlines()]
    return na == nb


def _prose_after(asst: str, blocks, py) -> str:
    pieces, cur = [], py["end"]
    for b in blocks:
        if b["start"] >= py["end"]:
            pieces.append(asst[cur:b["start"]])
            cur = b["end"]
    pieces.append(asst[cur:])
    return "\n".join(pieces)


_OUT_WORDS = re.compile(r"(出力|結果|表示|返し|戻り|prints?\b|printed|output|results?\b|returns?\b|displays?)", re.I)


def prose_number_contradiction(prose: str, stdout: str, code: str, user: str) -> bool:
    """True if a sentence of the prose that talks about the output states a number that occurs in neither stdout, code nor
    question (single digits are ignored: list numbering)."""
    ok = _numbers(stdout) + _numbers(code) + _numbers(user)
    for sent in re.split(r"[。\n]|(?<=[.!?])\s", prose):
        if not _OUT_WORDS.search(sent):
            continue
        for n in _numbers(sent):
            if abs(n) < 10 and float(n).is_integer():
                continue
            if not any(abs(n - m) <= 1e-9 * max(1.0, abs(n)) for m in ok):
                return True
    return False


# ====================================================================================================== (b) Synthetic-JP-EN-Coding
PLAN_FALLBACK = "コードを書いて実行し、出力を確かめる"


def pick_plan(asst: str, first_fence_start: int, ck: Checks) -> str:
    """One short Japanese sentence taken from the prose before the first code block; a fixed sentence otherwise."""
    for sent in re.split(r"[。\n]", asst[:first_fence_start]):
        sent = sent.strip()
        if 8 <= len(sent) <= 80 and _KANA.search(sent) and not re.search(r"[`*#<>|:：]", sent) and not ck.oov_count(sent):
            return sent
    return PLAN_FALLBACK


def _coding_one(idx, rec, st: Stats, ck: Checks, spec: Spec):
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise Reject("no_messages")
    ui = next((i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"), None)
    if ui is None:
        raise Reject("no_user_message")
    ai = next((i for i in range(ui + 1, len(msgs)) if isinstance(msgs[i], dict) and msgs[i].get("role") == "assistant"), None)
    if ai is None:
        raise Reject("no_assistant_message")
    if len(msgs) > ai + 1:
        st.extra["multiturn_first_pair_only"] += 1
    user, asst = clean_text(msgs[ui].get("content") or "").strip(), clean_text(msgs[ai].get("content") or "").strip()
    if not user or not asst:
        raise Reject("empty_message")
    if len(user) > ck.max_chars - 60:
        raise Reject("row_too_long")
    blocks = parse_fences(asst)
    if blocks is None:
        raise Reject("unbalanced_fence")
    if not blocks:
        raise Reject("no_code_block")
    ulang = detect_lang(user)
    py = [b for b in blocks if b["lang"] == "python"]
    if py:
        return _coding_python_row(idx, user, asst, blocks, py, ulang, st, ck, spec)
    return _coding_text_row(idx, user, asst, blocks, ulang, st, ck, spec)


def _coding_python_row(idx, user, asst, blocks, py, ulang, st, ck, spec):
    chosen, first_reason = None, None
    for b in py:
        ok, why = static_code_check(b["code"].rstrip())
        if not ok:
            first_reason = first_reason or why
            continue
        if "print(" not in b["code"]:
            first_reason = first_reason or "no_print"
            continue
        chosen = b
        break
    if chosen is None:
        raise Reject("static:" + (first_reason or "no_candidate"))
    code = chosen["code"].strip("\n").rstrip()
    if FENCE in code:
        raise Reject("fence_in_code")
    if len(user) + len(code) + 150 > ck.max_chars:
        raise Reject("row_too_long")
    res = ck.run_python(code)
    if not res["ok"]:
        raise Reject("sandbox_failed:" + (res.get("reason") or "unknown"))
    stdout = clean_text(res["stdout"])
    if not stdout.strip():
        raise Reject("empty_stdout")
    if not stdout.endswith("\n"):
        stdout += "\n"
    if FENCE in stdout:
        raise Reject("fence_in_stdout")
    if re.search(r"0x[0-9a-fA-F]{6,}", stdout):
        raise Reject("address_in_stdout")                        # object repr: not reproducible
    lines = [ln for ln in stdout.splitlines() if ln.strip()]
    ans = lines[-1].strip()
    if len(ans) > 120:
        raise Reject("answer_line_too_long")
    stored = _stored_outputs(asst, blocks, chosen)
    if stored and not any(_same_output(s, stdout) for s in stored):
        raise Reject("stored_output_mismatch")
    verification = "stored_output_equals_sandbox" if stored else "sandbox_only_no_stored_output"
    if not stored and ck.python_require_stored_output:
        raise Reject("no_stored_output_to_corroborate")
    if prose_number_contradiction(_prose_after(asst, blocks, chosen), stdout, code, user):
        raise Reject("prose_number_contradicts_output")
    plan = pick_plan(asst, blocks[0]["start"], ck)
    prompt = f"問題: {user}\n道具: Python\n思考:"
    head = f"\n計画: {plan}。\n{FENCE}python\n{code}\n{FENCE}"
    obs = f"\n{FENCE}output\n{stdout}{FENCE}\n"
    tail = f"確認: 出力の最後の行を答えとする。\n答え: {ans}。"
    target = head + obs + tail
    a = len(prompt) + len(head)
    gk = f"coding:{idx}"
    return build_row(id=f"v49acq_coding_{idx:07d}", family="coding_pytool", template="acq_coding_pytool", group="v49_coding_pytool",
                     category="python", prompt=prompt, target=target, ref=ans, answer_type="output_last_line", difficulty=3,
                     spec=spec, split=split_of(gk), spans=[(a, a + len(obs))],
                     steps=[{"index": 0, "code": code, "stdout": stdout, "observation": obs, "success": True}],
                     python_code=code, expected_stdout=stdout, plan_line=plan, verification=verification,
                     sandbox_backend=ck.backends["sandbox"], user_lang=ulang, group_key=gk,
                     quality_flag="llm-generated instruction+code; output replaced by sandbox stdout")


def _coding_text_row(idx, user, asst, blocks, ulang, st, ck, spec):
    if any(not b["lang"] for b in blocks):
        raise Reject("untagged_code_fence")
    code_blocks = [b for b in blocks if b["lang"] in CODE_LANGS]
    if not code_blocks:
        raise Reject("no_code_language_block")
    prompt = f"問題: {user}\n{INSTR}"
    target = "\n" + asst
    if len(prompt) + len(target) > ck.max_chars:
        raise Reject("row_too_long")
    lang = code_blocks[0]["lang"]
    alang = detect_lang(asst)
    gk = f"coding:{idx}"
    return build_row(id=f"v49acq_coding_{idx:07d}", family="code_text", template="acq_coding_codetext", group="v49_coding_codetext",
                     category="code", prompt=prompt, target=target, ref="", answer_type="free_code_text", difficulty=3, spec=spec,
                     split=split_of(gk), code_lang=lang, n_code_blocks=len(blocks), user_lang=ulang, answer_lang=alang,
                     verification="unverified_no_execution", group_key=gk,
                     quality_flag="llm-generated; code not executed (unverified)")


def convert_coding(records, stats: Stats, ck: Checks, spec: Spec):
    return run_converter(records, stats, lambda i, r: _coding_one(i, r, stats, ck, spec), ck)


# ====================================================================================================== (c) Tengentoppa
REFUSAL_RE = re.compile(r"^(?:申し訳|すみません|ごめんなさい|残念ながら|私はAI|私は人工知能|AIとして|言語モデルとして|I'm sorry|I am sorry|I cannot|I can't|As an AI)")
REFUSAL_ANY = re.compile(r"(お答えできません|回答できません|対応できません|お手伝いできません|お答えすることができません)")
BOILER_RE = re.compile(r"^(?:もちろん|承知(?:いたし)?ました|かしこまりました|了解(?:しました|です)|はい、(?:もちろん|承知)|ご質問ありがとう|ご依頼ありがとう|以下に(?:回答|説明|示)|以下の通り)")
TERMINATORS = tuple("。！？」』）)")
TENGEN_FLAG = "qwen2.5-32b generated, unverified"


def latin_ratio(s: str) -> float:
    core = [c for c in s if not c.isspace()]
    return sum(1 for c in core if "A" <= c <= "Z" or "a" <= c <= "z") / max(1, len(core))


def answer_format(a: str):
    """'prose' (ends with a sentence terminator), 'fenced' (balanced fences, ends with the closing fence), 'json' (the whole answer is
    valid JSON), or None = cut off (max_tokens) / unfinished."""
    if a.endswith(TERMINATORS):
        return "prose"
    if a.endswith(FENCE) and parse_fences(a):
        return "fenced"
    if a[-1:] in ("}", "]"):
        try:
            json.loads(a)
            return "json"
        except ValueError:
            pass
    return None


def looks_repetitive(s: str) -> bool:
    b = s.encode("utf-8")
    return len(b) > 300 and len(zlib.compress(b, 6)) / len(b) < 0.22


def reads_as_steps(s: str) -> bool:
    return len(re.findall(r"(?m)^\s*(?:\d+[.．)）]|[-・*]\s|ステップ\s*\d)", s)) >= 2


def _tengentoppa_one(idx, rec, st: Stats, ck: Checks, spec: Spec):
    if "__bad_json__" in rec:
        raise Reject("bad_json_line")
    instr = clean_text(rec.get("instruction") or "").strip()
    refined = clean_text(rec.get("refined_answer") or "").strip()
    reasoning = clean_text(rec.get("reasoning") or "").strip()
    if not instr or not refined:
        raise Reject("empty_field")
    if REFUSAL_RE.match(refined[:40]) or REFUSAL_ANY.search(refined[:120]):
        raise Reject("refusal_opening")
    if BOILER_RE.match(refined[:40]):
        raise Reject("boilerplate_opening")
    if latin_ratio(refined) > 0.30:
        raise Reject("latin_heavy")
    fmt = answer_format(refined)
    if fmt is None:
        raise Reject("answer_not_terminated")
    if looks_repetitive(refined):
        raise Reject("repetitive")
    prompt = f"問題: {instr}\n{INSTR}"
    target = f"\n答え: {refined}"
    family, tmpl = "tengentoppa_answer", "acq_tengentoppa_answer"
    if len(prompt) + len(target) > ck.max_chars:
        raise Reject("row_too_long")
    if ck.tengentoppa_steps and reasoning and reads_as_steps(reasoning) and latin_ratio(reasoning) <= 0.30 and not looks_repetitive(reasoning):
        t2 = f"\n手順: {reasoning}\n答え: {refined}"
        if len(prompt) + len(t2) <= ck.max_chars:
            target, family, tmpl = t2, "tengentoppa_steps", "acq_tengentoppa_steps"
    gk = f"tengentoppa:{idx}"
    return build_row(id=f"v49acq_tengentoppa_{idx:07d}", family=family, template=tmpl, group="v49_tengentoppa", category="free_text",
                     prompt=prompt, target=target, ref=refined, answer_type="free_text", difficulty=2, spec=spec,
                     split=split_of(gk), group_key=gk, answer_format=fmt, has_python_block=FENCE + "python" in target,
                     quality_flag=TENGEN_FLAG)


def convert_tengentoppa(records, stats: Stats, ck: Checks, spec: Spec):
    return run_converter(records, stats, lambda i, r: _tengentoppa_one(i, r, stats, ck, spec), ck)


# ====================================================================================================== (d) HotpotQA (distractor, train only)
HOTPOT_LEVEL_NOTE = {0: "supporting paragraphs up to the last supporting sentence + 2 distractors (2 sentences)",
                     1: "same, distractors cut to 1 sentence", 2: "supporting paragraphs only",
                     3: "supporting sentences + paragraph opening", 4: "supporting sentences only"}
HOTPOT_HEAD = "次の英語の段落を読んで、質問に英語で短く答えてください。"


def _hotpot_one(idx, rec, st: Stats, ck: Checks, spec: Spec):
    rid = str(rec.get("id") or "")
    q, ans = clean_ws(rec.get("question") or ""), clean_ws(rec.get("answer") or "")
    sf, ctx = rec.get("supporting_facts") or {}, rec.get("context") or {}
    if not rid or not q or not ans:
        raise Reject("missing_field")
    by_title = {}
    for t, ss in zip(ctx.get("title", []), ctx.get("sentences", [])):
        by_title.setdefault(str(t), [clean_ws(s) for s in ss])
    supp = {}
    for t, i in zip(sf.get("title", []), sf.get("sent_id", [])):
        supp.setdefault(str(t), set()).add(int(i))
    if len(supp) != 2:
        raise Reject("not_two_support_paragraphs")
    for t, ids in supp.items():
        ss = by_title.get(t)
        if ss is None or any(i < 0 or i >= len(ss) or not ss[i] for i in ids):
            raise Reject("bad_supporting_fact")
    full_len = sum(len(" ".join(ss)) for ss in by_title.values()) + len(q) + 120
    if full_len <= ck.max_chars:
        st.extra["fit_untrimmed_all_paragraphs"] += 1
    distr = sorted((t for t, ss in by_title.items() if t not in supp and any(ss)), key=lambda t: h("hp-dist", rid, t))[:2]

    def text_of(t, lv):
        ss = by_title[t]
        if t in supp:
            ids = sorted(supp[t])
            if lv <= 2:
                keep = ss[:ids[-1] + 1]
            elif lv == 3:
                keep = ([ss[0]] if 0 not in ids else []) + [ss[i] for i in ids]
            else:
                keep = [ss[i] for i in ids]
        else:
            keep = ss[:2 if lv == 0 else 1]
        return " ".join(x for x in keep if x)

    last_len = None
    chosen = None
    for lv in range(5):
        titles = list(supp) + (distr if lv <= 1 else [])
        titles.sort(key=lambda t: h("hp-order", rid, t))
        paras = [(t, text_of(t, lv)) for t in titles]
        quotes_by_para = [(n + 1, t, [by_title[t][i] for i in sorted(supp[t])]) for n, (t, _) in enumerate(paras) if t in supp]
        body = "\n".join(f"段落{n + 1}（{t}）: {txt}" for n, (t, txt) in enumerate(paras))
        prompt = f"問題: {HOTPOT_HEAD}\n{body}\n質問: {q}\n{INSTR}"
        find = "探す: " + "と".join(f"段落{n}（{t}）" for n, t, _ in quotes_by_para) + "が必要。"
        qlines = [f"根拠: 「{s}」" for _, _, ss in quotes_by_para for s in ss]
        yn = ans.lower() in ("yes", "no")
        target = "\n" + find + "\n" + "\n".join(qlines) + f"\n答え: {ans}。"
        last_len = len(prompt) + len(target)
        if last_len <= ck.max_chars:
            chosen = (lv, prompt, target, quotes_by_para, yn)
            break
    if chosen is None:
        raise Reject("row_too_long")
    lv, prompt, target, qbp, yn = chosen
    quotes = [s for _, _, ss in qbp for s in ss]
    norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()      # noqa: E731
    if not yn and norm(ans).strip(" .,;:\"'") not in norm(" ".join(quotes)):
        raise Reject("answer_not_in_evidence")
    if ck.oov_count(" ".join(quotes) + ans + " ".join(t for _, t, _ in qbp)):
        raise Reject("oov_in_quote_or_answer")
    gk = "hotpot:" + min(supp)
    diff = {"easy": 2, "medium": 3, "hard": 4}.get(str(rec.get("level")), 3)
    return build_row(id=f"v49acq_hotpot_{rid}", family="hotpot_en_short", template="acq_hotpot_en_short", group="v49_hotpot_en_short",
                     category="evidence", prompt=prompt, target=target, ref=ans, answer_type="yes_no_en" if yn else "span_en",
                     difficulty=diff, spec=spec, split=split_of(gk), quotes=quotes, hotpot_type=str(rec.get("type") or ""),
                     hotpot_level=str(rec.get("level") or ""), trim_level=lv, trim_note=HOTPOT_LEVEL_NOTE[lv], group_key=gk,
                     language="en-passages/ja-frame", quality_flag="human annotated (HotpotQA); English passages")


def convert_hotpot(records, stats: Stats, ck: Checks, spec: Spec):
    return run_converter(records, stats, lambda i, r: _hotpot_one(i, r, stats, ck, spec), ck)


# ====================================================================================================== (e) ToolACE (adoption: hold)
TOOL_MARKER = "list of functions in JSON format that you can invoke:"
TOOLACE_HOLD = "hold (generator unverified: the card and the paper text do not name the generator model)"


def parse_tool_functions(system: str):
    i = system.find(TOOL_MARKER)
    if i < 0:
        return None
    j = system.find("[", i + len(TOOL_MARKER))
    if j < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(system, j)
    except ValueError:
        return None
    return obj if isinstance(obj, list) and all(isinstance(f, dict) and f.get("name") for f in obj) else None


def compact_function(f: dict, desc_chars: int = 90) -> str:
    params = f.get("parameters") if isinstance(f.get("parameters"), dict) else {}
    props = params.get("properties") if isinstance(params.get("properties"), dict) else {}
    req = params.get("required") if isinstance(params.get("required"), list) else []
    parts = [f"{k}{'' if k in req else '?'}: {(v or {}).get('type', 'any') if isinstance(v, dict) else 'any'}" for k, v in props.items()]
    desc = clean_ws(f.get("description") or "")
    if len(desc) > desc_chars:
        desc = desc[:desc_chars].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"- {f['name']}({', '.join(parts)}): {desc}"


def is_tool_call(value: str) -> bool:
    """ToolACE assistant turns that are calls look like [Name(arg=value, ...), Name2(...)] (names may contain spaces / parentheses)."""
    v = value.strip()
    return v.startswith("[") and v.endswith(")]") and "(" in v


def _toolace_one(idx, rec, st: Stats, ck: Checks, spec: Spec):
    if "__bad_json__" in rec:
        raise Reject("bad_json_item")
    conv = rec.get("conversations")
    if not isinstance(conv, list) or len(conv) < 2 or conv[0].get("from") != "user":
        raise Reject("no_user_turn")
    user = clean_text(conv[0].get("value") or "").strip()
    lang = detect_lang(user)
    st.extra["lang_" + lang] += 1                                 # every item with a user turn, before any other filter
    funcs = parse_tool_functions(str(rec.get("system") or ""))
    if not funcs:
        # about a third of the file lists its tools as YAML / XML / markdown text and calls them in other syntaxes; not converted
        raise Reject("tool_list_not_json")
    steps, final, i = [], None, 1
    while i < len(conv):
        m = conv[i]
        who, val = m.get("from"), clean_text(m.get("value") or "").strip()
        if who == "assistant" and val and is_tool_call(val):
            obs = None
            if i + 1 < len(conv) and conv[i + 1].get("from") == "tool":
                obs = clean_text(conv[i + 1].get("value") or "").strip()
                i += 1
            steps.append((val, obs))
            i += 1
            continue
        if who == "assistant":
            final = val or None
            i += 1
        break
    if i < len(conv):
        st.extra["multiturn_first_exchange_only"] += 1
    if not user:
        raise Reject("empty_user")
    if steps and final is None and any(o is not None for _, o in steps):
        raise Reject("no_final_answer_after_tool")
    if steps and final is None and len(steps) != 1:
        raise Reject("multiple_calls_without_result")
    if not steps and not final:
        raise Reject("empty_assistant")
    if any(o is not None and not o for _, o in steps):
        raise Reject("empty_tool_result")
    if steps and final and steps[-1][1] is None:
        raise Reject("final_after_call_without_result")
    sigs = "\n".join(compact_function(f) for f in funcs)
    prompt = f"問題: {user}\n道具: 関数呼び出し\n関数一覧:\n{sigs}\n思考:"
    if len(prompt) > ck.max_chars - 40:
        raise Reject("row_too_long")
    spans, tsteps, target = [], [], ""
    base = len(prompt)
    for k, (call, obs) in enumerate(steps):
        target += ("\n" if k == 0 else "") + f"呼び出し: {call}"
        if obs is not None:
            o = f"\n結果: {obs}\n"
            a = base + len(target)
            spans.append((a, a + len(o)))
            tsteps.append({"index": k, "call": call, "result": obs, "observation": o, "success": True})
            target += o
        else:
            target += "\n"
    if final:
        target += f"答え: {final}"
        family, ans = ("toolace_call", final) if steps else ("toolace_nocall", final)
    else:
        target = target.rstrip("\n")
        family, ans = "toolace_callonly", steps[0][0]
    if not target.startswith("\n"):
        target = "\n" + target
        spans = [(a + 1, b + 1) for a, b in spans]
    if len(prompt) + len(target) > ck.max_chars:
        raise Reject("row_too_long")
    gk = f"toolace:{idx}"
    return build_row(id=f"v49acq_toolace_{idx:06d}", family=family, template="acq_" + family, group="v49_toolace", category="tool_use",
                     prompt=prompt, target=target, ref=ans, answer_type="tool_call" if family == "toolace_callonly" else "free_text",
                     difficulty=3, spec=spec, split=split_of(gk), spans=spans, steps=tsteps, n_functions=len(funcs),
                     n_calls=len(steps), user_lang=lang, group_key=gk,
                     quality_flag="synthetic (ToolACE); generator unverified")


def convert_toolace(records, stats: Stats, ck: Checks, spec: Spec):
    return run_converter(records, stats, lambda i, r: _toolace_one(i, r, stats, ck, spec), ck)


# ====================================================================================================== (f) wizardlm8x22b (adoption: hold / low priority)
_ARITH = re.compile(r"(?<![0-9.A-Za-z])(?P<expr>\(?-?\d+(?:\.\d+)?\)?(?:\s*[-+*/]\s*\(?-?\d+(?:\.\d+)?\)?)+)\s*=\s*"
                    r"(?P<res>-?\d+(?:\.\d+)?)(?![\d.]|\s*[-+*/^]\s*[\d(])")
WIZARD_HOLD = "hold / low priority (the owner's materials plan lists it under 'avoid: obvious mathematical errors')"


def arith_normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).translate({0xD7: "*", 0xF7: "/", 0x2212: "-", 0x2013: "-"})
    return re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)


def safe_eval_fraction(expr: str):
    """Exact value of an arithmetic expression (+ - * / parentheses, unary minus, decimals) or None.  No eval()."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return None
    n_nodes = [0]

    def ev(n):
        n_nodes[0] += 1
        if n_nodes[0] > 40:
            raise ValueError("too big")
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return Fraction(str(n.value))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if b == 0:
                raise ZeroDivisionError
            return a / b
        raise ValueError("node")

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def check_arithmetic(text: str):
    """[(statement, ok)] for the simple 'a op b ... = c' statements of `text`.  Rounded results are accepted (|diff| <= half a unit
    of the last written digit); a statement preceded by an operator / letter / '=' is skipped (chain or equation in a variable)."""
    s = arith_normalize(text)
    out = []
    for m in _ARITH.finditer(s):
        prev = s[:m.start()].rstrip()[-1:]
        if prev and (prev in "+-*/^(=" or (prev.isascii() and prev.isalnum())):
            continue
        expr, res = m.group("expr"), m.group("res")
        if expr.count("(") != expr.count(")") or not re.search(r"\d\s*[-+*/]\s*\(?-?\d", expr):
            continue
        val = safe_eval_fraction(expr)
        if val is None:
            continue
        dec = len(res.split(".")[1]) if "." in res else 0
        ok = abs(val - Fraction(res)) <= Fraction(1, 2 * 10 ** dec)
        out.append((f"{expr.strip()} = {res}", ok))
        if len(out) >= 60:
            break
    return out


def _wizard_one(idx, rec, st: Stats, ck: Checks, spec: Spec):
    msgs = rec.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise Reject("no_messages")
    ui = next((i for i, m in enumerate(msgs) if isinstance(m, dict) and m.get("role") == "user"), None)
    ai = None if ui is None else next((i for i in range(ui + 1, len(msgs)) if isinstance(msgs[i], dict) and msgs[i].get("role") == "assistant"), None)
    if ui is None or ai is None:
        raise Reject("no_user_assistant_pair")
    if len(msgs) > ai + 1:
        st.extra["multiturn_first_pair_only"] += 1
    user, asst = clean_text(msgs[ui].get("content") or "").strip(), clean_text(msgs[ai].get("content") or "").strip()
    if not user or not asst:
        raise Reject("empty_message")
    stmts = check_arithmetic(asst)                               # statistics over ALL parseable rows, before the length filter
    prompt = f"問題: {user}\n{INSTR}"
    target = f"\n答え: {asst}"
    fits = len(prompt) + len(target) <= ck.max_chars
    if stmts:
        wrong = sum(1 for _, ok in stmts if not ok)
        st.extra["arith_rows_checked"] += 1
        st.extra["arith_stmts_checked"] += len(stmts)
        st.extra["arith_stmts_wrong"] += wrong
        st.extra["arith_rows_wrong"] += wrong > 0
        if fits:
            st.extra["fit_arith_rows_checked"] += 1
            st.extra["fit_arith_rows_wrong"] += wrong > 0
    if not fits:
        raise Reject("row_too_long")
    if stmts and any(not ok for _, ok in stmts):
        raise Reject("arith_statement_wrong")
    if latin_ratio(asst) > 0.30:
        raise Reject("latin_heavy")
    if looks_repetitive(asst):
        raise Reject("repetitive")
    nums = _numbers(unicodedata.normalize("NFKC", asst))
    family = "wizard_math" if stmts else "wizard_text"
    gk = f"wizard:{idx}"
    extra = {"arith_statements_checked": len(stmts), "has_python_block": FENCE + "python" in asst, "group_key": gk,
             "quality_flag": "wizardlm-2-8x22b generated, unverified"}
    if stmts and nums:
        extra["final_number_guess"] = nums[-1]
    return build_row(id=f"v49acq_wizard_{idx:07d}", family=family, template="acq_" + family, group="v49_wizard", category="free_text",
                     prompt=prompt, target=target, ref=asst, answer_type="free_text", difficulty=3, spec=spec, split=split_of(gk),
                     **extra)


def convert_wizard(records, stats: Stats, ck: Checks, spec: Spec):
    return run_converter(records, stats, lambda i, r: _wizard_one(i, r, stats, ck, spec), ck)


# ====================================================================================================== registry
DATASETS: dict = {
    "jaquad": Spec("jaquad", "SkelterLabsInc/JaQuAD", "CC-BY-SA-3.0",
                   "JaQuAD (So, Byun, Kang, Cho; Skelter Labs) - CC BY-SA 3.0 - https://huggingface.co/datasets/SkelterLabsInc/JaQuAD",
                   "So et al. (2022) JaQuAD: Japanese Question Answering Dataset for Machine Reading Comprehension, arXiv:2202.01764; "
                   "contexts from Japanese Wikipedia (CC BY-SA); train files only",
                   "candidate", "human annotated; group split by article title", find_jaquad, read_jaquad, convert_jaquad),
    "coding": Spec("coding", "llm-jp/Synthetic-JP-EN-Coding-Dataset", "Apache-2.0",
                   "llm-jp/Synthetic-JP-EN-Coding-Dataset (subset of Aratako/Synthetic-JP-EN-Coding-Dataset-801k) - Apache-2.0",
                   "LLM-jp (Kiyomaru, Kodama); instructions by Nemotron-4-340B-Instruct, Phi-3-medium, Mixtral-8x22B-Instruct, calm3-22b-chat; "
                   "no post-filtering by the authors",
                   "candidate",
                   "python rows: stored output replaced by sandbox stdout (per-row 'verification': stored_output_equals_sandbox = "
                   "independently corroborated by the answer's own expected output, sandbox_only_no_stored_output = executed but not "
                   "corroborated: the code may still not solve the task); code_text rows unverified (not executed)", find_coding,
                   read_messages_parquet, convert_coding),
    "tengentoppa": Spec("tengentoppa", "DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k", "Apache-2.0",
                        "DeL-TaiseiOzaki/Tengentoppa-sft-qwen2.5-32b-reasoning-100k - Apache-2.0",
                        "Generated with Qwen2.5-32B-Instruct (multi-persona instructions, refined answers); no human check besides the card's note",
                        "candidate", "qwen2.5-32b generated, unverified; refined_answer only (initial_answer dropped)",
                        find_tengentoppa, read_tengentoppa, convert_tengentoppa),
    "hotpot": Spec("hotpot", "hotpotqa/hotpot_qa", "CC-BY-SA-4.0",
                   "HotpotQA distractor setting, train split (Yang et al., 2018) - CC BY-SA 4.0 - https://hotpotqa.github.io/",
                   "Yang, Qi, Zhang, Bengio, Cohen, Salakhutdinov, Manning (2018) HotpotQA: A Dataset for Diverse, Explainable Multi-hop "
                   "Question Answering, EMNLP; English Wikipedia paragraphs; validation split never used",
                   "candidate", "short English rows only: English passages inside a Japanese instruction frame",
                   find_hotpot, read_hotpot, convert_hotpot),
    "toolace": Spec("toolace", "Team-ACE/ToolACE", "Apache-2.0",
                    "Team-ACE/ToolACE - Apache-2.0 (dataset card)",
                    "Liu et al. (2024) ToolACE: Winning the Points of LLM Function Calling, arXiv:2409.00920; generator model not stated",
                    "hold", TOOLACE_HOLD, find_toolace, read_toolace, convert_toolace),
    "wizard": Spec("wizard", "llm-jp/wizardlm8x22b-logical-math-coding-sft-ja", "Apache-2.0",
                   "llm-jp/wizardlm8x22b-logical-math-coding-sft-ja - Apache-2.0 (from kanhatakeyama/wizardlm8x22b-logical-math-coding-sft-ja "
                   "and ..._additional-ja)",
                   "LLM-jp (Kiyomaru, Kodama); texts generated with WizardLM-2-8x22B",
                   "hold", WIZARD_HOLD, find_wizard, read_messages_parquet, convert_wizard),
}


def summarize(stats: Stats, ck: Checks | None = None) -> dict:
    """JSON-able summary of one converter run (counts, rejects by reason, per-family splits, derived rates, backends)."""
    spec = DATASETS.get(stats.key)
    rej_total = sum(stats.rejects.values())
    d = {
        "dataset": stats.key, "hf": spec.hf if spec else "", "license": spec.license if spec else "",
        "adoption": spec.adoption if spec else "", "adoption_note": spec.adoption_note if spec else "",
        "n_in": stats.n_in, "n_out": stats.n_out, "keep_rate": round(stats.n_out / stats.n_in, 6) if stats.n_in else None,
        "reject_total": rej_total, "balanced": stats.n_in == stats.n_out + rej_total,
        "rejects": dict(stats.rejects.most_common()), "families": dict(stats.families), "splits": dict(stats.splits),
        "family_splits": {k: dict(v) for k, v in stats.family_splits.items()}, "extra": dict(stats.extra),
    }
    e = stats.extra
    if stats.key == "wizard":
        d["arith_error_rate"] = {
            "statements_checked": e["arith_stmts_checked"], "statements_wrong": e["arith_stmts_wrong"],
            "statement_error_rate": round(e["arith_stmts_wrong"] / e["arith_stmts_checked"], 6) if e["arith_stmts_checked"] else None,
            "rows_with_checked_statements": e["arith_rows_checked"], "rows_with_a_wrong_statement": e["arith_rows_wrong"],
            "row_error_rate": round(e["arith_rows_wrong"] / e["arith_rows_checked"], 6) if e["arith_rows_checked"] else None,
            "rows_fitting_cap_checked": e["fit_arith_rows_checked"], "rows_fitting_cap_with_wrong": e["fit_arith_rows_wrong"],
            "note": "only simple 'a op b = c' statements are recomputed (exact fractions, rounded results accepted); lower bound of the true error rate"}
    if stats.key == "toolace":
        d["language"] = {"parsed_en": e["lang_en"], "parsed_zh": e["lang_zh"], "parsed_ja": e["lang_ja"], "parsed_other": e["lang_other"],
                         "kept_en": e["accepted_user_lang_en"], "kept_zh": e["accepted_user_lang_zh"]}
    if stats.key == "hotpot":
        d["fit"] = {"rows_emitted": stats.n_out, "fit_untrimmed_all_paragraphs": e["fit_untrimmed_all_paragraphs"],
                    "by_trim_level": {k: e[f"trim_level_{k}"] for k in range(5)}, "level_note": HOTPOT_LEVEL_NOTE}
    if stats.key == "coding":
        d["verification"] = {k[len("verification_"):]: v for k, v in e.items() if k.startswith("verification_")}
        d["code_text_languages"] = {k[len("code_text_lang_"):]: v for k, v in e.items() if k.startswith("code_text_lang_")}
    if ck is not None:
        d["backends"] = dict(ck.backends)
        d["max_chars"] = ck.max_chars
        d["vocab_size"] = len(ck.vocab)
        d["vocab_path"] = ck.vocab_path
    return d


# ====================================================================================================== output
class SplitWriter:
    """train/dev/heldout JSONL files for one dataset.  Files are written under a .part name and only renamed into place when the
    target does not exist; an existing target with a different sha256 is never touched (new content kept as .new-<sha8>)."""

    SPLITS = ("train", "dev", "heldout")

    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.tmp = {sp: self.dir / f"{sp}.jsonl.part-{os.getpid()}" for sp in self.SPLITS}
        self.fh = {sp: open(p, "w", encoding="utf-8", newline="\n") for sp, p in self.tmp.items()}
        self.counts: Counter = Counter()

    def write(self, row: dict):
        line = json.dumps(row, ensure_ascii=False)
        line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029").replace("\x85", "\\u0085")
        self.fh[row["split"]].write(line + "\n")
        self.counts[row["split"]] += 1

    @staticmethod
    def _free(path: Path) -> Path:
        """`path`, or path.1, path.2, ... : the first name that does not exist (existing files are never replaced)."""
        cand, i = path, 0
        while cand.exists():
            i += 1
            cand = path.with_name(path.name + f".{i}")
        return cand

    def close(self) -> dict:
        info = {}
        for sp in self.SPLITS:
            self.fh[sp].close()
        for sp in self.SPLITS:
            tmp = self.tmp[sp]
            data_sha = hashlib.sha256()
            with open(tmp, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    data_sha.update(chunk)
            sha = data_sha.hexdigest()
            target = self.dir / f"{sp}.jsonl"
            entry = {"rows": self.counts[sp], "bytes": tmp.stat().st_size, "sha256": sha}
            if not target.exists():
                tmp.rename(target)
                entry["status"] = "written"
            else:
                old = hashlib.sha256(target.read_bytes()).hexdigest()
                if old == sha:
                    entry["status"] = "identical_existing"
                    tmp.unlink()                                   # only this run's own temp file; the existing target is untouched
                else:
                    kept = self._free(self.dir / f"{sp}.jsonl.new-{sha[:8]}")
                    tmp.rename(kept)
                    entry["status"] = "REFUSED_existing_file_differs"
                    entry["kept_as"] = kept.name
            info[f"{sp}.jsonl"] = entry
        return info


def _write_new(path: Path, text: str) -> str:
    """Create `path` with `text`; never overwrite a file with different content."""
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == data:
            return "identical_existing"
        alt = SplitWriter._free(path.with_name(path.name + ".new-" + hashlib.sha256(data).hexdigest()[:8]))
        alt.write_bytes(data)
        return "REFUSED_existing_file_differs:" + alt.name
    path.write_bytes(data)
    return "written"


def format_dump(rows) -> str:
    out = []
    for r in rows:
        out.append("=" * 100)
        out.append(f"{r['id']}  family={r['family']}  split={r['split']}  len={len(r['full_text'])}  adoption={r['adoption']}")
        out.append("-" * 40 + " PROMPT")
        out.append(r["prompt"])
        out.append("-" * 40 + " TARGET   (loss_mask_spans=" + json.dumps(r["loss_mask_spans"]) + ")")
        out.append(r["target"])
        out.append("")
    return "\n".join(out)


def subdir_for(raw_dir, key: str) -> Path:
    spec = DATASETS[key]
    root = Path(raw_dir)
    for cand in (root / key, root / spec.hf.replace("/", "__"), root / spec.hf.split("/")[1], root / spec.hf):
        if cand.is_dir():
            return cand
    return root


def load_provenance(raw_dir):
    """provenance.json written by acquire.py next to the dataset's files (None if absent / unreadable)."""
    p = Path(raw_dir) / "provenance.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return d if isinstance(d, dict) and isinstance(d.get("files"), list) else None
        except ValueError:
            return None
    return None


def apply_provenance(spec: Spec, prov: dict) -> Spec:
    """Row source / license from provenance['row_defaults'] and the attribution lines (acquire.py's rules for converters)."""
    rd = prov.get("row_defaults") or {}
    lines = ((prov.get("attribution") or {}).get("lines")) or []
    return dc_replace(spec, source=str(rd.get("source") or spec.source), license=str(rd.get("license") or spec.license),
                      attribution=" | ".join(str(x) for x in lines) or spec.attribution)


def train_only(files, raw_dir, prov: dict):
    """(kept, dropped): only files that provenance lists with role == train and may_train (acquire.train_files' rule)."""
    ok = {f["dest_rel"] for f in prov["files"] if f.get("role") == "train" and f.get("may_train")}
    kept, dropped = [], []
    for fp in files:
        rel = "/".join(Path(fp).relative_to(raw_dir).parts)
        (kept if any(d == rel or d.endswith("/" + rel) for d in ok) else dropped).append(fp)
    return kept, dropped


def run_dataset(key: str, raw_dir, out_dir, ck: Checks, limit=None, sample_dump: int = 100, seed: int = 20261109) -> dict:
    spec = DATASETS[key]
    files = spec.find(raw_dir)
    prov = load_provenance(raw_dir)
    not_train = []
    if prov:
        spec = apply_provenance(spec, prov)
        files, not_train = train_only(files, raw_dir, prov)
    if not files:
        raise FileNotFoundError(f"{key}: no train-eligible raw files under {raw_dir}")
    stats = Stats(key)
    rows = spec.convert(spec.reader(files, limit), stats, ck, spec)
    writer = SplitWriter(Path(out_dir) / key)
    heap, tie = [], 0
    for row in rows:
        writer.write(row)
        if sample_dump:
            tie += 1
            heapq.heappush(heap, (-h("dump", seed, row["id"]), tie, row))
            if len(heap) > sample_dump:
                heapq.heappop(heap)
    info = writer.close()
    summary = summarize(stats, ck)
    summary["raw_files"] = [{"name": Path(p).relative_to(Path(raw_dir)).as_posix(), "bytes": p.stat().st_size} for p in files]
    summary["provenance"] = {"used": bool(prov), "commit_sha": (prov or {}).get("commit_sha"),
                             "skipped_not_train_files": [Path(p).relative_to(Path(raw_dir)).as_posix() for p in not_train]}
    summary["row_source"], summary["row_license"] = spec.source, spec.license
    summary["limit"] = limit
    manifest = dict(summary, files={n: {k: v for k, v in e.items() if k in ("rows", "bytes", "sha256")} for n, e in info.items()})
    d = Path(out_dir) / key
    summary["files"] = info                                       # per-run statuses (written / identical_existing / REFUSED) stay out of the manifest
    statuses = {"manifest.json": _write_new(d / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))}
    if sample_dump:
        dump_rows = [r for _, _, r in sorted(heap, key=lambda t: (-t[0], t[1]))]
        statuses["sample_dump.txt"] = _write_new(d / "sample_dump.txt", format_dump(dump_rows))
    ex = [{"reason": k, **e} for k, v in stats.reject_examples.items() for e in v]
    statuses["rejects_sample.jsonl"] = _write_new(d / "rejects_sample.jsonl", "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in ex))
    summary["write_status"] = statuses
    refused = any(v["status"].startswith("REFUSED") for v in info.values()) or any(s.startswith("REFUSED") for s in statuses.values())
    summary["refused_overwrite"] = refused
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="GeoCPU v49 acquisition converters")
    ap.add_argument("--dataset", required=True, help="|".join(DATASETS) + "|all")
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="stop after N input records per dataset")
    ap.add_argument("--vocab", default=None, help="base_metadata.json (default: search, then checks.load_vocab on --drive-root)")
    ap.add_argument("--drive-root", default=None, help="Drive folder that holds the model bundle (for checks.load_vocab)")
    ap.add_argument("--max-chars", type=int, default=DEFAULT_CAP)
    ap.add_argument("--sample-dump", type=int, default=100, help="seeded readable sample of N emitted rows per dataset (0 = off)")
    ap.add_argument("--no-sandbox", action="store_true", help="do not execute python (coding_pytool rows are then rejected)")
    ap.add_argument("--sandbox-timeout", type=float, default=5.0)
    ap.add_argument("--no-external-checks", action="store_true", help="ignore a sibling checks.py")
    ap.add_argument("--python-require-stored-output", action="store_true",
                    help="coding: keep python rows only when the answer's own expected output equals the sandbox stdout")
    ap.add_argument("--no-tengentoppa-steps", action="store_true", help="tengentoppa: never include the 'reasoning' field")
    a = ap.parse_args(argv)
    keys = list(DATASETS) if a.dataset == "all" else [a.dataset]
    for k in keys:
        if k not in DATASETS:
            print("unknown dataset:", k, file=sys.stderr)
            return 2
    ck = Checks(vocab_path=a.vocab, drive_root=a.drive_root, max_chars=a.max_chars, run_code=not a.no_sandbox, sandbox_timeout=a.sandbox_timeout,
                use_external=not a.no_external_checks, python_require_stored_output=a.python_require_stored_output,
                tengentoppa_steps=not a.no_tengentoppa_steps)
    code = 0
    allsum = {}
    for k in keys:
        raw = subdir_for(a.raw_dir, k) if a.dataset == "all" else Path(a.raw_dir)
        try:
            s = run_dataset(k, raw, a.out_dir, ck, limit=a.limit, sample_dump=a.sample_dump)
        except FileNotFoundError as e:
            print(f"[{k}] {e}", file=sys.stderr)
            code = max(code, 2)
            continue
        allsum[k] = {x: s[x] for x in ("n_in", "n_out", "keep_rate", "reject_total", "balanced", "families", "splits", "refused_overwrite")}
        print(json.dumps({k: allsum[k]}, ensure_ascii=False))
        if s["refused_overwrite"]:
            code = 3
    if len(keys) > 1:
        _write_new(Path(a.out_dir) / "convert_summary.json", json.dumps(allsum, ensure_ascii=False, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
