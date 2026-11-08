"""GeoCPU client for the RustXNG /geosearch endpoint (rule-based, no model, no training).

Drop-in for LocalSearchEngineV42: the tool loop calls `engine.search(query)` and injects the returned string.
Rules, all deterministic:
  * every result line names its source:  [RESULT#n (src: DOMAIN, date: DATE): SNIPPET]  (+ " <URL>" when show_url)
  * at most `per_domain` results per domain (diversity), `top_k` results in total, `max_total` characters in total
  * results from benchmark-answer sites (BENCH_BLOCK) are dropped, so knowledge benchmarks stay clean
  * a slow or failing server never raises: the optional `fallback` engine answers, else a 'not found' line
  * record/replay: every live answer can be appended to a JSONL file; replay answers from it with no network
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

# sites that host benchmark questions/answers (JMMLU, JamC-QA, JGLUE, ...). Matched against the domain and URL.
BENCH_BLOCK = (
    "huggingface.co", "hf.co", "github.com/nlp-waseda", "github.com/sbintuitions", "github.com/sbintuitions",
    "kaggle.com", "paperswithcode.com", "jmmlu", "jamc-qa", "jamcqa", "llm-jp-eval",
)
NOT_FOUND = "[RESULT (src: none, date: 時期未詳): 該当する情報は見つかりませんでした]"


def _clean(text: str) -> str:
    """One line, no markdown/code fences (the tool loop must never see ``` or newlines inside an observation)."""
    text = text.replace("```", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)      # markdown links -> their text
    text = re.sub(r"[*_`#>]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def blocked(domain: str, url: str, block=BENCH_BLOCK) -> bool:
    haystack = (domain + " " + url).lower()
    return any(b in haystack for b in block)


def format_results(results: list, *, top_k: int, per_domain: int, max_total: int, max_chars: int,
                   show_url: bool = False, block=BENCH_BLOCK) -> str:
    """Pure function: server JSON results -> the string injected into the model's context."""
    picked, seen, per = [], set(), {}
    for r in results:
        domain = str(r.get("source") or "unknown")
        url = str(r.get("url") or "")
        if blocked(domain, url, block):
            continue
        snippet = _clean(str(r.get("snippet") or r.get("title") or ""))
        if not snippet or snippet in seen:
            continue
        if per.get(domain, 0) >= per_domain:
            continue
        seen.add(snippet)
        per[domain] = per.get(domain, 0) + 1
        picked.append((domain, str(r.get("date") or "時期未詳"), snippet, url))
        if len(picked) >= top_k:
            break
    if not picked:
        return NOT_FOUND
    # budget: header is not part of max_chars on the server, so count the whole line here
    lines, used = [], 0
    for i, (domain, date, snippet, url) in enumerate(picked, 1):
        tag = f"[RESULT (src: {domain}, date: {date}): " if len(picked) == 1 else f"[RESULT#{i} (src: {domain}, date: {date}): "
        tail = (f" <{url}>" if show_url and url else "") + "]"
        room = max_total - used - len(tag) - len(tail)
        remaining_slots = len(picked) - i
        room = min(room, max_chars) if remaining_slots == 0 else min(max_chars, max(20, room // (remaining_slots + 1)))
        if room < 15:
            break
        body = snippet if len(snippet) <= room else snippet[: room - 1] + "…"
        line = tag + body + tail
        lines.append(line)
        used += len(line) + 1
    return "\n".join(lines) if lines else NOT_FOUND


class RemoteGeoSearch:
    def __init__(self, base_url: str, *, token: Optional[str] = None, timeout: float = 4.0, top_k: int = 3,
                 per_domain: int = 2, max_chars: int = 200, max_total: int = 520, show_url: bool = False,
                 fallback=None, record_path: Optional[str] = None, replay_path: Optional[str] = None,
                 block=BENCH_BLOCK, extra_params: Optional[dict] = None):
        self.base_url = base_url
        self.token, self.timeout = token, timeout
        self.top_k, self.per_domain, self.max_chars, self.max_total = top_k, per_domain, max_chars, max_total
        self.show_url, self.fallback, self.block = show_url, fallback, block
        self.extra = dict(extra_params or {})
        self.record_path = Path(record_path) if record_path else None
        self.replay = self._load_replay(replay_path) if replay_path else None
        self.cache: dict = {}
        self.stats = {"live": 0, "cache": 0, "replay": 0, "fallback": 0, "empty": 0, "error": 0}

    @staticmethod
    def _load_replay(path):
        table = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                table[row["query"]] = row["results"]
        return table

    def _fetch(self, query: str) -> list:
        params = {"q": query, "format": "json", "top_k": 5, "max_chars": self.max_chars}
        params.update(self.extra)
        url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "GeoCPU-remote/1.0"}
        if self.token:
            headers["X-Geo-Token"] = self.token
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return list(body.get("results") or [])

    def search(self, query: str, top_k: Optional[int] = None, max_chars: Optional[int] = None) -> str:
        query = _clean(query)[:300]
        if not query:
            return NOT_FOUND
        k = top_k or self.top_k
        key = (query, k)
        if key in self.cache:
            self.stats["cache"] += 1
            return self.cache[key]
        results = None
        if self.replay is not None:
            results = self.replay.get(query)
            if results is not None:
                self.stats["replay"] += 1
        if results is None and self.replay is None:
            try:
                results = self._fetch(query)
                self.stats["live"] += 1
                if self.record_path:
                    self.record_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.record_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"query": query, "ts": time.time(), "results": results},
                                           ensure_ascii=False) + "\n")
            except (urllib.error.URLError, TimeoutError, ValueError, OSError):
                self.stats["error"] += 1
                results = None
        if results is None:
            if self.fallback is not None:
                self.stats["fallback"] += 1
                text = self.fallback.search(query)
            else:
                text = NOT_FOUND
        else:
            text = format_results(results, top_k=k, per_domain=self.per_domain, max_total=self.max_total,
                                  max_chars=max_chars or self.max_chars, show_url=self.show_url, block=self.block)
            if text == NOT_FOUND:
                self.stats["empty"] += 1
        self.cache[key] = text
        return text
