"""CPU-only tests for remote_geosearch.py. A real local HTTP server plays the part of RustXNG /geosearch."""
import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import remote_geosearch as R  # noqa: E402


def result(source, snippet, date="2026-10-01", url=None, title="t"):
    return {"source": source, "snippet": snippet, "date": date, "url": url or f"https://{source}/x", "title": title}


class Server:
    """Fake /geosearch: counts hits, optional token check and delay."""
    def __init__(self, results, token=None, delay=0.0):
        outer = self
        self.hits, self.queries, self.results, self.token, self.delay = 0, [], results, token, delay

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                outer.hits += 1
                if outer.delay:
                    time.sleep(outer.delay)
                if outer.token and self.headers.get("X-Geo-Token") != outer.token:
                    self.send_response(403)
                    self.end_headers()
                    return
                outer.queries.append(self.path)
                body = json.dumps({"query": "q", "count": len(outer.results), "results": outer.results}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:      # the client already timed out
                    pass

        self.httpd = HTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/geosearch"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class FormatRules(unittest.TestCase):
    def fmt(self, results, **kw):
        base = dict(top_k=3, per_domain=2, max_total=520, max_chars=200)
        base.update(kw)
        return R.format_results(results, **base)

    def test_single_result_names_its_source(self):
        text = self.fmt([result("ja.wikipedia.org", "ヘスの法則は総熱量保存の法則")])
        self.assertEqual(text, "[RESULT (src: ja.wikipedia.org, date: 2026-10-01): ヘスの法則は総熱量保存の法則]")

    def test_every_line_has_a_source_and_numbering(self):
        text = self.fmt([result("a.jp", "一つ目"), result("b.jp", "二つ目"), result("c.jp", "三つ目")])
        lines = text.splitlines()
        self.assertEqual(len(lines), 3)
        for i, line in enumerate(lines, 1):
            self.assertTrue(line.startswith(f"[RESULT#{i} (src: "), line)
            self.assertIn("date: ", line)

    def test_domain_cap_gives_diversity(self):
        results = [result("same.jp", f"文{i}") for i in range(5)] + [result("other.jp", "別のサイト")]
        text = self.fmt(results, top_k=5, per_domain=2)
        self.assertEqual(text.count("src: same.jp"), 2)
        self.assertEqual(text.count("src: other.jp"), 1)

    def test_total_budget_is_respected_including_headers(self):
        results = [result(f"d{i}.jp", "あ" * 300) for i in range(5)]
        for budget in (200, 300, 520):
            text = self.fmt(results, top_k=5, max_total=budget, max_chars=300)
            self.assertLessEqual(len(text), budget, (budget, len(text)))
            self.assertIn("src:", text)

    def test_benchmark_answer_sites_are_dropped(self):
        results = [result("huggingface.co", "答えはB"), result("example.com", "正しい資料", url="https://example.com/jmmlu-answers"),
                   result("ok.jp", "使える資料")]
        text = self.fmt(results)
        self.assertNotIn("huggingface.co", text)
        self.assertNotIn("example.com", text)
        self.assertIn("src: ok.jp", text)

    def test_nothing_usable_is_a_not_found_line(self):
        self.assertEqual(self.fmt([]), R.NOT_FOUND)
        self.assertEqual(self.fmt([result("huggingface.co", "x")]), R.NOT_FOUND)

    def test_markdown_newlines_and_code_fences_never_reach_the_model(self):
        r = result("w.org", "- **第709条** [故意](/wiki/x)又は過失\n```python\nprint(1)\n```\n続き")
        text = self.fmt([r])
        self.assertNotIn("\n", text)
        self.assertNotIn("```", text)
        self.assertNotIn("](", text)
        self.assertIn("故意", text)

    def test_duplicate_snippets_are_dropped(self):
        text = self.fmt([result("a.jp", "同じ文"), result("b.jp", "同じ文"), result("c.jp", "別の文")], top_k=3)
        self.assertEqual(len(text.splitlines()), 2)

    def test_url_is_shown_only_on_request(self):
        r = [result("a.jp", "文", url="https://a.jp/p")]
        self.assertNotIn("https://a.jp/p", self.fmt(r))
        self.assertIn("<https://a.jp/p>", self.fmt(r, show_url=True))


class Client(unittest.TestCase):
    def setUp(self):
        self.servers = []

    def tearDown(self):
        for s in self.servers:
            s.stop()

    def serve(self, *a, **kw):
        s = Server(*a, **kw)
        self.servers.append(s)
        return s

    def test_live_answer_cache_and_stats(self):
        s = self.serve([result("ja.wikipedia.org", "答えを含む一節")])
        c = R.RemoteGeoSearch(s.url, timeout=3)
        first = c.search("ヘスの法則")
        self.assertIn("src: ja.wikipedia.org", first)
        self.assertEqual(c.search("ヘスの法則"), first)
        self.assertEqual((s.hits, c.stats["live"], c.stats["cache"]), (1, 1, 1))
        self.assertIn("format=json", s.queries[0])

    def test_token_is_sent_and_a_wrong_token_falls_back_without_raising(self):
        s = self.serve([result("a.jp", "文")], token="secret")
        ok = R.RemoteGeoSearch(s.url, token="secret", timeout=3).search("q")
        self.assertIn("src: a.jp", ok)
        bad = R.RemoteGeoSearch(s.url, token="wrong", timeout=3)
        self.assertEqual(bad.search("q"), R.NOT_FOUND)
        self.assertEqual(bad.stats["error"], 1)

    def test_slow_server_uses_the_fallback_engine(self):
        class Fallback:
            def search(self, query, **kw):
                return "[RESULT (src: local, date: 常駐): ローカルの答え]"
        s = self.serve([result("a.jp", "文")], delay=1.5)
        c = R.RemoteGeoSearch(s.url, timeout=0.3, fallback=Fallback())
        started = time.time()
        self.assertIn("ローカルの答え", c.search("q"))
        self.assertLess(time.time() - started, 1.2)
        self.assertEqual(c.stats["fallback"], 1)

    def test_unreachable_server_never_raises(self):
        c = R.RemoteGeoSearch("http://127.0.0.1:9/geosearch", timeout=0.5)
        self.assertEqual(c.search("q"), R.NOT_FOUND)

    def test_record_then_replay_without_network(self):
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "rec.jsonl"
            s = self.serve([result("ja.wikipedia.org", "記録される一節")])
            live = R.RemoteGeoSearch(s.url, timeout=3, record_path=str(log))
            expected = live.search("光電効果")
            s.stop()                                   # no server any more
            replay = R.RemoteGeoSearch("http://127.0.0.1:9/geosearch", timeout=0.2, replay_path=str(log))
            self.assertEqual(replay.search("光電効果"), expected)
            self.assertEqual(replay.stats["replay"], 1)
            self.assertEqual(replay.search("記録にない問い"), R.NOT_FOUND)      # replay never goes to the network
            self.assertEqual(replay.stats["error"], 0)

    def test_empty_query_and_long_query(self):
        s = self.serve([result("a.jp", "文")])
        c = R.RemoteGeoSearch(s.url, timeout=3)
        self.assertEqual(c.search("   "), R.NOT_FOUND)
        c.search("あ" * 1000)
        import urllib.parse
        sent = urllib.parse.parse_qs(urllib.parse.urlsplit(s.queries[0]).query)["q"][0]
        self.assertEqual(len(sent), 300)           # clamped to 300 characters (decoded)


if __name__ == "__main__":
    unittest.main()
