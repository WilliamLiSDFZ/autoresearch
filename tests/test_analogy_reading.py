"""Offline corpus/reading regressions, including real PDF parsing in its worker."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoresearch_analogy.corpus import PaperCorpus, load_corpus, read_corpus_dir
from autoresearch_analogy.fulltext import (
    FullTextConfig, FullTextError, FullTextStore, PaperReadingSession, digest, resolve_urls,
)
from autoresearch_analogy.fulltext_worker import check_url, split_pages, title_matches


QUOTE = "Minority group performance depends on the sampling distribution."
RECORD = {
    "id": "venue/test", "title": "Sampling Distribution and Minority Group Performance",
    "venue": "venue", "category": "sampling", "categories": ["sampling"],
    "abstract": QUOTE, "tldr": "Correct sampling to control group risk.",
    "source": "https://arxiv.org/abs/2407.13957v1", "pdf_url": "",
}


class TinyCorpus:
    def __init__(self, records=None):
        self.by_id = {r["id"]: r for r in records or [RECORD]}

    def __contains__(self, pid):
        return pid in self.by_id


def write_corpus(directory, records=None):
    records = records or [RECORD]
    raw = "".join(json.dumps(r) + "\n" for r in records).encode()
    (directory / "records.jsonl").write_bytes(raw)
    venues = {}
    for r in records:
        venues[r["venue"]] = venues.get(r["venue"], 0) + 1
    manifest = {"level": "paper", "schema_version": 2, "count": len(records),
                "venues": venues, "records_sha1": hashlib.sha1(raw).hexdigest()[:12]}
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def fixture_document(text=QUOTE):
    chunks = split_pages([{"text": "# Method\n\n" + text}, {"text": "# Appendix\n\n" + QUOTE}])
    return {"paper_id": RECORD["id"], "title": RECORD["title"], "chunks": chunks,
            "pdf_sha256": "fixture", "text_sha256": digest(chunks), "parser": {},
            "page_count": 2, "source_url": resolve_urls(RECORD)[0], "warnings": []}


def cache_directory(store, record=RECORD):
    return store.root / digest({"paper_id": record["id"], "title": record["title"],
                               "urls": resolve_urls(record), "parser": store.versions})


def seed_real_pdf(store):
    """Seed only downloaded bytes/receipt so FullTextStore still launches the real parser."""
    import pymupdf
    directory = cache_directory(store)
    directory.mkdir(parents=True, exist_ok=True)
    with pymupdf.open() as pdf:
        for i in range(3):
            page = pdf.new_page(width=650, height=850)
            title = RECORD["title"] if i == 0 else "Appendix: Sampling Procedure"
            page.insert_text((40, 42), title, fontsize=12)
            body = "\n".join([QUOTE] * 32)
            page.insert_text((40, 78), body, fontsize=10)
        pdf.save(directory / "paper.pdf")
    receipt = {"urls": resolve_urls(RECORD), "source_url": resolve_urls(RECORD)[0],
               "fetched_at": "offline-fixture",
               "pdf_sha256": hashlib.sha256((directory / "paper.pdf").read_bytes()).hexdigest()}
    (directory / "download.json").write_text(json.dumps(receipt))
    return directory


class CorpusTests(unittest.TestCase):
    def test_manifest_content_and_schema_are_verified(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            manifest = write_corpus(directory)
            records, verified = read_corpus_dir(directory)
            self.assertEqual(records[0]["id"], RECORD["id"])
            self.assertEqual(len(verified["records_sha256"]), 64)
            (directory / "records.jsonl").write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "records_sha1"):
                read_corpus_dir(directory)
            write_corpus(directory, [RECORD, RECORD])
            with self.assertRaisesRegex(ValueError, "unique"):
                read_corpus_dir(directory)
            write_corpus(directory)
            manifest["schema_version"] = 1
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "schema-2"):
                read_corpus_dir(directory)

    def test_stemming_and_stable_ties(self):
        records = [{**RECORD, "id": "venue/first"}, {**RECORD, "id": "venue/second"},
                   {**RECORD, "id": "venue/other", "title": "Quantum Device",
                    "tldr": "photons", "abstract": "Quantum apparatus"}]
        corpus = PaperCorpus(records, {})
        self.assertEqual(corpus.tokenize("The samplings"), ["sampl"])
        with patch.object(corpus.bm25, "get_scores", return_value=__import__("numpy").array([1., 1., 0.])):
            self.assertEqual([r["id"] for r in corpus.search("sampling")], ["venue/first", "venue/second"])
        self.assertEqual(corpus.get(["unknown", "venue/first"])[0]["id"], "venue/first")

    def test_reload_rejects_content_change_even_with_updated_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            write_corpus(directory)
            corpus = load_corpus(directory)
            self.assertIs(load_corpus(directory), corpus)
            write_corpus(directory, [{**RECORD, "abstract": "A changed sampling abstract"}])
            with self.assertRaisesRegex(ValueError, "changed after"):
                load_corpus(directory)


class ReadingTests(unittest.TestCase):
    def session(self, **options):
        return PaperReadingSession(TinyCorpus(), FullTextConfig(enabled=True, **options))

    def test_source_resolution_and_url_boundaries(self):
        self.assertEqual(resolve_urls(RECORD), ["https://arxiv.org/pdf/2407.13957v1"])
        self.assertEqual(resolve_urls({"source": "https://openreview.net/forum?id=abc"}),
                         ["https://openreview.net/pdf?id=abc"])
        for url in ["file:///etc/passwd", "http://arxiv.org/pdf/1", "https://127.0.0.1/x",
                    "https://raw.githubusercontent.com/random/repo/paper.pdf"]:
            with self.assertRaises(FullTextError):
                check_url(url)
        check_url("https://raw.githubusercontent.com/mlresearch/v1/main/example.pdf")

    def test_chunks_preserve_appendix_and_complete_body(self):
        chunks = split_pages([{"text": "# Method\n\n" + "A" * 6001},
                              {"text": "# Appendix\n\n" + QUOTE}])
        self.assertTrue(all(c["chars"] <= 2000 for c in chunks))
        self.assertEqual(sum(c["text"].count("A") for c in chunks if c["page"] == 1), 6001)
        self.assertEqual(chunks[-1]["section"], "Appendix")
        self.assertEqual(chunks[-1]["page"], 2)
        self.assertFalse(title_matches(RECORD["title"], "Unrelated chemistry research"))

    def test_only_seen_opened_read_chunks_become_evidence(self):
        session = self.session()
        pid = RECORD["id"]
        self.assertEqual(session.call("open_paper", {"paper_id": pid}, set())["status"], "rejected")
        self.assertEqual(session.call("read_paper", {"paper_id": pid, "chunk_ids": ["p001-c001"]}, {pid})["status"], "not_open")
        with patch.object(session.store, "get", return_value=(fixture_document(), True)):
            opened = session.call("open_paper", {"paper_id": pid}, {pid})
        self.assertNotIn("text", opened["outline"][0])
        self.assertFalse(session.delivered)
        result = session.call("read_paper", {"paper_id": pid, "chunk_ids": ["p001-c001"]}, {pid})
        self.assertEqual(result["status"], "ok")
        self.assertIn(QUOTE, session.delivered[(pid, "p001-c001")]["text"])
        self.assertNotIn((pid, "p002-c001"), session.delivered)
        self.assertEqual(session.snapshot()["events"][-1]["result"], result)
        bad = session.call("read_paper", {"paper_id": pid, "chunk_ids": ["p999-c001"]}, {pid})
        self.assertEqual(bad["status"], "invalid_arguments")

    def test_limits_charge_repeated_reads_and_failures(self):
        session = self.session(read_chars=2100, total_chars=4000, max_read_calls=2, max_papers=1)
        pid = RECORD["id"]
        session.documents[pid] = fixture_document("a" * 6000)
        args = {"paper_id": pid, "chunk_ids": [c["chunk_id"] for c in session.documents[pid]["chunks"][:4]]}
        results = [session.call("read_paper", args, {pid}) for _ in range(2)]
        self.assertEqual(session.chars, sum(c["chars"] for r in results for c in r["chunks"]))
        self.assertLessEqual(session.chars, 4000)
        self.assertEqual(session.call("read_paper", args, {pid})["status"], "read_budget_exhausted")
        session = self.session(max_papers=1)
        session.corpus.by_id["venue/other"] = {**RECORD, "id": "venue/other"}
        with patch.object(session.store, "get", side_effect=FullTextError("not_pdf", "not PDF")) as get:
            for candidate in [pid, pid, "venue/other"]:
                result = session.call("open_paper", {"paper_id": candidate}, set(session.corpus.by_id))
        self.assertEqual(get.call_count, 1)
        self.assertEqual(result["status"], "paper_budget_exhausted")

    def test_invalid_budgets_and_outline_arguments(self):
        for options in [{"max_papers": 1.5}, {"read_chars": 0}, {"total_open_seconds": float("nan")}]:
            with self.assertRaises(ValueError):
                self.session(**options)
        session = self.session()
        for offset in [-1, 1.5, True, "not-an-integer"]:
            result = session.call("open_paper", {"paper_id": RECORD["id"], "outline_offset": offset}, {RECORD["id"]})
            self.assertEqual(result["status"], "invalid_arguments")

    def test_real_worker_pdf_parse_cache_offline_and_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = FullTextConfig(enabled=True, cache_dir=d)
            store = FullTextStore(cfg)
            directory = seed_real_pdf(store)
            document, hit = store.get(RECORD, 60)
            self.assertFalse(hit)
            self.assertEqual(document["page_count"], 3)
            self.assertEqual(document["fetched_at"], "offline-fixture")
            self.assertIn(QUOTE, " ".join(c["text"] for c in document["chunks"]))
            self.assertEqual(document["text_sha256"], digest(document["chunks"]))
            offline = FullTextStore(dataclasses.replace(cfg, offline=True))
            with patch("autoresearch_analogy.fulltext.subprocess.run") as worker:
                cached, hit = offline.get(RECORD, 5)
                self.assertTrue(hit)
                self.assertEqual(cached["pdf_sha256"], document["pdf_sha256"])
                worker.assert_not_called()
            (directory / "paper.pdf").write_bytes(b"corrupted")
            with self.assertRaises(FullTextError) as exc:
                offline.get(RECORD, 5)
            self.assertEqual(exc.exception.status, "cache_error")

    def test_offline_miss_does_not_create_cache_or_worker(self):
        with tempfile.TemporaryDirectory() as d:
            store = FullTextStore(FullTextConfig(cache_dir=d, offline=True))
            with patch("autoresearch_analogy.fulltext.subprocess.run") as worker:
                with self.assertRaises(FullTextError) as exc:
                    store.get(RECORD, 1)
            self.assertEqual(exc.exception.status, "offline_cache_miss")
            self.assertFalse(list(Path(d).iterdir()))
            worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
