"""On-demand paper reading. No PDF/ML imports in the threaded agent process.

The corpus remains the retrieval authority. A subprocess downloads and parses each PDF;
immutable cached documents and the exact excerpts delivered to an episode are versioned.
"""
# Adapted from MLEvolve engine/analogy/fulltext.py; no sibling runtime import.
from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

SCHEMA_VERSION = "paper-fulltext-v1"
CHUNK_CHARS = 2000
OUTLINE_PAGE_SIZE = 30


@dataclass
class FullTextConfig:
    enabled: bool = False
    cache_dir: str = ""  # default: autoresearch/results/analogy-cache/paper_fulltext (shared on the PVC)
    offline: bool = False
    max_papers: int = 3
    max_read_calls: int = 12
    read_chars: int = 8000
    total_chars: int = 40000
    open_timeout_seconds: float = 60.0
    total_open_seconds: float = 150.0


# The standalone entry point may call this an options object; retain both names.
FullTextOptions = FullTextConfig


# 从上层配置提取全文读取参数，并补齐默认值。
def options_from_config(acfg: Any) -> FullTextConfig:
    raw = getattr(acfg, "fulltext", None)
    return FullTextConfig(**{f.name: getattr(raw, f.name, f.default)
                             for f in dataclasses.fields(FullTextConfig)})


# 汇总解析依赖版本和读取器源码哈希以标识缓存版本。
def parser_versions() -> dict:
    out = {"schema": SCHEMA_VERSION}
    out["reader_sha256"] = hashlib.sha256(Path(__file__).read_bytes() +
        Path(__file__).with_name("fulltext_worker.py").read_bytes()).hexdigest()
    for name in ("pymupdf", "pymupdf4llm", "pymupdf_layout", "tabulate"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = "unavailable"
    return out


# 将结构化内容规范序列化后计算 SHA256。
def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


# 根据论文已有来源生成去重的候选全文链接。
def resolve_urls(record: dict) -> list[str]:
    """Only deterministic links belonging to the selected corpus record; no web search."""
    urls = [str(record.get("pdf_url") or "").strip()]
    source = str(record.get("source") or "").strip()
    parsed = urlparse(source)
    host = (parsed.hostname or "").lower()
    if host == "aclanthology.org":
        urls.append(source.rstrip("/") if parsed.path.endswith(".pdf") else source.rstrip("/") + ".pdf")
    elif host == "openreview.net":
        paper = parse_qs(parsed.query).get("id", [""])[0]
        if paper:
            from urllib.parse import urlencode
            urls.append("https://openreview.net/pdf?" + urlencode({"id": paper}))
    elif host in {"arxiv.org", "export.arxiv.org"} and parsed.path.startswith(("/abs/", "/pdf/")):
        urls.append("https://arxiv.org/pdf/" + parsed.path.split("/", 2)[2])
    elif host in {"doi.org", "ojs.aaai.org", "proceedings.mlr.press", "papers.nips.cc",
                  "proceedings.neurips.cc", "openaccess.thecvf.com"}:
        urls.append(source)  # the worker may follow citation_pdf_url on the publisher page
    return list(dict.fromkeys(u for u in urls if u))


class FullTextError(Exception):
    def __init__(self, status: str, message: str):
        self.status = status
        super().__init__(message)


class FullTextStore:
    def __init__(self, cfg: FullTextConfig):
        self.cfg = cfg
        self.root = Path(cfg.cache_dir).expanduser() if cfg.cache_dir else (
            Path(__file__).resolve().parents[1] / "results" / "analogy-cache" / "paper_fulltext")
        self.root = self.root.resolve()
        self.versions = parser_versions()

    def _cached(self, directory: Path, record: dict | None = None) -> dict | None:
        p = directory / "document.json"
        if not p.exists():
            return None
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc["parser"] != self.versions or digest(doc["chunks"]) != doc["text_sha256"]:
                raise ValueError("document version or text hash mismatch")
            if record is not None and (doc["paper_id"] != record["id"] or doc["title"] != record["title"]):
                raise ValueError("cached document identity differs from the corpus record")
            if type(doc["page_count"]) is not int or not 1 <= doc["page_count"] <= 100:
                raise ValueError("invalid cached page count")
            if not isinstance(doc["warnings"], list) or not isinstance(doc["source_url"], str):
                raise ValueError("invalid cached document metadata")
            chunks, seen = doc["chunks"], set()
            if not isinstance(chunks, list) or not chunks:
                raise ValueError("cached document has no chunks")
            for chunk in chunks:
                if (not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str)
                        or not chunk["chunk_id"] or chunk["chunk_id"] in seen
                        or not isinstance(chunk.get("text"), str)
                        or type(chunk.get("chars")) is not int
                        or chunk["chars"] != len(chunk["text"])
                        or not 1 <= chunk["chars"] <= CHUNK_CHARS
                        or type(chunk.get("page")) is not int
                        or not 1 <= chunk["page"] <= doc["page_count"]
                        or not isinstance(chunk.get("section"), str)):
                    raise ValueError("invalid cached chunk or character count")
                seen.add(chunk["chunk_id"])
            pdf_hash = hashlib.sha256((directory / "paper.pdf").read_bytes()).hexdigest()
            if pdf_hash != doc["pdf_sha256"]:
                raise ValueError("PDF hash mismatch")
            return doc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise FullTextError("cache_error", f"Invalid cached document: {exc}") from exc

    # 优先读取有效缓存，否则在加锁后启动限时解析子进程。
    def get(self, record: dict, timeout: float) -> tuple[dict, bool]:
        urls = resolve_urls(record)
        if not urls:
            raise FullTextError("unavailable", "No supported full-text link in this corpus record")
        key = digest({"paper_id": record["id"], "title": record["title"],
                      "urls": urls, "parser": self.versions})
        directory = self.root / key
        cached = self._cached(directory, record)
        if cached is not None:
            return cached, True
        if self.cfg.offline:
            raise FullTextError("offline_cache_miss", "This exact paper/parser version is not cached")
        deadline = time.monotonic() + timeout
        directory.mkdir(parents=True, exist_ok=True)
        # A persistent flock file avoids stale locks after a killed worker, including across
        # processes/pods sharing the PVC. The bounded lock wait is part of the open budget.
        with (directory / ".lock").open("a") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise FullTextError("timeout", "Timed out waiting for another PDF reader")
                    time.sleep(0.05)
            cached = self._cached(directory, record)
            if cached is not None:
                return cached, True
            failed = directory / "failure.json"
            if failed.exists():
                error = json.loads(failed.read_text(encoding="utf-8"))
                if error.get("retry_after", 0) > time.time():
                    raise FullTextError(error["status"], error["message"])
            request = {"record": record, "urls": urls, "parser": self.versions}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise FullTextError("timeout", "Paper open budget exhausted")
            try:
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).with_name("fulltext_worker.py")), str(directory)],
                    input=json.dumps(request), capture_output=True, text=True,
                    encoding="utf-8", timeout=remaining, cwd=directory)
            except subprocess.TimeoutExpired as exc:
                raise FullTextError("timeout", "PDF download/parse exceeded the paper open budget") from exc
            cached = self._cached(directory, record)
            if proc.returncode == 0 and cached is not None:
                return cached, False
            try:
                error = json.loads((directory / "failure.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                error = {"status": "parse_error", "message": "PDF worker failed without a document"}
            raise FullTextError(error["status"], error["message"])


class PaperReadingSession:
    """Per-invocation budgets and provenance; never shared between concurrent agents."""
    def __init__(self, corpus: Any, cfg: FullTextConfig):
        self.corpus, self.cfg = corpus, cfg
        for field in ("max_papers", "max_read_calls", "read_chars", "total_chars"):
            if type(getattr(cfg, field)) is not int or getattr(cfg, field) <= 0:
                raise ValueError(f"analogy.fulltext.{field} must be a positive integer")
        for field in ("open_timeout_seconds", "total_open_seconds"):
            value = getattr(cfg, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"analogy.fulltext.{field} must be finite and positive")
        self.store = FullTextStore(cfg)
        self.documents: dict[str, dict] = {}
        self.attempts: dict[str, dict] = {}
        self.delivered: dict[tuple[str, str], dict] = {}
        self.events: list[dict] = []
        self.read_calls = self.chars = 0
        self.open_seconds = 0.0

    # 校验已检索论文 ID，执行阅读工具并记录调用轨迹。
    def call(self, tool: str, args: dict, seen_ids: set) -> dict:
        start = time.monotonic()
        pid = str(args.get("paper_id") or "")
        try:
            if pid not in seen_ids or pid not in self.corpus:
                raise FullTextError("rejected", "Only paper ids from this episode's search results are accepted")
            if tool == "open_paper":
                result = self._open(pid, args)
            elif tool == "read_paper":
                result = self._read(pid, args)
            else:
                raise FullTextError("invalid_arguments", "Unknown paper reading tool")
        except FullTextError as exc:
            result = {"status": exc.status, "paper_id": pid, "message": str(exc),
                      "note": "No full-text evidence returned by this call; read_abstract remains available"}
        except Exception as exc:
            result = {"status": "tool_error", "paper_id": pid, "message": f"{type(exc).__name__}: {exc}"[:400]}
        self.events.append({"tool": tool, "arguments": args, "result": result,
                            "seconds": round(time.monotonic() - start, 3)})
        return result

    def _open(self, pid: str, args: dict) -> dict:
        try:
            value = args.get("outline_offset", 0)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise ValueError
            offset = int(value)
            if offset < 0:
                raise ValueError
        except (ValueError, TypeError):
            raise FullTextError("invalid_arguments", "outline_offset must be a nonnegative integer")
        if pid not in self.documents:
            if pid in self.attempts:
                error = self.attempts[pid]
                raise FullTextError(error["status"], error["message"])
            if len(self.attempts) >= self.cfg.max_papers:
                raise FullTextError("paper_budget_exhausted", "No more distinct papers may be opened")
            remaining = self.cfg.total_open_seconds - self.open_seconds
            if remaining <= 0:
                raise FullTextError("time_budget_exhausted", "Total paper opening budget exhausted")
            self.attempts[pid] = {"status": "tool_error", "message": "Paper opening failed"}
            start = time.monotonic()
            try:
                doc, hit = self.store.get(self.corpus.by_id[pid], min(remaining, self.cfg.open_timeout_seconds))
                self.documents[pid] = doc
                self.attempts[pid] = {"status": "ok", "cache_hit": hit}
            except FullTextError as exc:
                self.attempts[pid] = {"status": exc.status, "message": str(exc)}
                raise
            finally:
                self.open_seconds += time.monotonic() - start
        doc = self.documents[pid]
        chunks = doc["chunks"]
        outline = [{k: c[k] for k in ("chunk_id", "page", "section", "chars")} for c in
                   chunks[offset:offset + OUTLINE_PAGE_SIZE]]
        return {"status": "ok", "paper_id": pid, "title": doc["title"],
                "source_url": doc["source_url"], "pdf_sha256": doc["pdf_sha256"],
                "text_sha256": doc["text_sha256"], "parser": doc["parser"],
                "page_count": doc["page_count"], "warnings": doc["warnings"],
                "cache_hit": self.attempts[pid]["cache_hit"], "outline": outline,
                "next_outline_offset": offset + len(outline) if offset + len(outline) < len(chunks) else None,
                "remaining_chars": self.cfg.total_chars - self.chars,
                "note": "Opening a paper returns no body text. Use read_paper to obtain citable excerpts."}

    def _read(self, pid: str, args: dict) -> dict:
        if self.read_calls >= self.cfg.max_read_calls:
            raise FullTextError("read_budget_exhausted", "No reading calls remain")
        self.read_calls += 1
        if pid not in self.documents:
            raise FullTextError("not_open", "Call open_paper successfully before read_paper")
        ids = args.get("chunk_ids")
        if not isinstance(ids, list) or not ids or len(ids) > 8 or not all(isinstance(i, str) for i in ids):
            raise FullTextError("invalid_arguments", "chunk_ids must contain 1-8 ids from the outline")
        doc = self.documents[pid]
        by_id = {c["chunk_id"]: c for c in doc["chunks"]}
        if any(i not in by_id for i in ids):
            raise FullTextError("invalid_arguments", "Unknown chunk id; use the paper's outline")
        remaining = min(self.cfg.read_chars, self.cfg.total_chars - self.chars)
        chunks, pending = [], []
        for cid in dict.fromkeys(ids):
            chunk = by_id[cid]
            if chunk["chars"] > remaining:
                pending.append(cid)
                continue
            chunks.append(chunk)
            remaining -= chunk["chars"]
            self.chars += chunk["chars"]
            self.delivered[(pid, cid)] = chunk
        return {"status": "ok" if chunks else "character_budget_exhausted", "paper_id": pid,
                "pdf_sha256": doc["pdf_sha256"], "text_sha256": doc["text_sha256"],
                "chunks": chunks, "not_returned_chunk_ids": pending,
                "remaining_chars": self.cfg.total_chars - self.chars,
                "remaining_read_calls": self.cfg.max_read_calls - self.read_calls}

    # 汇总本次阅读的配置、耗用、文档来源和工具记录。
    def snapshot(self) -> dict:
        return {"schema": SCHEMA_VERSION, "config": dataclasses.asdict(self.cfg),
                "cache_dir": str(self.store.root), "parser": self.store.versions,
                "read_calls": self.read_calls, "body_chars": self.chars,
                "open_seconds": round(self.open_seconds, 3), "attempts": self.attempts,
                "documents": {pid: {k: v for k, v in doc.items() if k != "chunks"}
                              for pid, doc in self.documents.items()}, "events": self.events}
