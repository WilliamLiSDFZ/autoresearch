"""Paper corpus + BM25 search for the analogy agent.

The corpus is `output/paper_corpus/records.jsonl` + `manifest.json` from the KB repo
(`scripts/6_build_paper_corpus.py`): one record per paper with title, tldr and abstract. There
is no preprocessing on the KB side and no embedding anywhere — the agent's queries are short
mechanism-vocabulary strings, and lexical matching over title+tldr+abstract is the whole
retrieval (design: `Agentic_Knowledge_Base/docs/analogy_bm25_agent_design.md` §4.1).

BM25 is built at load time (rank_bm25.BM25Okapi, default k1/b). Loading happens once per
process, with the actual corpus digest checked on every load. After that search/get
are read-only and thread-safe. CLI invocations rebuild their own in-memory indexes.

Tokenizer: lowercase -> [a-z0-9]+ -> drop stopwords (function words plus paper boilerplate) ->
Porter stem. nltk is required so missing dependencies cannot change retrieval behavior.
The title is repeated once in the document text as a cheap title boost.
"""
# Adapted from MLEvolve engine/analogy/corpus.py; no sibling runtime import.
from __future__ import annotations

import json
import hashlib
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List

logger = logging.getLogger("autoresearch.analogy")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Function words plus the words that appear in nearly every ML abstract. Dropping the second
# group shortens documents without losing signal (their IDF is ~0 anyway) and, more importantly,
# keeps a long query from being diluted by them.
STOPWORDS = frozenset("""
a an the of to in on for with by and or as at from is are was were be been being it its we our
they their which who whom what where when how than then there here also into via using use used
based can could may might will would should must not no nor do does did done such each any all
both more most much many few less least very between among within without through over under
across per about after before during while this that these those has have had having if else
one two three first second i ii iii etc et al
paper propose proposed proposes present presents presented method methods approach approaches
results result show shows shown novel new existing work works model models data however
demonstrate demonstrates extensive experiments experimental performance state art outperform
outperforms achieve achieves achieved significantly
""".split())


def _make_stemmer() -> Callable[[str], str]:
    from nltk.stem import PorterStemmer
    stem = PorterStemmer().stem
    memo: Dict[str, str] = {}

    def cached(w: str) -> str:               # vocabulary << token count; Porter is the slow part
        s = memo.get(w)
        if s is None:
            s = memo[w] = stem(w)
        return s
    return cached


class PaperCorpus:
    def __init__(self, records: List[dict], manifest: dict):
        self.records = records
        self.manifest = manifest
        self.by_id: Dict[str, dict] = {r["id"]: r for r in records}
        self._stem = _make_stemmer()
        t0 = time.time()
        docs = [self.tokenize(f"{r['title']} {r['title']} {r.get('tldr', '')} {r.get('abstract', '')}")
                for r in records]
        t1 = time.time()
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(docs)
        logger.info("[analogy] corpus: %d papers, sha1 %s (tokenize %.1fs, bm25 %.1fs)",
                    len(records), self.digest, t1 - t0, time.time() - t1)

    # ------------------------------------------------------------------ identity

    # 返回语料清单中的短 SHA1 标识。
    @property
    def digest(self) -> str:
        return str(self.manifest.get("records_sha1", "?"))

    # 返回加载时校验并计算的语料完整 SHA256。
    @property
    def records_sha256(self) -> str:
        return str(self.manifest.get("records_sha256", ""))

    # 提供语料完整 SHA256 的简写访问入口。
    @property
    def sha256(self) -> str:
        return self.records_sha256

    # 返回各会议论文数量的独立副本。
    @property
    def venues(self) -> Dict[str, int]:
        return dict(self.manifest.get("venues", {}))

    # ------------------------------------------------------------------ tools

    # 统一小写、去除停用词并提取 Porter 词干。
    def tokenize(self, text: str) -> List[str]:
        return [self._stem(w) for w in _TOKEN_RE.findall(text.lower())
                if w not in STOPWORDS and len(w) > 1]

    # 按 BM25 分数稳定排序，返回紧凑的论文候选信息。
    def search(self, query: str, k: int = 10) -> List[dict]:
        """Top-k papers for a query: [{id, venue, title, tldr, score}]. No abstract, on purpose:
        the agent asks for those separately, so a broad search does not flood its context."""
        toks = self.tokenize(query)
        if not toks:
            return []
        import numpy as np
        scores = self.bm25.get_scores(toks)
        k = max(1, min(int(k), len(scores)))
        top = np.argsort(-scores, kind="stable")[:k]
        out = []
        for i in top:
            if scores[i] <= 0:
                break
            r = self.records[int(i)]
            out.append({"id": r["id"], "venue": r["venue"], "title": r["title"],
                        "tldr": (r.get("tldr") or "")[:300], "score": round(float(scores[i]), 2)})
        return out

    # 按给定 ID 顺序读取摘要，跳过不存在的论文。
    def get(self, ids: List[str]) -> List[dict]:
        """Full abstracts for known ids, in the order given; unknown ids are skipped."""
        out = []
        for pid in ids:
            r = self.by_id.get(str(pid))
            if r is not None:
                out.append({"id": r["id"], "venue": r["venue"], "title": r["title"],
                            "abstract": r.get("abstract", "")})
        return out

    def __contains__(self, pid: object) -> bool:
        return pid in self.by_id

    def __len__(self) -> int:
        return len(self.records)


# 校验语料格式、数量、唯一 ID 和内容哈希后读取记录。
def read_corpus_dir(corpus_dir: Path) -> tuple[List[dict], dict]:
    """Validate the portable schema-2 corpus before returning records and its identity."""
    corpus_dir = Path(corpus_dir).expanduser().resolve()
    raw = (corpus_dir / "records.jsonl").read_bytes()
    manifest = json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 2
            or manifest.get("level") != "paper"):
        raise ValueError("Expected a schema-2 paper corpus manifest")
    actual = hashlib.sha1(raw).hexdigest()[:12]
    if manifest.get("records_sha1") != actual:
        raise ValueError("Paper corpus records_sha1 does not match records.jsonl")
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    if (not records or type(manifest.get("count")) is not int
            or manifest["count"] != len(records)):
        raise ValueError("Paper corpus count is empty or differs from the manifest")
    ids, venues = set(), {}
    for record in records:
        required = ("id", "venue", "category", "title", "source", "pdf_url", "tldr", "abstract")
        if not isinstance(record, dict) or any(not isinstance(record.get(k), str) for k in required):
            raise ValueError("Paper corpus records require schema-2 string fields")
        if any(not record[k].strip() for k in ("id", "venue", "title", "abstract")):
            raise ValueError("Paper ids, venues, titles and abstracts must be nonempty")
        categories = record.get("categories")
        if (not isinstance(categories, list) or not categories
                or not all(isinstance(c, str) and c.strip() for c in categories)
                or record["category"] not in categories):
            raise ValueError("Paper corpus categories must contain the primary category")
        if record["id"] in ids or not record["id"].startswith(record["venue"] + "/"):
            raise ValueError("Paper corpus IDs must be unique and prefixed by their venue")
        ids.add(record["id"])
        venues[record["venue"]] = venues.get(record["venue"], 0) + 1
    if manifest.get("venues") != venues:
        raise ValueError("Paper corpus venue counts differ from the manifest")
    manifest = {**manifest, "records_sha256": hashlib.sha256(raw).hexdigest()}
    return records, manifest


_CACHE: Dict[str, PaperCorpus] = {}
_LOCK = threading.Lock()


# 复用已加载的 BM25 索引，并拒绝同路径语料发生变化。
def load_corpus(corpus_dir: str | Path) -> PaperCorpus:
    """Load a verified corpus, rejecting changes at an already loaded path."""
    key = str(Path(corpus_dir).expanduser().resolve())
    records, manifest = read_corpus_dir(Path(key))
    with _LOCK:
        if key in _CACHE:
            if _CACHE[key].records_sha256 != manifest["records_sha256"]:
                raise ValueError("Paper corpus changed after it was loaded")
            return _CACHE[key]
        corpus = PaperCorpus(records, manifest)
        _CACHE[key] = corpus
        return corpus
