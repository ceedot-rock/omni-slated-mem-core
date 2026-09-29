"""Thread-safe in-memory memory store with per-user namespaces.

Durability model: everything is synchronous and in-memory. An Add returns
HTTP 200 only after the documents are chunked, embedded, appended, and
visible to Search under the same lock — so "stored AND searchable" holds
by construction.

Concurrency model: one RLock per user namespace plus a global RLock for the
user registry. 64 concurrent Add workers each take their own user's lock;
different users never block each other.

v0.2.0 governance: contradiction / supersede tracking (old facts contradicted
by new ones are linked, not deleted), an anchor pre-filter index, and
per-session consolidated fact documents. All of it mutates only under the
user's lock, so supersede writes are atomic with the Add.
"""
from __future__ import annotations

import math
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from . import config

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= config.CHUNK_WORDS:
        return [text]
    chunks: list[str] = []
    step = config.CHUNK_WORDS - config.CHUNK_OVERLAP
    for start in range(0, len(words), step):
        piece = words[start : start + config.CHUNK_WORDS]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + config.CHUNK_WORDS >= len(words):
            break
    return chunks


def iso_from_ms(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).isoformat()


@dataclass
class Doc:
    id: str
    content: str
    user_id: str
    session_id: str
    role: str
    ts_ms: int
    created_at: str
    tokens: list[str] = field(default_factory=list)
    length: int = 0
    # v0.2.0 governance fields
    anchors: frozenset = frozenset()          # frame anchors + proper nouns
    frame_anchors: frozenset = frozenset()    # frame anchor keys only
    frames: list = field(default_factory=list)  # governance.Frame list
    superseded_by: int | None = None         # doc index of the contradicting doc
    supersedes: list = field(default_factory=list)  # doc indices this doc supersedes
    is_consolidated: bool = False            # per-session compacted fact doc


class UserIndex:
    """All state for one user_id. Every public method takes self.lock."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.docs: list[Doc] = []
        self.embeddings: list[np.ndarray] = []  # parallel to docs; L2-normalized
        self.df: Counter = Counter()            # term -> #docs containing term
        self.total_len = 0
        self.anchor_index: dict[str, list[int]] = {}   # anchor -> doc indices
        self.session_consolidated: dict[str, int] = {}  # session_id -> doc index

    @property
    def n_docs(self) -> int:
        return len(self.docs)

    def add(
        self,
        user_id: str,
        chunks: list[tuple[str, str, str, str, int, str]],
        vectors: np.ndarray,
        embedder=None,
    ) -> None:
        """chunks: (doc_id, content, session_id, role, ts_ms, created_at).

        v0.2.0: under the same lock, runs contradiction detection (new docs
        supersede contradicted old docs — linked, not deleted) and rebuilds
        per-session consolidated fact documents. embedder is the local
        embedding model (needed for consolidated-doc embeddings); when None,
        consolidation is skipped.
        """
        # Local import: governance imports store, so this must not be top-level.
        from . import governance as gov

        with self.lock:
            base = len(self.docs)
            new_docs: list[Doc] = []
            for k, ((doc_id, content, session_id, role, ts_ms, created_at),
                    vec) in enumerate(zip(chunks, vectors)):
                toks = tokenize(content)
                frames = gov.extract_frames(content)
                frame_anchors = frozenset(f.anchor for f in frames)
                doc = Doc(
                    id=doc_id,
                    content=content,
                    user_id=user_id,
                    session_id=session_id,
                    role=role,
                    ts_ms=ts_ms,
                    created_at=created_at,
                    tokens=toks,
                    length=len(toks),
                    anchors=gov.extract_anchors(content, frames),
                    frame_anchors=frame_anchors,
                    frames=frames,
                )
                new_docs.append(doc)
                self.docs.append(doc)
                self.embeddings.append(vec)
                for t in set(toks):
                    self.df[t] += 1
                self.total_len += len(toks)
                doc_idx = base + k
                for a in doc.anchors:
                    self.anchor_index.setdefault(a, []).append(doc_idx)

            # --- contradiction / supersede detection (newest wins) ---
            # Candidates are restricted to strictly older docs (index <
            # doc_idx): within one Add batch, an earlier message can never
            # be superseded by a later message in the same batch.
            for k, doc in enumerate(new_docs):
                doc_idx = base + k
                cand_idx: set[int] = set()
                for a in doc.anchors:
                    cand_idx.update(self.anchor_index.get(a, ()))
                cand_idx = {i for i in cand_idx if i < doc_idx}
                candidates = [(i, self.docs[i]) for i in sorted(cand_idx)]
                hit_idx = gov.detect_contradictions(doc, candidates)
                for old_idx in hit_idx:
                    if self.docs[old_idx].superseded_by is None:
                        self.docs[old_idx].superseded_by = doc_idx
                        doc.supersedes.append(old_idx)

            # --- per-session consolidation ---
            if embedder is not None:
                sessions = list(dict.fromkeys(c[2] for c in chunks))
                for session_id in sessions:
                    self._consolidate_session(
                        gov, user_id, session_id, embedder)

    def _consolidate_session(self, gov, user_id: str, session_id: str,
                             embedder) -> None:
        """Rebuild the session's compacted fact list; index it as one doc.

        The new consolidated doc supersedes the previous one for the session
        (linked, not deleted). Skipped when the fact set is unchanged.
        Must be called with self.lock held.
        """
        facts: list[str] = []
        session_ts = 0
        session_created = ""
        for d in self.docs:
            if d.session_id != session_id:
                continue
            if d.ts_ms >= session_ts:
                session_ts = d.ts_ms
                session_created = d.created_at
            if d.is_consolidated or d.superseded_by is not None:
                continue
            facts.extend(gov.extract_fact_sentences(d.content))
        facts = gov.dedupe_facts(facts)[-config.CONSOLIDATED_MAX_FACTS:]
        if len(facts) < config.CONSOLIDATED_MIN_FACTS:
            return
        content = "\n".join(facts)
        old_idx = self.session_consolidated.get(session_id)
        if old_idx is not None and self.docs[old_idx].content == content:
            return  # unchanged; no churn
        doc_id = f"{session_id}#consolidated.{len(self.docs)}"
        toks = tokenize(content)
        frames = gov.extract_frames(content)
        doc = Doc(
            id=doc_id,
            content=content,
            user_id=user_id,
            session_id=session_id,
            role="system",
            ts_ms=session_ts,
            created_at=session_created,
            tokens=toks,
            length=len(toks),
            anchors=gov.extract_anchors(content, frames),
            frame_anchors=frozenset(f.anchor for f in frames),
            frames=frames,
            is_consolidated=True,
        )
        if old_idx is not None:
            doc.supersedes.append(old_idx)
            self.docs[old_idx].superseded_by = len(self.docs)
        vec = embedder.embed([content])[0]
        self.docs.append(doc)
        self.embeddings.append(vec)
        for t in set(toks):
            self.df[t] += 1
        self.total_len += len(toks)
        for a in doc.anchors:
            self.anchor_index.setdefault(a, []).append(len(self.docs) - 1)
        self.session_consolidated[session_id] = len(self.docs) - 1

    def embedding_matrix(self) -> np.ndarray | None:
        with self.lock:
            if not self.embeddings:
                return None
            return np.vstack(self.embeddings)

    def bm25(self, q_tokens: list[str]) -> np.ndarray:
        """BM25 scores for every doc. Must be called with lock held."""
        n = len(self.docs)
        if n == 0:
            return np.array([], dtype=np.float64)
        avgdl = self.total_len / n if n else 0.0
        qtf = Counter(q_tokens)
        scores = np.zeros(n, dtype=np.float64)
        k1, b = config.BM25_K1, config.BM25_B
        for term, qf in qtf.items():
            df = self.df.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, doc in enumerate(self.docs):
                tf = doc.tokens.count(term)
                if tf == 0:
                    continue
                denom = tf + k1 * (1 - b + b * (doc.length / avgdl if avgdl else 0.0))
                scores[i] += idf * (tf * (k1 + 1) / denom) * qf
        return scores


class Store:
    def __init__(self) -> None:
        self._global = threading.RLock()
        self._users: dict[str, UserIndex] = {}

    def get_or_create(self, user_id: str) -> UserIndex:
        with self._global:
            idx = self._users.get(user_id)
            if idx is None:
                idx = UserIndex()
                self._users[user_id] = idx
            return idx

    def get(self, user_id: str) -> UserIndex | None:
        with self._global:
            return self._users.get(user_id)

    def user_count(self) -> int:
        with self._global:
            return len(self._users)

    def doc_count(self, user_id: str) -> int:
        idx = self.get(user_id)
        if idx is None:
            return 0
        with idx.lock:
            return idx.n_docs


def now_ms() -> int:
    return int(time.time() * 1000)
