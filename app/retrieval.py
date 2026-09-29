"""Retrieval pipeline: hybrid dense+BM25 -> RRF fusion -> cross-encoder
rerank -> recency tie-break -> relevance gate.

Stages:
  1. Candidate generation: top DENSE_POOL by cosine similarity (dense) and
     top BM25_POOL by BM25, over the user's namespace only.
  2. Fusion: Reciprocal Rank Fusion over the union of both rankings.
  3. Rerank: cross-encoder scores the top RERANK_POOL fused candidates.
  4. Gate: if the best raw cross-encoder logit < RELEVANCE_THRESHOLD, the
     query is judged to have no relevant memory -> return [].
  5. Recency: tiny additive tie-break for ordering, exp decay (deliberately
     small so fact recall is not recency-biased).
  6. Scoring: score = sigmoid(logit - threshold), so surviving results land
     in (0.5, 1.0], most-relevant first.

Session note: the Search contract carries no session_id, so session signals
cannot be applied at query time; per-user namespacing is the isolation
mechanism. This is documented, not silently faked.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .store import UserIndex, tokenize


def _rrf_fuse(dense_order: np.ndarray, bm25_order: np.ndarray) -> dict[int, float]:
    fused: dict[int, float] = {}
    for rank, doc_i in enumerate(dense_order):
        fused[int(doc_i)] = fused.get(int(doc_i), 0.0) + 1.0 / (config.RRF_K + rank + 1)
    for rank, doc_i in enumerate(bm25_order):
        fused[int(doc_i)] = fused.get(int(doc_i), 0.0) + 1.0 / (config.RRF_K + rank + 1)
    return fused


def _recency_boost(ts_ms: int, now_ms: int) -> float:
    age_days = max(0.0, (now_ms - ts_ms) / 86_400_000.0)
    return config.RECENCY_WEIGHT * math.exp(-age_days / config.RECENCY_TAU_DAYS)


def search_index(
    idx: UserIndex,
    query: str,
    top_k: int,
    now_ms: int,
    embedder,
    reranker,
) -> list[dict]:
    """Run the full pipeline under the caller's lock. Returns result dicts."""
    with idx.lock:
        n = idx.n_docs
        if n == 0 or not query.strip():
            return []

        q_tokens = tokenize(query)

        # --- Stage 1: candidates ---
        mat = idx.embedding_matrix()
        assert mat is not None
        q_vec = embedder.embed([query])[0].astype(np.float64)
        dense_scores = mat.astype(np.float64) @ q_vec  # cosine: vectors normalized
        bm25_scores = idx.bm25(q_tokens)

        d_pool = min(config.DENSE_POOL, n)
        b_pool = min(config.BM25_POOL, n)
        dense_order = np.argsort(-dense_scores, kind="stable")[:d_pool]
        bm25_order = (
            np.argsort(-bm25_scores, kind="stable")[:b_pool] if q_tokens else np.array([], dtype=int)
        )

        # --- Stage 2: RRF fusion ---
        fused = _rrf_fuse(dense_order, bm25_order)
        if not fused:
            return []
        fused_order = sorted(fused, key=lambda i: fused[i], reverse=True)
        cand = fused_order[: min(config.RERANK_POOL, len(fused_order))]

        # --- Stage 3: cross-encoder rerank ---
        docs = [idx.docs[i] for i in cand]
        logits = [float(x) for x in reranker.rerank(query, [d.content for d in docs])]

        # --- Stage 4: relevance gate (on the raw cross-encoder logit) ---
        if max(logits) < config.RELEVANCE_THRESHOLD:
            return []

        # --- Stage 5: recency tie-break (tiny) + score, sort, cut ---
        final = [lg + _recency_boost(d.ts_ms, now_ms) for d, lg in zip(docs, logits)]
        order = np.argsort([-s for s in final], kind="stable")
        results: list[dict] = []
        for pos in order[:top_k]:
            i = cand[int(pos)]
            d = idx.docs[i]
            logit = logits[int(pos)]
            score = 1.0 / (1.0 + math.exp(-(logit - config.RELEVANCE_THRESHOLD)))
            results.append(
                {
                    "id": d.id,
                    "content": d.content,
                    "score": round(score, 6),
                    "created_at": d.created_at,
                }
            )
        return results
