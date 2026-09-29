"""Retrieval pipeline v0.2.0: hybrid dense+BM25 -> RRF fusion -> cross-encoder
rerank -> governance (temporal boost, volatile recency, supersede filter,
history boost) -> multi-hop expansion when needed -> relevance gate.

Stages:
  1. Intent parsing (pure functions of the query): time window, history
     intent, volatile-fact intent.
  2. Candidate generation: top DENSE_POOL by cosine similarity (dense) and
     top BM25_POOL by BM25, over the user's namespace only.
  3. Fusion: Reciprocal Rank Fusion over the union of both rankings.
  4. Rerank: cross-encoder scores the top RERANK_POOL fused candidates.
  5. Multi-hop (gated): when the first pass is weak or leaves query entities
     uncovered, expand the query with rare terms from the top hits and run
     a second pass; fuse both passes by max logit. Capped at 2 hops,
     deterministic.
  6. Governance boosts: +TEMPORAL_BOOST for docs inside the query's time
     window.
  7. Supersede filter: superseded docs are excluded unless the query shows
     history intent, in which case they get +HISTORY_BOOST (the old fact
     is the answer).
  8. Volatile-fact boost: when the query asks about a changeable attribute
     in the present tense, the newest surviving doc per volatile anchor
     gets +VOLATILE_BOOST. Runs after the supersede filter so the boost
     never lands on a doc that is then thrown away.
  9. Gate: if the best boosted score < RELEVANCE_THRESHOLD, the query is
     judged to have no relevant memory -> return [].
  10. Recency: tiny additive tie-break for ordering (deliberately small so
     fact recall is not recency-biased).
  11. Scoring: score = sigmoid(boosted - threshold), most-relevant first.

Session note: the Search contract carries no session_id, so session signals
cannot be applied at query time; per-user namespacing is the isolation
mechanism. This is documented, not silently faked.
"""
from __future__ import annotations

import math

import numpy as np

from . import config
from .store import UserIndex, tokenize
from . import governance as gov


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


def _single_pass(idx: UserIndex, query: str, embedder, reranker,
                 ) -> tuple[list[int], list[float]]:
    """Stages 2-4 for one query string. Returns (doc indices, raw logits).

    Must be called with idx.lock held.
    """
    n = idx.n_docs
    q_tokens = tokenize(query)

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

    fused = _rrf_fuse(dense_order, bm25_order)
    if not fused:
        return [], []
    fused_order = sorted(fused, key=lambda i: fused[i], reverse=True)
    cand = fused_order[: min(config.RERANK_POOL, len(fused_order))]

    docs = [idx.docs[i] for i in cand]
    logits = [float(x) for x in reranker.rerank(query, [d.content for d in docs])]
    return cand, logits


def _expansion_terms(idx: UserIndex, query: str,
                     top_ids: list[int]) -> list[str]:
    """Rare content words from the top hits that are not in the query.

    Ranked by corpus rarity then alphabetically — deterministic. Must be
    called with idx.lock held.
    """
    q_toks = set(tokenize(query))
    seen: dict[str, int] = {}
    for i in top_ids:
        for t in idx.docs[i].tokens:
            if t in q_toks or len(t) < 4:
                continue
            seen[t] = seen.get(t, 0) + 1
    ranked = sorted(seen, key=lambda t: (-1.0 / (1 + idx.df.get(t, 0)), t))
    return ranked[: config.MULTIHOP_MAX_TERMS]


def _needs_second_hop(query: str, cand: list[int], logits: list[float],
                      idx: UserIndex) -> bool:
    """Trigger hop 2 when the first pass is weak, or when the query names
    entities the top hits do not cover. Pure/deterministic."""
    if not cand:
        return False
    if max(logits) < config.MULTIHOP_WEAK_THRESHOLD:
        return True
    entities = [e for e in gov.extract_proper_nouns(query)]
    if len(entities) >= 2:
        covered: set[str] = set()
        for i in cand[:5]:
            covered |= set(idx.docs[i].anchors)
        if any(e not in covered for e in entities):
            return True
    return False


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

        # --- Stage 1: intent parsing (pure functions of the query) ---
        t_start, t_end = gov.parse_time_window(query, now_ms)
        history = gov.is_history_query(query)
        volatile_q = gov.is_volatile_query(query)

        # --- Stages 2-4: first pass ---
        cand, logits = _single_pass(idx, query, embedder, reranker)
        fused: dict[int, float] = dict(zip(cand, logits))

        # --- Stage 5: gated second hop ---
        if _needs_second_hop(query, cand, logits, idx):
            terms = _expansion_terms(idx, query, cand[:3])
            if terms:
                cand2, logits2 = _single_pass(
                    idx, query + " " + " ".join(terms), embedder, reranker)
                for i, lg in zip(cand2, logits2):
                    if i not in fused or lg > fused[i]:
                        fused[i] = lg

        if not fused:
            return []

        # --- Stage 6: governance boosts ---
        # Temporal boost for docs inside the query's parsed time window.
        scored: dict[int, float] = {}
        for i, lg in fused.items():
            d = idx.docs[i]
            s = lg
            if t_start is not None and t_start <= d.ts_ms < t_end:
                s += config.TEMPORAL_BOOST
            scored[i] = s

        # --- Stage 7: supersede filter / history boost ---
        # Runs BEFORE the volatile boost: the "newest doc per anchor" must
        # be chosen among the docs that survive filtering, otherwise the
        # boost lands on a superseded doc and is thrown away with it.
        kept: dict[int, float] = {}
        for i, s in scored.items():
            d = idx.docs[i]
            if d.superseded_by is not None and not history:
                continue  # demoted hard: old facts don't surface
            if history and d.superseded_by is not None:
                s += config.HISTORY_BOOST  # the old fact is the answer
            kept[i] = s
        if not kept:
            return []

        # --- Stage 8: volatile-fact boost (on surviving docs only) ---
        # When the query asks about a changeable attribute in the present
        # tense, the newest doc per volatile anchor wins ties.
        if volatile_q:
            newest: dict[str, int] = {}
            for i in kept:
                for a in idx.docs[i].frame_anchors:
                    if gov.is_volatile_anchor(a):
                        if (a not in newest
                                or idx.docs[newest[a]].ts_ms < idx.docs[i].ts_ms):
                            newest[a] = i
            for i in set(newest.values()):
                kept[i] += config.VOLATILE_BOOST

        # --- Stage 9: relevance gate (on the boosted best score) ---
        if max(kept.values()) < config.RELEVANCE_THRESHOLD:
            return []

        # --- Stages 10-11: recency tie-break, score, order, cut ---
        items = list(kept.items())
        final = [s + _recency_boost(idx.docs[i].ts_ms, now_ms) for i, s in items]
        order = np.argsort([-s for s in final], kind="stable")
        results: list[dict] = []
        for pos in order[:top_k]:
            i, boosted = items[int(pos)]
            d = idx.docs[i]
            score = 1.0 / (1.0 + math.exp(-(boosted - config.RELEVANCE_THRESHOLD)))
            results.append(
                {
                    "id": d.id,
                    "content": d.content,
                    "score": round(score, 6),
                    "created_at": d.created_at,
                }
            )
        return results
