"""Shared configuration for Omni Slated Mem Core v0.2.0."""
from __future__ import annotations

from pathlib import Path

SYSTEM_NAME = "Omni Slated Mem Core"
SYSTEM_VERSION = "0.2.0"

# Repo root resolved from this file so model paths work regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Model locations (vendored, loaded from local disk; no network at runtime) ---
EMBED_MODEL_DIR = str(_REPO_ROOT / "models" / "embed")
EMBED_MODEL_FILE = "model_optimized.onnx"   # BAAI/bge-small-en-v1.5 (ONNX, 384-dim)
RERANK_MODEL_DIR = str(_REPO_ROOT / "models" / "rerank")
RERANK_MODEL_FILE = "onnx/model.onnx"        # Xenova/ms-marco-MiniLM-L-6-v2 (cross-encoder)

# --- Retrieval pipeline tunables ---
DENSE_POOL = 200        # top-N dense candidates entering fusion
BM25_POOL = 200         # top-N BM25 candidates entering fusion
RRF_K = 60              # RRF constant: score = 1 / (RRF_K + rank)
RERANK_POOL = 50        # fused candidates sent to the cross-encoder
TOP_K_MAX = 100         # hard cap on requested top_k
TOP_K_DEFAULT = 5

# Relevance gate: cross-encoder raw logit threshold. Result sets whose best
# candidate scores below this are treated as "nothing relevant" and return [].
# Calibrated on a synthetic set (scripts/calibrate.py): 12 relevant queries
# (11/12 kept; the one lost is a paraphrase the cross-encoder scored -5.20),
# 12 off-topic queries (max -8.57, all rejected), 7 near-miss queries
# (closest: -5.84, rejected). t=0.0 maximizes recall while rejecting every
# irrelevant/near-miss case; t=2.0 additionally dropped a clean fact recall.
RELEVANCE_THRESHOLD = 0.0

# Recency: tiny additive tie-break, exp decay with a 1-year half-life scale.
# Deliberately small so fact recall is not recency-biased.
RECENCY_WEIGHT = 0.02
RECENCY_TAU_DAYS = 365.0

# --- Chunking ---
CHUNK_WORDS = 200       # long messages split into word windows of this size
CHUNK_OVERLAP = 40      # overlapping words between consecutive chunks

# --- BM25 ---
BM25_K1 = 1.2
BM25_B = 0.75

# --- v0.2.0 governance tunables ---
# Temporal boost: additive logit bonus for docs inside the query's parsed
# time window. Large enough to dominate ranking among topically similar docs.
TEMPORAL_BOOST = 3.0

# Volatile-fact boost: when the query asks about a changeable attribute in
# the present tense (location, employer, favorites, attitudes), the newest
# doc per volatile anchor gets this bonus — newer values win ties.
VOLATILE_BOOST = 2.0

# History boost: for explicit history queries ("where did I used to live"),
# superseded docs get this bonus so the old fact surfaces as the answer.
HISTORY_BOOST = 2.0

# Multi-hop: second retrieval pass triggers when the first pass's best logit
# is below this, or when the query names 2+ entities the top hits don't
# cover. Expansion takes this many rare terms from the top-3 hits.
MULTIHOP_WEAK_THRESHOLD = 1.0
MULTIHOP_MAX_TERMS = 8

# Consolidation: per-session compacted fact docs.
CONSOLIDATED_MAX_FACTS = 48   # cap on facts per session doc
CONSOLIDATED_MIN_FACTS = 2    # don't bother below this many facts

# --- Concurrency ---
# 64 concurrent Add workers per the platform contract; headroom above that.
THREADPOOL_WORKERS = 128
