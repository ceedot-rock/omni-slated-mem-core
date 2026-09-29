# Omni Slated Mem Core — v0.1.0

Agent Memory Challenge Cycle 2 entry: **textual track, academic division**.

A zero-LLM local memory system. No language-model calls at any stage — no
API keys, no per-query inference cost, fully deterministic and exactly
reproducible. Retrieval only: local embeddings, lexical search, fusion,
cross-encoder reranking, and relevance gating.

## Contract

| Endpoint | Behavior |
|---|---|
| `POST /v1/memories/add` | `{request_id, user_id, session_id, messages:[{role, content, timestamp?}]}` → `{success:true, request_id, user_id, session_id}`, HTTP 200 **only after** the messages are stored **and** searchable (synchronous durability). |
| `POST /v1/memories/search` | `{query, user_id, top_k, options?}` → `{data:[{id, content, score, created_at}]}`, most-relevant first. `[]` when nothing is relevant. |
| `GET /health` | 2xx, unauthenticated: `{status:"ok", system, version}` |

`user_id` is the isolation boundary: a user only ever retrieves their own
memories. Message `timestamp` is Unix milliseconds; `created_at` in results
is ISO-8601 UTC.

## Method

**Ingest.** Each message becomes one document (long messages are split into
200-word windows with 40 words of overlap so no fact is truncated away).
Every chunk is embedded with a local ONNX model
(`BAAI/bge-small-en-v1.5`, 384-dim, first-token pooling, L2-normalized —
the same post-processing the reference pipeline uses) and indexed for BM25
(k1=1.2, b=0.75, lowercase alphanumeric tokens). Document IDs are
`{request_id}#{message}.{chunk}` — deterministic and traceable.

**Retrieval pipeline** (per query, inside the user's namespace only):

1. **Candidates** — top 200 by cosine similarity (dense) plus top 200 by
   BM25 (lexical). The lexical arm catches exact terms, codes, and names
   that dense retrieval can miss.
2. **Fusion** — Reciprocal Rank Fusion over the union of both rankings
   (k=60). No score normalization needed; rank-based.
3. **Rerank** — a local cross-encoder (`ms-marco-MiniLM-L-6-v2`, ONNX)
   scores the top 50 fused candidates as (query, document) pairs; the raw
   logit is the relevance signal.
4. **Gate** — if the best logit is below the relevance threshold, the
   system returns `[]`. Relevance gating is a first-class behavior, not
   an afterthought.
5. **Order & score** — a tiny recency tie-break (exponential decay,
   one-year scale, weight 0.02 — deliberately too small to bias fact
   recall), then score = sigmoid(logit − threshold), so returned results
   land in (0.5, 1.0], most-relevant first.

**Session note (honest limitation).** The Search contract carries no
`session_id`, so session signals cannot be applied at query time. We do
not fake them: per-user namespacing is the isolation mechanism, and
session IDs are stored on every document for auditability.

## Calibration

`scripts/calibrate.py` builds a 28-document synthetic memory and runs
relevant, off-topic, and near-miss queries through the pipeline, reporting
cross-encoder logit distributions. Results:

- Relevant queries: mostly +1.5 to +10 (one paraphrased process question
  scored −5.20 — a genuine cross-encoder miss).
- Off-topic queries: max −8.57, all rejected.
- Near-miss queries (topic-adjacent but unanswerable, e.g. "What is my
  passport number?" when only a passport *mention* exists): closest −5.84,
  rejected.

Threshold **0.0** keeps 11/12 relevant queries while rejecting every
off-topic and near-miss case. A threshold of 2.0 additionally dropped a
clean fact recall ("What food do I hate?"), so 0.0 was chosen to favor
recall where the eval rewards it and precision where gating is scored.

## Concurrency

The platform runs 64 concurrent Add workers. The store holds one `RLock`
per user namespace plus a global registry lock — different users never
block each other, and an Add holds its user's lock from chunking through
index update, so HTTP 200 implies searchable. ONNX sessions run
single-threaded internally behind a lock; request-level parallelism comes
from a 128-worker thread pool. `scripts/smoke_test.py` hammers the service
with 64 concurrent writers and asserts durability, isolation, schema,
ordering, and gating.

## Layout

```
app/            service (main.py), store, retrieval, local ONNX inference
models/         vendored model files (loaded from disk; no downloads)
vendor/wheels/  pinned wheels for an offline Docker build
scripts/        smoke_test.py (contract test), calibrate.py (threshold)
Dockerfile      builds and serves on $PORT (default 8000)
```

## Run locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --port 8000
python scripts/smoke_test.py --port 18001   # boots its own server
```

## Docker

```bash
docker build -t omni-slated-mem-core:0.1.0 .
docker run -p 8000:8000 omni-slated-mem-core:0.1.0
# or: docker run -e PORT=8080 -p 8080:8080 omni-slated-mem-core:0.1.0
```

The build installs exclusively from `vendor/wheels` (`--no-index`); the
image contains no build tools beyond the slim base, and the models are
baked in — nothing is downloaded at build or run time.
