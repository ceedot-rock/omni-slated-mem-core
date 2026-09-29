# Submission notes (draft) — Agent Memory Challenge Cycle 2

> Draft prepared 2026-09-29, updated for v0.2.0. The repository goes public
> only after Corey Tasz reviews it. Nothing here has been submitted.

- **System:** Omni Slated Mem Core, version 0.2.0
- **Track:** textual · **Division:** academic (open-source methods)
- **Entrant:** Corey Ptaszenski — corey@slidphilabs.com
- **Submission form:** public GitHub repository + Dockerfile (this tree).
  The platform builds the image and runs it; the evaluated code is exactly
  what is in the repo.

## What the system does

A memory service exposing the platform's Add/Search contract. Memories are
stored per `user_id` (the isolation boundary) and retrieved through a
zero-LLM local pipeline: local ONNX text embeddings plus BM25, fused with
reciprocal rank fusion, reranked by a local cross-encoder, and passed
through a calibrated relevance gate that returns `[]` when nothing is
relevant. v0.2.0 adds a memory-management layer: contradiction/supersede
tracking (new facts atomically supersede contradicted old ones; history
queries surface the old facts), temporal understanding (time expressions
in queries boost documents in the parsed window; changeable facts get
recency preference), gated two-hop retrieval, and per-session
consolidation. No language-model calls, no API keys, no network access at
runtime — deterministic and exactly reproducible.

## Reproducibility

- All dependencies pinned in `requirements.txt`; exact wheels vendored in
  `vendor/wheels/` and installed with `--no-index`.
- Both models vendored under `models/` and loaded from local disk.
- Fixed random seed surface: none — there is no sampling anywhere in the
  pipeline. Same inputs → same outputs, byte for byte.

## Method summary

1. Ingest: one document per message (200-word overlapping chunks for long
   messages); embedded with `BAAI/bge-small-en-v1.5` (ONNX, 384-dim);
   BM25-indexed.
2. Retrieve: top-200 dense + top-200 BM25 candidates → RRF fusion (k=60) →
   cross-encoder (`ms-marco-MiniLM-L-6-v2`, ONNX) over top 50 → relevance
   gate (threshold 0.0, calibrated on a synthetic set; see README) →
   tiny recency tie-break → score = sigmoid(logit − threshold).
3. Concurrency: per-user locks; 200 is returned only after the write is
   searchable; verified with 64 concurrent writers.
4. Memory management (v0.2.0): structural contradiction detection with
   atomic supersede linking (precision/recall 1.000/1.000 on a 22-pair
   synthetic set); temporal parsing + window boost; changeable-fact
   recency; deterministic gated two-hop retrieval; per-session
   extractive consolidation.

## Checklist before going public

- [ ] Corey reviews the repository contents.
- [ ] Confirm the submission form's expected Dockerfile contract
      (port, health path) against the live evaluation page.
- [ ] Push to a new public GitHub repository and submit the repo URL.
- [ ] Materials due **Oct 31, 2026 23:59**; evaluation queue closes
      **Nov 4, 2026 23:59** (Beijing time presumed).
