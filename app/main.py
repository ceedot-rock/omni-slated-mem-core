"""Omni Slated Mem Core v0.2.0 — Agent Memory Challenge Cycle 2 entry
(textual track, academic division).

Three endpoints per the platform contract:
  POST /v1/memories/add     store messages (200 only after searchable)
  POST /v1/memories/search  retrieve evidence (most-relevant first)
  GET  /health              unauthenticated liveness

Zero-LLM local pipeline: local ONNX embeddings + BM25, RRF fusion,
local cross-encoder rerank, relevance gating — plus a memory-management
layer: contradiction/supersede tracking, temporal understanding, gated
multi-hop retrieval, and per-session consolidation. See README.
"""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import config
from .models_local import LocalEmbedder, LocalReranker
from .retrieval import search_index
from .store import Store, chunk_text, iso_from_ms, now_ms

store = Store()
embedder: LocalEmbedder | None = None
reranker: LocalReranker | None = None


# ---------------- request models ----------------

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[int] = None  # Unix ms


class AddRequest(BaseModel):
    request_id: str
    user_id: str
    session_id: str
    messages: list[Message] = Field(default_factory=list)


class SearchRequest(BaseModel):
    query: str
    user_id: str
    top_k: int = config.TOP_K_DEFAULT
    options: Optional[dict] = None


# ---------------- app lifecycle ----------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Room for the platform's 64 concurrent Add workers plus headroom.
    loop = asyncio.get_running_loop()
    loop.set_default_executor(
        ThreadPoolExecutor(max_workers=config.THREADPOOL_WORKERS)
    )
    global embedder, reranker
    embedder = LocalEmbedder(config.EMBED_MODEL_DIR, config.EMBED_MODEL_FILE)
    reranker = LocalReranker(config.RERANK_MODEL_DIR, config.RERANK_MODEL_FILE)
    # Warm up both sessions so the first request pays no init cost.
    embedder.embed(["warmup"])
    reranker.rerank("warmup", ["warmup"])
    yield


app = FastAPI(
    title=config.SYSTEM_NAME,
    version=config.SYSTEM_VERSION,
    lifespan=lifespan,
)


# ---------------- endpoints ----------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "system": config.SYSTEM_NAME,
        "version": config.SYSTEM_VERSION,
    }


@app.get("/service/about/endpoints")
def service_about_endpoints():
    """Machine-readable service description (outside the platform contract)."""
    return {
        "service": config.SYSTEM_NAME,
        "version": config.SYSTEM_VERSION,
        "contract_endpoints": [
            {
                "method": "POST",
                "path": "/v1/memories/add",
                "description": "Store messages; HTTP 200 only after searchable.",
            },
            {
                "method": "POST",
                "path": "/v1/memories/search",
                "description": "Retrieve evidence, most-relevant first; [] when nothing is relevant.",
            },
            {
                "method": "GET",
                "path": "/health",
                "description": "Unauthenticated liveness: {status, system, version}.",
            },
        ],
        "introspection": {
            "method": "GET",
            "path": "/service/about/endpoints",
        },
    }


@app.post("/v1/memories/add")
def memories_add(req: AddRequest):
    """Store messages. Returns 200 only after they are searchable."""
    assert embedder is not None
    idx = store.get_or_create(req.user_id)

    received_ms = now_ms()
    chunks: list[tuple[str, str, str, str, int, str]] = []
    for m_i, m in enumerate(req.messages):
        ts = m.timestamp if m.timestamp is not None else received_ms
        for c_i, piece in enumerate(chunk_text(m.content)):
            doc_id = f"{req.request_id}#{m_i}.{c_i}"
            chunks.append((doc_id, piece, req.session_id, m.role, ts, iso_from_ms(ts)))

    if chunks:
        vectors = embedder.embed([c[1] for c in chunks])
        idx.add(req.user_id, chunks, vectors, embedder=embedder)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "request_id": req.request_id,
            "user_id": req.user_id,
            "session_id": req.session_id,
        },
    )


@app.post("/v1/memories/search")
def memories_search(req: SearchRequest):
    """Retrieve evidence, most-relevant first. [] when nothing is relevant."""
    assert embedder is not None and reranker is not None
    top_k = max(1, min(req.top_k, config.TOP_K_MAX))
    idx = store.get(req.user_id)
    if idx is None:
        return {"data": []}
    data = search_index(idx, req.query, top_k, now_ms(), embedder, reranker)
    return {"data": data}


# Back-compat alias some harnesses use; harmless if unused.
@app.post("/v1/memories/search/")
def memories_search_slash(req: SearchRequest):
    return memories_search(req)
