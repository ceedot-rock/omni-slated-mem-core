"""Local ONNX inference for embeddings and cross-encoder reranking.

Loads the vendored models straight from disk (no network, no model hub).
Tokenizer handling replicates fastembed's local pipeline exactly:
truncation to the tokenizer config's max length, special tokens registered
from special_tokens_map.json, padding normalized to batch-longest.
Embedding post-processing replicates fastembed's OnnxTextEmbedding:
first-token pooling + L2 normalization (eps 1e-12).
Reranker replicates fastembed's OnnxTextCrossEncoder: pair tokenization,
raw logit output (outputs[0][:, 0]).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import AddedToken, Tokenizer


def _resolve_max_context(tokenizer_config: dict) -> int:
    def _valid(v):
        # fastembed treats absurd values (>100k) and non-ints as invalid
        return v if isinstance(v, int) and 0 < v <= 100000 else None

    cands = [
        c
        for c in (
            _valid(tokenizer_config.get("model_max_length")),
            _valid(tokenizer_config.get("max_length")),
        )
        if c is not None
    ]
    if not cands:
        raise ValueError("Could not determine maximum context length from tokenizer_config.json")
    return min(cands)


def load_local_tokenizer(model_dir: Path) -> Tokenizer:
    """Replicates fastembed.common.preprocessor_utils.load_tokenizer (local-only)."""
    tok_path = model_dir / "tokenizer.json"
    cfg_path = model_dir / "tokenizer_config.json"
    if not tok_path.exists():
        raise ValueError(f"tokenizer.json missing in {model_dir}")
    if not cfg_path.exists():
        raise ValueError(f"tokenizer_config.json missing in {model_dir}")

    with open(cfg_path) as f:
        tokenizer_config = json.load(f)
    max_context = _resolve_max_context(tokenizer_config)

    config_path = model_dir / "config.json"
    config: dict = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    tokenizer = Tokenizer.from_file(str(tok_path))
    tokenizer.enable_truncation(max_length=max_context)

    # special tokens (registered before padding is resolved)
    stm_path = model_dir / "special_tokens_map.json"
    if stm_path.exists():
        with open(stm_path) as f:
            tokens_map = json.load(f)
        for value in tokens_map.values():
            values = value if isinstance(value, list) else [value]
            for token in values:
                if isinstance(token, str):
                    tokenizer.add_special_tokens([token])
                elif isinstance(token, dict):
                    tokenizer.add_special_tokens([AddedToken(**token)])

    padding = tokenizer.padding or {}
    pad_token = padding.get("pad_token", tokenizer_config.get("pad_token"))
    if pad_token is None:
        raise ValueError(f"Could not find a pad token for {model_dir}")
    pad_id = padding.get("pad_id", config.get("pad_token_id"))
    if pad_id is None:
        pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise ValueError(f"Could not resolve an id for pad token {pad_token!r}")
    tokenizer.enable_padding(
        direction=padding.get("direction", "right"),
        pad_id=pad_id,
        pad_token=pad_token,
        pad_type_id=padding.get("pad_type_id", 0),
        pad_to_multiple_of=padding.get("pad_to_multiple_of"),
        length=None,
    )
    return tokenizer


class _OnnxBase:
    def __init__(self, model_dir: str, model_file: str):
        d = Path(model_dir)
        self.tokenizer = load_local_tokenizer(d)
        opts = ort.SessionOptions()
        # Keep each session single-threaded internally; parallelism comes from
        # request-level threads. Sessions are used under a lock regardless.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(d / model_file),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self._lock = threading.Lock()


class LocalEmbedder(_OnnxBase):
    """BAAI/bge-small-en-v1.5 via ONNX. embed() -> L2-normalized float32 vectors."""

    def embed(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            feed: dict = {"input_ids": ids}
            if "attention_mask" in self.input_names:
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in enc], dtype=np.int64
                )
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.zeros_like(ids)
            with self._lock:
                raw = self.session.run(None, feed)[0]
            if raw.ndim == 3:  # (batch, seq, dim) -> first-token pooling
                raw = raw[:, 0]
            norm = np.linalg.norm(raw, axis=1, keepdims=True)
            out.append(raw / np.maximum(norm, 1e-12))
        return np.vstack(out).astype(np.float32)


class LocalReranker(_OnnxBase):
    """ms-marco-MiniLM-L-6-v2 cross-encoder via ONNX. rerank() -> raw logits."""

    def rerank(self, query: str, docs: list[str], batch_size: int = 32) -> np.ndarray:
        scores: list[np.ndarray] = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            enc = self.tokenizer.encode_batch([(query, d) for d in batch])
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            feed: dict = {"input_ids": ids}
            if "attention_mask" in self.input_names:
                feed["attention_mask"] = np.array(
                    [e.attention_mask for e in enc], dtype=np.int64
                )
            if "token_type_ids" in self.input_names:
                feed["token_type_ids"] = np.array(
                    [e.type_ids for e in enc], dtype=np.int64
                )
            with self._lock:
                raw = self.session.run(None, feed)[0]
            scores.append(raw[:, 0].astype(np.float64))
        return np.concatenate(scores) if scores else np.array([], dtype=np.float64)
