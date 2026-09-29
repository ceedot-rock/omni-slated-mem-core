"""Threshold calibration for Omni Slated Mem Core.

Builds a synthetic memory (~48 docs, mixed topics), then runs relevant and
irrelevant queries directly against the retrieval pipeline and reports the
cross-encoder logit distributions so the relevance gate threshold can be set
from data rather than guessed.

Usage: python scripts/calibrate.py
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from app import config
from app.models_local import LocalEmbedder, LocalReranker
from app.store import Store, chunk_text, iso_from_ms
from app.retrieval import _rrf_fuse, _recency_boost
import numpy as np

DOCS = [
    # personal facts
    "My dog's name is Biscuit and he loves peanut butter.",
    "I was born in 1988 in Lisbon, Portugal.",
    "My sister Ana is a marine biologist studying octopus cognition.",
    "I am allergic to penicillin; my doctor prescribed azithromycin instead.",
    # preferences / personalization
    "I hate cilantro; it tastes like soap to me.",
    "I prefer window seats on flights and always board early.",
    "My favorite cuisine is Ethiopian, especially injera with lentils.",
    # work / rules / process
    "The launch code for project Zephyr is ZEPH-4417.",
    "Expense reports over $500 need VP approval before submission.",
    "The deployment checklist requires a database snapshot first, then migrations.",
    "Code reviews must be requested in the #eng-review channel, not by DM.",
    # temporal / events
    "My dentist appointment is on the third Thursday of next month.",
    "We moved to the new office on March 3rd, 2024.",
    "The product launch is scheduled for October 14, 2026.",
    "Last summer we hiked the Tour du Mont Blanc over eight days.",
    # multi-hop-ish (related facts across docs)
    "Dr. Chen's lab published the quantum error correction paper in Nature.",
    "The Nature paper used a surface code with distance 7.",
    "Surface code distance 7 requires 97 physical qubits per logical qubit.",
    # general knowledge-ish memories
    "The capital of Burkina Faso is Ouagadougou.",
    "Photosynthesis converts CO2 and water into glucose using sunlight.",
    "The Eiffel Tower was completed in 1889 for the World's Fair.",
    "Mitochondria are the powerhouse of the cell, producing ATP.",
    # noisy / similar-but-different (distractors)
    "My neighbor's dog is named Cookie and he loves cheese.",
    "I was born in 1998, not 1988, according to my passport.",
    "The launch code for project Boreal is BOR-9921.",
    "My brother likes cilantro in his tacos.",
]

RELEVANT = [
    ("What is my dog's name?", "Biscuit"),
    ("Where was I born?", "Lisbon"),
    ("What am I allergic to?", "penicillin"),
    ("What is the Zephyr launch code?", "ZEPH-4417"),
    ("Who needs to approve big expense reports?", "VP approval"),
    ("When did we move offices?", "March 3rd, 2024"),
    ("How many physical qubits for the surface code?", "97 physical qubits"),
    ("What is the capital of Burkina Faso?", "Ouagadougou"),
    ("What food do I hate?", "cilantro"),
    ("What did Dr. Chen's lab publish?", "quantum error correction"),
    ("What must happen before migrations?", "database snapshot"),
    ("Where should code reviews be requested?", "#eng-review"),
]

IRRELEVANT = [
    "quantum chromodynamics lattice gauge beta function two-loop",
    "symplectic geometry of K3 surfaces and mirror symmetry",
    "recipes for sourdough starter hydration schedules",
    "2026 FIFA World Cup qualifying standings group C",
    "troubleshooting a leaking dishwasher drain pump",
    "the mating habits of the Patagonian mara",
    "ancient Sumerian cuneiform tablet transliteration",
    "best practices for bonsai juniper wiring",
    "history of the Hanseatic League trade routes",
    "how to change a timing belt on a 2011 Honda Civic",
    "Mongolian throat singing techniques for beginners",
    "tax implications of ISOs vs NSOs in California",
]


def main() -> None:
    embedder = LocalEmbedder(config.EMBED_MODEL_DIR, config.EMBED_MODEL_FILE)
    reranker = LocalReranker(config.RERANK_MODEL_DIR, config.RERANK_MODEL_FILE)
    store = Store()
    idx = store.get_or_create("cal")
    chunks = []
    for i, d in enumerate(DOCS):
        chunks.append((f"cal#{i}.0", d, "s1", "user", 1720000000000, iso_from_ms(1720000000000)))
    vecs = embedder.embed([c[1] for c in chunks])
    idx.add("cal", chunks, vecs)

    def top_logit(query: str) -> tuple[float, str]:
        with idx.lock:
            mat = idx.embedding_matrix()
            q = embedder.embed([query])[0].astype(np.float64)
            dense = mat.astype(np.float64) @ q
            d_order = np.argsort(-dense, kind="stable")[: config.DENSE_POOL]
            fused = _rrf_fuse(d_order, np.array([], dtype=int))
            order = sorted(fused, key=lambda i: fused[i], reverse=True)[: config.RERANK_POOL]
            docs = [idx.docs[i].content for i in order]
            logits = [float(x) for x in reranker.rerank(query, docs)]
            best = int(np.argmax(logits))
            return logits[best], docs[best][:60]

    print("=== relevant queries (want HIGH logit, correct doc) ===")
    rel_scores = []
    for q, want in RELEVANT:
        s, doc = top_logit(q)
        rel_scores.append(s)
        ok = want.lower() in doc.lower()
        print(f"{s:7.2f} {'OK ' if ok else 'MISS'} {q} -> {doc}")
    print("=== irrelevant queries (want LOW logit) ===")
    irr_scores = []
    for q in IRRELEVANT:
        s, doc = top_logit(q)
        irr_scores.append(s)
        print(f"{s:7.2f} {q[:50]}")
    rel = np.array(rel_scores)
    irr = np.array(irr_scores)
    print(f"\nrelevant: min {rel.min():.2f}  p10 {np.percentile(rel,10):.2f}  med {np.median(rel):.2f}")
    print(f"irrelevant: max {irr.max():.2f}  p90 {np.percentile(irr,90):.2f}  med {np.median(irr):.2f}")
    for t in [0.0, 1.0, 2.0, 3.0, 4.0]:
        rec = (rel >= t).mean()
        rej = (irr < t).mean()
        print(f"threshold {t:.1f}: relevant kept {rec:.0%}, irrelevant rejected {rej:.0%}")


if __name__ == "__main__":
    main()
