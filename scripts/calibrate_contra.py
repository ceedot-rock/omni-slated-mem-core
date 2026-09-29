"""Calibrate the contradiction / supersede detector on synthetic pairs.

Builds (new, old) text pairs with known labels, runs the structural
governance detector, and reports precision/recall. Exit 0 only if both
are 1.0 — the detector must be exact on this set before it ships.

The detector is deliberately model-free: the vendored cross-encoder was
evaluated as a topic-confirmation signal and vetoed true contradictions
(ms-marco scores QA relevance, not topic sameness), so no model loads here.

Usage:  .venv/bin/python scripts/calibrate_contra.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import governance as gov
from app.store import Doc, tokenize

# (new_text, old_text, should_contradict)
PAIRS: list[tuple[str, str, bool]] = [
    # --- true contradictions: same anchor, conflicting value ---
    ("I moved to Austin, Texas.", "I live in Philadelphia.", True),
    ("My favorite food is sushi now.", "My favorite food is pizza.", True),
    ("My dog's name is now Waffles.", "My dog's name is Biscuit.", True),
    ("I quit Acme Corp and now work at Globex.", "I work at Acme Corp.", True),
    ("I hate cilantro.", "I like cilantro.", True),
    ("I don't love hiking anymore.", "I love hiking.", True),
    ("My address is 5th Avenue, New York.", "I live in Boston.", True),
    ("My favorite color is red.", "My favorite color is blue.", True),
    ("I work for Umbrella Corp now.", "I work for Initech.", True),
    ("I live in Austin.", "I live in Philadelphia.", True),
    # --- non-contradictions: must NOT fire ---
    ("My dog loves peanut butter.", "My dog's name is Biscuit.", False),
    ("I live in Philadelphia, Pennsylvania.", "I live in Philadelphia.", False),
    ("My cat is small.", "My dog is big.", False),
    ("I like sushi.", "I like pizza.", False),
    ("The meeting is on Monday.", "The meeting is on Monday.", False),
    ("Philadelphia has great cheesesteaks.", "I live in Philadelphia.", False),
    ("My favorite movie is Dune.", "My favorite food is pizza.", False),
    ("My brother works at Acme.", "I work at Acme.", False),
    ("I was born in Lisbon.", "I was born in 1988.", False),
    ("Cilantro tastes like soap.", "I hate cilantro.", False),
    ("I live in Philadelphia.", "I visited Austin last year.", False),
    ("My favorite food is pizza.", "I ate pizza yesterday.", False),
]


def make_doc(i: int, content: str) -> Doc:
    frames = gov.extract_frames(content)
    return Doc(
        id=f"t{i}", content=content, user_id="cal", session_id="s",
        role="user", ts_ms=0, created_at="",
        tokens=tokenize(content), length=len(tokenize(content)),
        anchors=gov.extract_anchors(content, frames),
        frame_anchors=frozenset(f.anchor for f in frames),
        frames=frames,
    )


def main() -> int:
    tp = fp = tn = fn = 0
    for i, (new_t, old_t, expected) in enumerate(PAIRS):
        new_doc = make_doc(2 * i, new_t)
        old_doc = make_doc(2 * i + 1, old_t)
        got = bool(gov.detect_contradictions(new_doc, [(1, old_doc)]))
        mark = "ok" if got == expected else "MISS"
        print(f"[{mark}] expected={expected!s:5} got={got!s:5} "
              f"new={new_t[:44]!r:48} old={old_t[:36]!r}")
        if expected and got:
            tp += 1
        elif expected:
            fn += 1
        elif got:
            fp += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    print(f"\nprecision={precision:.3f} recall={recall:.3f} "
          f"(tp={tp} fp={fp} tn={tn} fn={fn})")
    if precision == 1.0 and recall == 1.0:
        print("CONTRADICTION DETECTOR CALIBRATED: exact on synthetic set")
        return 0
    print("CALIBRATION FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
