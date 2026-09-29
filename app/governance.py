"""Memory governance for Omni Slated Mem Core v0.2.0.

Zero-LLM, fully deterministic memory management layered over the retrieval
core:

  * contradiction / supersede tracking — structural fact frames
    (anchor / relation / value) plus explicit change markers; the vendored
    cross-encoder confirms topic sameness so anchor collisions across
    different topics do not false-positive.
  * temporal understanding — deterministic time-expression parsing turning
    queries into [start, end) millisecond windows.
  * history-vs-current intent detection ("where did I used to live").
  * volatile-fact detection (changeable attributes: location, employer,
    favorites, attitudes) for intelligent recency.
  * extractive per-session consolidation — deduped fact lists maintained
    per session and indexed as compact documents.

No randomness anywhere: insertion-ordered iteration, stable sorts, fixed
regex evaluation order. Same inputs -> same outputs, byte for byte.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .store import tokenize

_DAY_MS = 86_400_000


# ---------------------------------------------------------------------------
# Fact frames
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    anchor: str       # normalized anchor key, e.g. "location", "my dog name"
    relation: str     # relation label, e.g. "live-in", "named", "like"
    value: str        # normalized value
    exclusive: bool   # different values for this anchor are mutually exclusive


def _nv(s: str) -> str:
    """Normalize a value: lowercase, collapse whitespace, strip punctuation."""
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .,;:!?\"'()")


_TRAILING_MODIFIERS = ("anymore", "now", "again", "today", "lately",
                        "currently", "still", "these days", "nowadays")


def _strip_modifiers(value: str) -> str:
    """Drop temporal/adverbial modifiers so "hiking anymore" and "hiking"
    (or "now waffles" and "waffles") share an anchor. Deterministic."""
    v = value
    if v.startswith("now "):
        v = v[4:]
    changed = True
    while changed:
        changed = False
        for mod in _TRAILING_MODIFIERS:
            if v.endswith(" " + mod):
                v = v[: -len(mod) - 1]
                changed = True
    return v or value


# (pattern, anchor_template, relation_template, value_group, exclusive)
# Templates reference match groups 0-indexed: {0} is group 1, {1} is group 2.
# A None anchor means an attitude frame whose anchor is derived from value.
_FRAME_PATTERNS: list[tuple[str, str | None, str | None, int, bool]] = [
    (r"\bi moved to ([a-z][^.,;!?]{1,50})", "location", "live-in", 1, True),
    (r"\bi now live in ([a-z][^.,;!?]{1,50})", "location", "live-in", 1, True),
    (r"\bi live in ([a-z][^.,;!?]{1,50})", "location", "live-in", 1, True),
    (r"\bmy (?:current |present )?address is ([a-z0-9][^.,;!?]{1,50})",
     "location", "address", 1, True),
    (r"\bi work at ([a-z][^.,;!?]{1,50})", "employer", "work-at", 1, True),
    (r"\bi work for ([a-z][^.,;!?]{1,50})", "employer", "work-for", 1, True),
    (r"\bnow (?:i )?(?:work|employed) (?:at|for|with) ([a-z0-9][^.,;!?]{1,50})",
     "employer", "work-at", 1, True),
    (r"\bcurrently (?:i )?(?:work|employed) (?:at|for|with) ([a-z0-9][^.,;!?]{1,50})",
     "employer", "work-at", 1, True),
    (r"\bmy (?:favourite|favorite) (\w+) is ([a-z][^.,;!?]{1,50})",
     "favorite {0}", "is", 2, True),
    (r"\bmy (\w+)'s name is ([a-z][^.,;!?]{1,50})", "my {0} name", "named", 2, True),
    (r"\bher name is ([a-z][^.,;!?]{1,50})", "her name", "named", 1, True),
    (r"\bhis name is ([a-z][^.,;!?]{1,50})", "his name", "named", 1, True),
    # Attitude frames: negated forms first (finditer scans are independent,
    # but this documents precedence).
    (r"\bi (?:do not|don't|do n't) (?:like|love) ([a-z][^.,;!?]{1,40})",
     None, "neg-like", 1, False),
    (r"\bi (?:do not|don't|do n't) (?:hate|dislike) ([a-z][^.,;!?]{1,40})",
     None, "neg-hate", 1, False),
    (r"\bi (like|love) ([a-z][^.,;!?]{1,40})", None, "{0}", 2, False),
    (r"\bi (hate|dislike) ([a-z][^.,;!?]{1,40})", None, "{0}", 2, False),
]


def extract_frames(text: str) -> list[Frame]:
    """Extract structural fact frames from text (deterministic)."""
    low = text.lower()
    frames: list[Frame] = []
    for pat, anchor_tpl, rel_tpl, vg, exclusive in _FRAME_PATTERNS:
        for m in re.finditer(pat, low):
            groups = [_nv(g or "") for g in m.groups()]
            value = _strip_modifiers(groups[vg - 1])
            if not value:
                continue
            if anchor_tpl is None:
                anchor = f"attitude:{value}"
            elif "{" in anchor_tpl:
                anchor = _nv(anchor_tpl.format(*groups))
            else:
                anchor = anchor_tpl
            if rel_tpl and "{" in rel_tpl:
                relation = _nv(rel_tpl.format(*groups))
            else:
                relation = rel_tpl or ""
            frames.append(Frame(anchor=anchor, relation=relation,
                                value=value, exclusive=exclusive))
    return frames


# ---------------------------------------------------------------------------
# Anchors (entity signatures for the pre-filter index)
# ---------------------------------------------------------------------------

_PROPER_STOP = frozenset({
    "i", "my", "the", "a", "an", "and", "but", "or", "in", "on", "at", "to",
    "for", "of", "with", "is", "are", "was", "were", "he", "she", "it",
    "they", "we", "you", "this", "that", "these", "those", "his", "her",
    "its", "our", "your", "their", "noted", "meeting", "notes",
    # question words are not entities
    "who", "what", "where", "when", "why", "how", "which", "whom", "whose",
})


def extract_proper_nouns(text: str) -> list[str]:
    """Capitalized words (len>2) not in the stoplist, order-preserving dedupe."""
    out: list[str] = []
    for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", text):
        w = m.group(0)
        if w.lower() in _PROPER_STOP:
            continue
        out.append(w.lower())
    return list(dict.fromkeys(out))


def extract_anchors(text: str, frames: list[Frame] | None = None) -> frozenset[str]:
    """All anchors: frame anchor keys plus proper nouns."""
    frames = frames if frames is not None else extract_frames(text)
    return frozenset([f.anchor for f in frames] + extract_proper_nouns(text))


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

# Explicit change/supersede language in the NEW chunk. Any one of these plus
# a shared anchor with different values marks a contradiction even for
# non-exclusive relations.
_CHANGE_RES = [
    r"\bno longer\b",
    r"\bnot any ?more\b",
    r"\banymore\b",
    r"\bused to\b",
    r"\bchanged\b",
    r"\bquit\b",
    r"\binstead of\b",
    r"\bmoved to\b",
    r"\brenamed to\b",
    r"\bswitched to\b",
    r"\bnow (?:live|work|called|named|love|like|hate)\b",
]
_CHANGE_RE = re.compile("|".join(_CHANGE_RES))

_POS_REL = {"like", "love", "neg-hate"}
_NEG_REL = {"hate", "dislike", "neg-like"}


def _polarity(relation: str) -> int:
    if relation in _POS_REL:
        return 1
    if relation in _NEG_REL:
        return -1
    return 0


def _values_compatible(v1: str, v2: str) -> bool:
    """True when one value refines the other (token-subset), e.g.
    'philadelphia' vs 'philadelphia, pennsylvania' — same place, not a
    contradiction."""
    t1, t2 = set(v1.split()), set(v2.split())
    return bool(t1) and bool(t2) and (t1 <= t2 or t2 <= t1)


def frames_contradict(new: Frame, old: Frame) -> bool:
    """Structural contradiction test between two frames (no model)."""
    if new.anchor != old.anchor:
        return False
    if new.anchor.startswith("attitude:"):
        # Attitude frames: the value is the object ("cilantro"), the stance
        # is the relation — same object with opposite polarity contradicts.
        pn, po = _polarity(new.relation), _polarity(old.relation)
        return pn != 0 and po != 0 and pn != po
    if new.value == old.value:
        return False
    if not (new.exclusive and old.exclusive):
        return False
    # Exclusive anchor, different values: a refinement is not a conflict.
    return not _values_compatible(new.value, old.value)


def detect_contradictions(new_doc, candidate_docs) -> list:
    """Return the indices of candidate_docs contradicted by new_doc.

    candidate_docs are (index, doc) pairs. Purely structural: fact frames
    decide (same anchor + conflicting value on an exclusive relation, or an
    attitude polarity flip, or explicit change language with a shared
    anchor). Deliberately model-free — the vendored cross-encoder was
    evaluated as a topic-confirmation signal and vetoed true contradictions
    (ms-marco scores QA relevance, not topic sameness: "I moved to Austin"
    vs "I live in Philadelphia" scores -7.94), so it is not used here.
    """
    new_low = new_doc.content.lower()
    has_marker = bool(_CHANGE_RE.search(new_low))
    hits: list[int] = []
    for idx, old in candidate_docs:
        if old.superseded_by is not None or old.is_consolidated:
            continue
        contradicted = False
        for fn in new_doc.frames:
            for fo in old.frames:
                if frames_contradict(fn, fo):
                    contradicted = True
                    break
                if (has_marker and fn.anchor == fo.anchor
                        and fn.value != fo.value):
                    contradicted = True
                    break
            if contradicted:
                break
        if contradicted:
            hits.append(idx)
    return hits


# ---------------------------------------------------------------------------
# Temporal understanding
# ---------------------------------------------------------------------------

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def parse_time_window(query: str, now_ms: int) -> tuple[int | None, int | None]:
    """Parse a time expression in the query into a [start, end) ms window.

    Returns (None, None) when the query carries no time expression.
    Deterministic given now_ms. Checks run in a fixed order: explicit
    dates first, then named months/years, then relative expressions.
    """
    q = query.lower()
    now = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # ISO date: 2024-06-15
    m = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", q)
    if m:
        try:
            day = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           tzinfo=timezone.utc)
        except ValueError:
            return None, None
        return _ms(day), _ms(day + timedelta(days=1))

    # "June 15, 2024" / "15 June 2024"
    m = re.search(rf"\b({_MONTH_RE})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", q)
    order = "mdy"
    if not m:
        m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH_RE})\s+(\d{{4}})\b", q)
        order = "dmy"
    if m:
        if order == "mdy":
            month, day_n, year = _MONTHS[m.group(1)], int(m.group(2)), int(m.group(3))
        else:
            day_n, month, year = int(m.group(1)), _MONTHS[m.group(2)], int(m.group(3))
        try:
            day = datetime(year, month, day_n, tzinfo=timezone.utc)
        except ValueError:
            return None, None
        return _ms(day), _ms(day + timedelta(days=1))

    # "June 2024" / "in June 2024"
    m = re.search(rf"\b({_MONTH_RE})\s+(\d{{4}})\b", q)
    if m:
        month, year = _MONTHS[m.group(1)], int(m.group(2))
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
        return _ms(start), _ms(end)

    # bare year: "in 2024" / "2024"
    m = re.search(r"\b((?:19|20)\d{2})\b", q)
    if m:
        year = int(m.group(1))
        return _ms(datetime(year, 1, 1, tzinfo=timezone.utc)), \
               _ms(datetime(year + 1, 1, 1, tzinfo=timezone.utc))

    # bare month name: most recent such month at or before now
    m = re.search(rf"\b({_MONTH_RE})\b", q)
    if m:
        month = _MONTHS[m.group(1)]
        year = now.year if month <= now.month else now.year - 1
        start = datetime(year, month, 1, tzinfo=timezone.utc)
        end = datetime(year + (month == 12), (month % 12) + 1, 1, tzinfo=timezone.utc)
        return _ms(start), _ms(end)

    # relative days
    if re.search(r"\byesterday\b", q):
        d = midnight - timedelta(days=1)
        return _ms(d), _ms(d + timedelta(days=1))
    if re.search(r"\blast night\b", q):
        d = midnight - timedelta(days=1)
        return _ms(d + timedelta(hours=18)), _ms(midnight + timedelta(hours=6))
    if re.search(r"\btoday\b", q):
        return _ms(midnight), now_ms
    if re.search(r"\bthis morning\b", q):
        return _ms(midnight), _ms(midnight + timedelta(hours=12))
    if re.search(r"\bthis afternoon\b", q):
        return _ms(midnight + timedelta(hours=12)), _ms(midnight + timedelta(hours=18))
    if re.search(r"\bthis evening\b", q):
        return _ms(midnight + timedelta(hours=18)), now_ms

    # relative weeks / months (rolling)
    if re.search(r"\blast week\b", q):
        return now_ms - 14 * _DAY_MS, now_ms - 7 * _DAY_MS
    if re.search(r"\bthis week\b", q):
        return now_ms - 7 * _DAY_MS, now_ms
    if re.search(r"\blast month\b", q):
        return now_ms - 60 * _DAY_MS, now_ms - 30 * _DAY_MS
    if re.search(r"\bthis month\b", q):
        return now_ms - 30 * _DAY_MS, now_ms

    m = re.search(r"\b(\d+)\s+days?\s+ago\b", q)
    if m:
        d = midnight - timedelta(days=int(m.group(1)))
        return _ms(d), _ms(d + timedelta(days=1))
    m = re.search(r"\b(\d+)\s+weeks?\s+ago\b", q)
    if m:
        n = int(m.group(1))
        return now_ms - (n + 1) * 7 * _DAY_MS, now_ms - n * 7 * _DAY_MS
    m = re.search(r"\b(\d+)\s+months?\s+ago\b", q)
    if m:
        n = int(m.group(1))
        return now_ms - (n + 1) * 30 * _DAY_MS, now_ms - n * 30 * _DAY_MS

    # "last Monday" — most recent such weekday strictly before today
    m = re.search(r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", q)
    if m:
        target = _WEEKDAYS[m.group(1)]
        delta = (now.weekday() - target) % 7 or 7
        d = midnight - timedelta(days=delta)
        return _ms(d), _ms(d + timedelta(days=1))

    return None, None


# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_HISTORY_RES = [
    r"\bused to\b",
    r"\bdid\b.{0,24}?\buse to\b",
    r"\bpreviously\b",
    r"\bformerly\b",
    r"\bformer\b",
    r"\bwhat was my (?:old|previous|former)\b",
    r"\bwhere did i (?:used to |previously )?(?:live|work)\b",
    r"\bmy (?:old|previous|former)\b",
    r"\bhistory\b",
]
_HISTORY_RE = re.compile("|".join(_HISTORY_RES))


def is_history_query(query: str) -> bool:
    """True when the query explicitly asks about past/superseded state."""
    return bool(_HISTORY_RE.search(query.lower()))


_VOLATILE_Q_RES = [
    r"\bwhere do i live\b",
    r"\bwhere am i living\b",
    r"\bmy (?:current |present )?(?:address|location)\b",
    r"\bwhat is my favorite\b",
    r"\bmy (?:current |present )?job\b",
    r"\bwhere do i work\b",
    r"\bwho do i work for\b",
]
_VOLATILE_Q_RE = re.compile("|".join(_VOLATILE_Q_RES))

_VOLATILE_ANCHORS = frozenset({"location", "employer"})


def is_volatile_anchor(anchor: str) -> bool:
    return (anchor in _VOLATILE_ANCHORS
            or anchor.startswith("favorite ")
            or anchor.startswith("attitude:"))


def is_volatile_query(query: str) -> bool:
    """True when the query asks about a changeable attribute in the present
    tense — the newest value wins among same-anchor candidates."""
    low = query.lower()
    if _VOLATILE_Q_RE.search(low):
        return True
    return any(is_volatile_anchor(f.anchor) for f in extract_frames(query))


# ---------------------------------------------------------------------------
# Extractive consolidation
# ---------------------------------------------------------------------------

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_FACT_CUE = re.compile(
    r"\b(is|are|was|were|my|have|has|had|live|lives|lived|work|works|"
    r"like|likes|love|loves|hate|named|called|born)\b"
)


def extract_fact_sentences(text: str) -> list[str]:
    """Extractive fact candidates: 4-40 word non-question sentences with a
    fact cue. Purely local, deterministic."""
    out: list[str] = []
    for sent in _SENT_SPLIT.split(text):
        s = sent.strip().strip("\"'“”‘’")
        words = s.split()
        if not (4 <= len(words) <= 40):
            continue
        if s.endswith("?"):
            continue
        if not _FACT_CUE.search(s.lower()):
            continue
        out.append(s)
    return out


def dedupe_facts(facts: list[str], threshold: float = 0.85) -> list[str]:
    """Near-duplicate removal on normalized token Jaccard; keeps the later
    fact on a match. Deterministic."""
    kept: list[str] = []
    kept_toks: list[set[str]] = []
    for f in facts:
        ft = set(tokenize(f))
        replaced = False
        for i, kt in enumerate(kept_toks):
            union = ft | kt
            if union and len(ft & kt) / len(union) >= threshold:
                kept[i] = f
                kept_toks[i] = ft
                replaced = True
                break
        if not replaced:
            kept.append(f)
            kept_toks.append(ft)
    return kept
