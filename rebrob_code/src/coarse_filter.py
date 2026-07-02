"""
coarse_filter.py — Fast rule-based elimination for the Redrob Candidate Ranking System.

Reduces ~100K candidates to ~5-10K by eliminating clearly irrelevant profiles.
This filter is intentionally LENIENT — edge cases pass through; only obvious
mismatches get eliminated. Downstream scoring handles nuance.

Rules applied:
  1. Experience band:  < 2 years or > 20 years → eliminate
  2. Title relevance:  Irrelevant title AND no career positive keywords → eliminate
                       (career-pivot escape hatch keeps pivoting candidates)
  3. Minimum technical signal: At least 1 matching skill OR ≥2 career positive keywords

NOT filtered here (by design):
  - Location / country  — JD says international candidates considered case-by-case
  - Services-only gate  — handled separately in feature_engineer.py
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Config import — project root is one level above src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    EXPERIENCE_HARD_MIN,
    EXPERIENCE_HARD_MAX,
    IRRELEVANT_TITLES,
    CAREER_POSITIVE_KEYWORDS,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
)


# ---------------------------------------------------------------------------
# Pre-compute lowercased keyword sets once at module load (O(1) lookups)
# ---------------------------------------------------------------------------
_IRRELEVANT_TITLES_LOWER: set[str] = {t.lower() for t in IRRELEVANT_TITLES}
_CAREER_POS_KW_LOWER: set[str] = {kw.lower() for kw in CAREER_POSITIVE_KEYWORDS}
_ALL_RELEVANT_SKILLS_LOWER: set[str] = {
    s.lower() for s in (MUST_HAVE_SKILLS | NICE_TO_HAVE_SKILLS)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _text_contains_any(text: str, keywords: set[str]) -> bool:
    """Check if *lowercased* ``text`` contains any keyword from ``keywords``.

    Uses substring matching (``kw in text``) so that multi-word keywords like
    ``"search system"`` match naturally inside longer descriptions.

    Parameters
    ----------
    text:
        Free-text string to search (will be lowercased internally).
    keywords:
        Set of **already-lowercased** keyword strings.

    Returns
    -------
    bool
        ``True`` if at least one keyword is found inside ``text``.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _count_keyword_hits(text: str, keywords: set[str]) -> int:
    """Count how many distinct keywords from ``keywords`` appear in ``text``.

    Parameters
    ----------
    text:
        Free-text string (lowercased internally).
    keywords:
        Set of **already-lowercased** keyword strings.

    Returns
    -------
    int
        Number of distinct keywords found.
    """
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def _build_career_text(candidate: dict) -> str:
    """Concatenate all career descriptions + profile summary into one blob.

    Gives downstream checks a single string to search against instead of
    iterating career_history entries repeatedly.

    Parameters
    ----------
    candidate:
        Full candidate dict with ``profile.summary`` and ``career_history[].description``.

    Returns
    -------
    str
        Lowercased concatenation of summary + all career descriptions.
    """
    parts: list[str] = []

    # Profile summary
    summary = candidate.get("profile", {}).get("summary", "")
    if summary:
        parts.append(summary)

    # Career history descriptions
    for entry in candidate.get("career_history", []):
        desc = entry.get("description", "")
        if desc:
            parts.append(desc)

    return " ".join(parts).lower()


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------
def passes_coarse_filter(candidate: dict) -> tuple[bool, str]:
    """Determine whether a candidate passes the coarse-grained elimination filter.

    This function is designed to run **fast** over 100K+ candidates (pure Python
    string ops, pre-computed sets, no I/O).  It is intentionally lenient — when
    in doubt, the candidate passes.

    Parameters
    ----------
    candidate:
        A single candidate dict matching the schema in ``sample_candidates.json``.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` if the candidate passes.
        ``(False, reason)`` if eliminated, where *reason* is a human-readable
        string explaining the rejection.

    Examples
    --------
    >>> cand = {"profile": {"years_of_experience": 0.5, ...}, ...}
    >>> passes_coarse_filter(cand)
    (False, 'experience_too_low: 0.5 years (min 2.0)')
    """
    profile = candidate.get("profile", {})

    # ------------------------------------------------------------------
    # Rule 1 — Experience band
    # ------------------------------------------------------------------
    yoe = profile.get("years_of_experience", 0.0)

    if yoe < EXPERIENCE_HARD_MIN:
        return (False, f"experience_too_low: {yoe} years (min {EXPERIENCE_HARD_MIN})")

    if yoe > EXPERIENCE_HARD_MAX:
        return (
            False,
            f"experience_too_high: {yoe} years (max {EXPERIENCE_HARD_MAX})",
        )

    # ------------------------------------------------------------------
    # Pre-compute career text blob (used by rules 2 & 3)
    # ------------------------------------------------------------------
    career_text = _build_career_text(candidate)

    # ------------------------------------------------------------------
    # Rule 2 — Title relevance (with career-pivot escape hatch)
    # ------------------------------------------------------------------
    current_title = profile.get("current_title", "").lower().strip()

    if current_title and current_title in _IRRELEVANT_TITLES_LOWER:
        # Title is irrelevant — but maybe they're pivoting into AI/ML/search.
        # Check career descriptions + summary for positive technical keywords.
        if not _text_contains_any(career_text, _CAREER_POS_KW_LOWER):
            return (
                False,
                f"irrelevant_title_no_pivot: '{current_title}' with no "
                f"career positive keywords in descriptions/summary",
            )
        # else: title is irrelevant but career shows technical pivot → KEEP

    # ------------------------------------------------------------------
    # Rule 3 — Minimum technical signal
    # ------------------------------------------------------------------
    # Path A: At least 1 skill name overlaps with relevant skill sets
    skills = candidate.get("skills", [])
    has_matching_skill = any(
        skill.get("name", "").lower() in _ALL_RELEVANT_SKILLS_LOWER
        for skill in skills
    )

    if not has_matching_skill:
        # Path B: Career text contains ≥ 2 career positive keywords
        positive_hit_count = _count_keyword_hits(career_text, _CAREER_POS_KW_LOWER)
        if positive_hit_count < 2:
            return (
                False,
                f"no_technical_signal: 0 matching skills and only "
                f"{positive_hit_count} career positive keyword(s) (need ≥2)",
            )

    # ------------------------------------------------------------------
    # All rules passed
    # ------------------------------------------------------------------
    return (True, "")
