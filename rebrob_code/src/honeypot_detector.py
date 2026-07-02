"""
honeypot_detector.py — Detect subtly impossible candidate profiles.

The challenge dataset contains ~80 "honeypot" candidates with fabricated
profiles. If >10% of our top-100 output are honeypots, we're disqualified.

Six independent checks flag specific impossibilities:
  1. Experience-tenure mismatch
  2. Skill inflation (too many "expert" skills with short tenure)
  3. Title-description semantic mismatch
  4. Assessment paradox (claims expert, scores terribly)
  5. Impossible timeline (dates that violate causality)
  6. Endorsement anomaly (inflated endorsements, zero substance)

A candidate is classified as a honeypot only when they accumulate
≥ HONEYPOT['flag_threshold'] (default 3) independent red flags.
"""

import sys
from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Config import — project root is one level above src/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    HONEYPOT,
    CAREER_POSITIVE_KEYWORDS,
    CAREER_NEGATIVE_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Pre-compute lowercased keyword sets once at module load
# ---------------------------------------------------------------------------
_CAREER_POS_KW_LOWER: set[str] = {kw.lower() for kw in CAREER_POSITIVE_KEYWORDS}
_CAREER_NEG_KW_LOWER: set[str] = {kw.lower() for kw in CAREER_NEGATIVE_KEYWORDS}

# Keywords indicating a technical title
_TECHNICAL_TITLE_MARKERS: set[str] = {"engineer", "developer", "scientist"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _text_contains_any(text: str, keywords: set[str]) -> bool:
    """Check if *lowercased* ``text`` contains any keyword from ``keywords``.

    Parameters
    ----------
    text:
        Free-text string to search (lowercased internally).
    keywords:
        Set of **already-lowercased** keyword strings.

    Returns
    -------
    bool
        ``True`` if at least one keyword is found inside ``text``.
    """
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)


def _is_technical_title(title: str) -> bool:
    """Return ``True`` if the title suggests a technical/engineering role.

    Parameters
    ----------
    title:
        Job title string (case-insensitive check).
    """
    title_lower = title.lower()
    return any(marker in title_lower for marker in _TECHNICAL_TITLE_MARKERS)


def _build_all_descriptions(candidate: dict) -> str:
    """Concatenate all career_history descriptions into one lowercased blob.

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str
        Lowercased concatenation of all career descriptions.
    """
    parts: list[str] = []
    for entry in candidate.get("career_history", []):
        desc = entry.get("description", "")
        if desc:
            parts.append(desc)
    return " ".join(parts).lower()


def _safe_parse_date(date_str: str | None) -> date | None:
    """Parse an ISO date string (YYYY-MM-DD) safely, returning ``None`` on failure.

    Parameters
    ----------
    date_str:
        Date string or ``None``.

    Returns
    -------
    date | None
        Parsed ``date`` object or ``None``.
    """
    if not date_str:
        return None
    try:
        return date.fromisoformat(date_str)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Individual honeypot checks
# ---------------------------------------------------------------------------

def _check_experience_tenure_mismatch(candidate: dict) -> str | None:
    """Flag 1: Claimed years_of_experience vastly exceeds sum of career durations.

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str | None
        A red-flag description string, or ``None`` if clean.
    """
    claimed_years = candidate.get("profile", {}).get("years_of_experience", 0.0)
    if claimed_years <= 0:
        return None

    total_career_months = sum(
        entry.get("duration_months", 0)
        for entry in candidate.get("career_history", [])
    )
    actual_years = total_career_months / 12.0

    if actual_years <= 0:
        # No career history at all but claims experience — suspicious
        return (
            f"experience_tenure_mismatch: claims {claimed_years:.1f}yr but "
            f"career_history sums to 0 months"
        )

    ratio = claimed_years / actual_years
    threshold = HONEYPOT["experience_tenure_ratio_threshold"]

    if ratio > threshold:
        return (
            f"experience_tenure_mismatch: claims {claimed_years:.1f}yr but "
            f"career durations sum to {actual_years:.1f}yr "
            f"(ratio {ratio:.2f} > {threshold})"
        )

    return None


def _check_skill_inflation(candidate: dict) -> str | None:
    """Flag 2: Too many 'expert' skills with very short duration.

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str | None
        A red-flag description string, or ``None`` if clean.
    """
    short_duration = HONEYPOT["short_duration_months"]
    max_allowed = HONEYPOT["max_expert_skills_short_duration"]

    expert_short_count = sum(
        1
        for skill in candidate.get("skills", [])
        if skill.get("proficiency", "").lower() == "expert"
        and skill.get("duration_months", 0) < short_duration
    )

    if expert_short_count > max_allowed:
        return (
            f"skill_inflation: {expert_short_count} expert-level skills with "
            f"<{short_duration} months tenure (max allowed {max_allowed})"
        )

    return None


def _check_title_description_mismatch(candidate: dict) -> str | None:
    """Flag 3: Title and career descriptions tell contradictory stories.

    Technical title + all-negative descriptions = flag.
    Non-technical title + all-positive descriptions = flag.

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str | None
        A red-flag description string, or ``None`` if clean.
    """
    current_title = candidate.get("profile", {}).get("current_title", "")
    if not current_title:
        return None

    career_text = _build_all_descriptions(candidate)
    if not career_text:
        return None

    has_positive = _text_contains_any(career_text, _CAREER_POS_KW_LOWER)
    has_negative = _text_contains_any(career_text, _CAREER_NEG_KW_LOWER)
    is_tech_title = _is_technical_title(current_title)

    # Technical title but descriptions are purely non-technical
    if is_tech_title and has_negative and not has_positive:
        return (
            f"title_description_mismatch: technical title '{current_title}' "
            f"but career descriptions contain only non-technical keywords"
        )

    # Non-technical title but descriptions are purely technical
    if not is_tech_title and has_positive and not has_negative:
        # Only flag if title is clearly non-technical (not ambiguous)
        title_lower = current_title.lower()
        non_tech_markers = {
            "manager", "accountant", "analyst", "coordinator",
            "support", "sales", "marketing", "hr", "recruiter",
            "writer", "designer", "teacher", "nurse", "lawyer",
        }
        if any(marker in title_lower for marker in non_tech_markers):
            return (
                f"title_description_mismatch: non-technical title '{current_title}' "
                f"but career descriptions are entirely technical"
            )

    return None


def _check_assessment_paradox(candidate: dict) -> str | None:
    """Flag 4: Claims 'expert' proficiency but scores poorly on assessments.

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str | None
        A red-flag description string, or ``None`` if clean.
    """
    assessment_scores: dict[str, float] = (
        candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})
    )
    if not assessment_scores:
        return None

    # Build a lookup: skill name (lowered) → proficiency
    skill_proficiency: dict[str, str] = {
        skill.get("name", "").lower(): skill.get("proficiency", "").lower()
        for skill in candidate.get("skills", [])
    }

    threshold = HONEYPOT["expert_with_low_assessment_threshold"]
    paradoxes: list[str] = []

    for skill_name, score in assessment_scores.items():
        prof = skill_proficiency.get(skill_name.lower(), "")
        if prof == "expert" and score < threshold:
            paradoxes.append(f"{skill_name}(score={score}, claims=expert)")

    if paradoxes:
        return (
            f"assessment_paradox: {len(paradoxes)} skill(s) claim expert but "
            f"scored <{threshold}: {', '.join(paradoxes)}"
        )

    return None


def _check_impossible_timeline(candidate: dict) -> str | None:
    """Flag 5: Dates that violate causality or basic arithmetic.

    Checks:
      - Any career entry where ``start_date > end_date``
      - Any single job with ``duration_months`` exceeding total claimed
        experience by more than 12 months (grace buffer for rounding)

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str | None
        A red-flag description string, or ``None`` if clean.
    """
    claimed_years = candidate.get("profile", {}).get("years_of_experience", 0.0)
    max_allowed_months = claimed_years * 12 + 12  # 12-month grace buffer

    issues: list[str] = []

    for entry in candidate.get("career_history", []):
        company = entry.get("company", "unknown")
        title = entry.get("title", "unknown")

        # Check start_date > end_date
        start = _safe_parse_date(entry.get("start_date"))
        end = _safe_parse_date(entry.get("end_date"))

        if start and end and start > end:
            issues.append(
                f"{company}/{title}: start ({start}) > end ({end})"
            )

        # Check if single job duration exceeds total claimed experience
        duration = entry.get("duration_months", 0)
        if duration > max_allowed_months:
            issues.append(
                f"{company}/{title}: duration {duration}mo > "
                f"claimed total {claimed_years}yr + 12mo buffer"
            )

    if issues:
        return f"impossible_timeline: {'; '.join(issues)}"

    return None


def _check_endorsement_anomaly(candidate: dict) -> str | None:
    """Flag 6: Suspiciously high endorsements with zero substance.

    Flags when total endorsements > (skills count × HONEYPOT threshold)
    AND github_activity_score ≤ 0 AND recruiter_response_rate < 0.1.

    Parameters
    ----------
    candidate:
        Full candidate dict.

    Returns
    -------
    str | None
        A red-flag description string, or ``None`` if clean.
    """
    skills = candidate.get("skills", [])
    if not skills:
        return None

    total_endorsements = sum(
        skill.get("endorsements", 0) for skill in skills
    )
    skill_count = len(skills)
    threshold_per_skill = HONEYPOT["suspicious_endorsement_count"]
    endorsement_threshold = skill_count * threshold_per_skill

    if total_endorsements <= endorsement_threshold:
        return None

    signals = candidate.get("redrob_signals", {})
    github_score = signals.get("github_activity_score", 0)
    response_rate = signals.get("recruiter_response_rate", 1.0)

    if github_score <= 0 and response_rate < 0.1:
        return (
            f"endorsement_anomaly: {total_endorsements} total endorsements "
            f"across {skill_count} skills (>{endorsement_threshold} threshold) "
            f"but github_score={github_score}, response_rate={response_rate:.2f}"
        )

    return None


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect_honeypot(candidate: dict) -> tuple[bool, list[str]]:
    """Analyze a candidate for honeypot (fabricated profile) indicators.

    Runs six independent checks. A candidate is classified as a honeypot only
    when they accumulate ``≥ HONEYPOT['flag_threshold']`` (default 3) red flags.

    Parameters
    ----------
    candidate:
        A single candidate dict matching the schema in ``sample_candidates.json``.

    Returns
    -------
    tuple[bool, list[str]]
        ``(True, [list of red-flag descriptions])`` if the candidate is a
        likely honeypot.
        ``(False, [])`` if the candidate is clean.

    Examples
    --------
    >>> is_hp, flags = detect_honeypot(candidate)
    >>> if is_hp:
    ...     print(f"Honeypot detected with {len(flags)} flags:")
    ...     for f in flags:
    ...         print(f"  - {f}")
    """
    checks = [
        _check_experience_tenure_mismatch,
        _check_skill_inflation,
        _check_title_description_mismatch,
        _check_assessment_paradox,
        _check_impossible_timeline,
        _check_endorsement_anomaly,
    ]

    red_flags: list[str] = []
    for check_fn in checks:
        result = check_fn(candidate)
        if result is not None:
            red_flags.append(result)

    flag_threshold: int = HONEYPOT["flag_threshold"]
    is_honeypot = len(red_flags) >= flag_threshold

    return (is_honeypot, red_flags) if is_honeypot else (False, [])
