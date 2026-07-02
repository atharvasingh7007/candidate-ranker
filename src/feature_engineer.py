"""
feature_engineer.py — Extract structured ranking features from a candidate dict.

Each feature is a float in [0.0, 1.0].  The six features feed into the
composite scorer and (optionally) an XGBoost ranker.

Features:
    skills_match       – How well the candidate's skills match the JD.
    career_trajectory  – Product-company experience, shipped systems, tenure.
    experience_fit     – Gaussian fit around ideal years of experience.
    location_fit       – Preference score based on candidate location.
    education_fit      – Relevance of degree field, tier, and level.
    anti_pattern       – Aggregated anti-pattern signals (higher = worse).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ANTI_SKILLS,
    CAREER_POSITIVE_KEYWORDS,
    CONSULTING_FIRMS,
    EXPERIENCE_IDEAL_CENTER,
    EXPERIENCE_SIGMA,
    LOCATION_SCORES,
    DEFAULT_INDIA_LOCATION_SCORE,
    DEFAULT_NON_INDIA_LOCATION_SCORE,
    MUST_HAVE_SKILLS,
    NICE_TO_HAVE_SKILLS,
    NON_TECHNICAL_SKILLS,
    PRODUCT_COMPANIES,
)

# ---------------------------------------------------------------------------
# Proficiency → numeric weight mapping
# ---------------------------------------------------------------------------
_PROFICIENCY_WEIGHT: dict[str, float] = {
    "expert": 1.0,
    "advanced": 0.8,
    "intermediate": 0.5,
    "beginner": 0.2,
}


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to the closed interval [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================================
# A.  Skills Match Score
# ============================================================================

def _skills_match_score(candidate: dict) -> float:
    """Compute a 0–1 score reflecting skill alignment with the JD.

    Scoring logic
    -------------
    1. Weighted count of must-have skills (proficiency × duration bonus).
    2. Same for nice-to-have skills, weighted at 0.5× importance.
    3. Normalize by dividing by 8.0 (max practical score).
    4. Add bonus from ``skill_assessment_scores`` for relevant skills.
    5. Penalize if anti-skills dominate relevant skills.
    """
    skills: list[dict] = candidate.get("skills", [])

    weighted_must: float = 0.0
    weighted_nice: float = 0.0
    relevant_count: int = 0
    anti_count: int = 0

    for skill in skills:
        name_lower: str = skill.get("name", "").lower().strip()
        proficiency: str = skill.get("proficiency", "").lower().strip()
        duration_months: float = float(skill.get("duration_months", 0))

        weight: float = _PROFICIENCY_WEIGHT.get(proficiency, 0.3)

        # Duration bonus: > 24 months → ×1.2
        if duration_months > 24:
            weight *= 1.2

        if name_lower in MUST_HAVE_SKILLS:
            weighted_must += weight
            relevant_count += 1
        elif name_lower in NICE_TO_HAVE_SKILLS:
            weighted_nice += weight
            relevant_count += 1

        if name_lower in ANTI_SKILLS:
            anti_count += 1

    # Base normalized score
    score: float = min(1.0, (weighted_must + 0.5 * weighted_nice) / 8.0)

    # Assessment-scores bonus (average of relevant assessments / 100 × 0.2)
    assessment_scores: dict[str, float] = candidate.get(
        "skill_assessment_scores", {}
    )
    if assessment_scores:
        relevant_assessments: list[float] = [
            float(v)
            for k, v in assessment_scores.items()
            if k.lower().strip() in MUST_HAVE_SKILLS
            or k.lower().strip() in NICE_TO_HAVE_SKILLS
        ]
        if relevant_assessments:
            avg_assessment: float = sum(relevant_assessments) / len(
                relevant_assessments
            )
            score += (avg_assessment / 100.0) * 0.2

    # Anti-skill penalty
    if anti_count > relevant_count and relevant_count >= 0:
        score -= 0.3

    return _clamp(score)


# ============================================================================
# B.  Career Trajectory Score
# ============================================================================

_BUILDER_KEYWORDS: set[str] = {
    "engineer", "developer", "scientist", "architect", "programmer",
    "sde", "swe", "researcher", "analyst",
}

_CODE_SIGNAL_KEYWORDS: set[str] = {
    "code", "coding", "engineering", "develop", "implement", "build",
    "deploy", "python", "java", "golang", "rust", "c++", "scala",
    "programming", "software",
}


def _career_trajectory_score(candidate: dict) -> float:
    """Score the candidate's career history for product-company experience,
    shipped-system evidence, builder-track titles, and tenure stability.
    """
    career: list[dict] = candidate.get("career_history", [])
    if not career:
        return 0.0

    score: float = 0.0

    # --- has_product_company (0.3) ---
    has_product: bool = any(
        entry.get("company", "").lower().strip() in PRODUCT_COMPANIES
        for entry in career
    )
    if has_product:
        score += 0.3

    # --- services_only_gate (hard zero) ---
    all_consulting: bool = all(
        entry.get("company", "").lower().strip() in CONSULTING_FIRMS
        for entry in career
    )
    if all_consulting:
        return 0.0  # hard gate

    # --- has_shipped_systems (up to 0.3) ---
    positive_hit_count: int = 0
    all_descriptions: str = " ".join(
        entry.get("description", "") for entry in career
    ).lower()
    for kw in CAREER_POSITIVE_KEYWORDS:
        if kw in all_descriptions:
            positive_hit_count += 1
    score += min(0.3, positive_hit_count * 0.05)

    # --- is_builder_track (0.2) ---
    current_title: str = candidate.get("profile", {}).get(
        "current_title", ""
    ).lower()
    if any(kw in current_title for kw in _BUILDER_KEYWORDS):
        score += 0.2

    # --- tenure_stability (±0.1) ---
    tenures: list[float] = [
        float(entry.get("duration_months", 0))
        for entry in career
        if entry.get("duration_months") is not None
    ]
    if tenures:
        avg_tenure: float = sum(tenures) / len(tenures)
        if avg_tenure > 24:
            score += 0.1
        elif avg_tenure < 18 and len(career) >= 3:
            score -= 0.1  # title-chaser penalty

    # --- recent_code_signal (0.1) ---
    if career:
        current_desc: str = career[0].get("description", "").lower()
        if any(kw in current_desc for kw in _CODE_SIGNAL_KEYWORDS):
            score += 0.1

    return _clamp(score)


# ============================================================================
# C.  Experience Fit Score
# ============================================================================

def _experience_fit_score(candidate: dict) -> float:
    """Gaussian fit around the ideal years of experience."""
    years: float = float(
        candidate.get("profile", {}).get("years_of_experience", 0)
    )
    return math.exp(
        -((years - EXPERIENCE_IDEAL_CENTER) ** 2)
        / (2.0 * EXPERIENCE_SIGMA ** 2)
    )


# ============================================================================
# D.  Location Score
# ============================================================================

def _location_score(candidate: dict) -> float:
    """Score based on candidate's location and country."""
    profile: dict = candidate.get("profile", {})
    location: str = profile.get("location", "").lower().strip()
    country: str = profile.get("country", "").lower().strip()

    # Check if any LOCATION_SCORES key is a substring of the location
    for key, loc_score in LOCATION_SCORES.items():
        if key in location:
            return loc_score

    # Fallback by country
    if country == "india":
        return DEFAULT_INDIA_LOCATION_SCORE
    return DEFAULT_NON_INDIA_LOCATION_SCORE


# ============================================================================
# E.  Education Score
# ============================================================================

_RELEVANT_FIELDS: set[str] = {
    "cs", "computer science", "it", "information technology",
    "mathematics", "statistics", "data science", "ai",
    "machine learning", "electronics", "ece",
    "electrical and computer engineering",
    "electronic and communication engineering",
}

_TIER_BONUS: dict[str, float] = {
    "tier_1": 0.3,
    "tier_2": 0.2,
    "tier_3": 0.1,
    "tier_4": 0.0,
}

_ADVANCED_DEGREES: set[str] = {"m.tech", "ms", "m.sc", "mba", "phd"}


def _education_score(candidate: dict) -> float:
    """Score education relevance, institution tier, and degree level."""
    education: list[dict] = candidate.get("education", [])
    if not education:
        return 0.3  # base score only

    score: float = 0.3  # base

    # Take the highest-tier / most relevant entry
    best_field_bonus: float = 0.0
    best_tier_bonus: float = 0.0
    has_advanced: bool = False

    for entry in education:
        field: str = entry.get("field_of_study", "").lower().strip()
        tier: str = entry.get("tier", "").lower().strip()
        degree: str = entry.get("degree", "").lower().strip()

        # Relevant field bonus
        if any(rf in field for rf in _RELEVANT_FIELDS):
            best_field_bonus = max(best_field_bonus, 0.3)

        # Tier bonus
        tier_val: float = _TIER_BONUS.get(tier, 0.0)
        best_tier_bonus = max(best_tier_bonus, tier_val)

        # Advanced degree
        if any(ad in degree for ad in _ADVANCED_DEGREES):
            has_advanced = True

    score += best_field_bonus
    score += best_tier_bonus
    if has_advanced:
        score += 0.1

    return _clamp(score)


# ============================================================================
# F.  Anti-Pattern Score  (higher = MORE anti-patterns = BAD)
# ============================================================================

def _anti_pattern_score(candidate: dict) -> float:
    """Detect red-flag anti-patterns.  Higher score means *more* risk."""
    score: float = 0.0

    career: list[dict] = candidate.get("career_history", [])
    skills: list[dict] = candidate.get("skills", [])
    skill_names_lower: list[str] = [
        s.get("name", "").lower().strip() for s in skills
    ]

    # --- services_only (0.4) ---
    if career and all(
        entry.get("company", "").lower().strip() in CONSULTING_FIRMS
        for entry in career
    ):
        score += 0.4

    # --- title_chaser (0.2) ---
    tenures: list[float] = [
        float(entry.get("duration_months", 0))
        for entry in career
        if entry.get("duration_months") is not None
    ]
    if tenures and len(career) >= 3:
        avg_tenure: float = sum(tenures) / len(tenures)
        if avg_tenure < 18:
            score += 0.2

    # --- cv_speech_only (0.2) ---
    anti_count: int = sum(1 for n in skill_names_lower if n in ANTI_SKILLS)
    must_count: int = sum(
        1 for n in skill_names_lower if n in MUST_HAVE_SKILLS
    )
    total_skills: int = len(skill_names_lower)
    if total_skills > 0 and anti_count > total_skills / 2 and must_count < 2:
        score += 0.2

    # --- non_technical_dominant (0.2) ---
    non_tech_count: int = sum(
        1 for n in skill_names_lower if n in NON_TECHNICAL_SKILLS
    )
    if total_skills > 0 and non_tech_count > total_skills / 2:
        score += 0.2

    return _clamp(score)


# ============================================================================
# Public API
# ============================================================================

def extract_features(candidate: dict) -> dict[str, float]:
    """Extract all ranking features from a single candidate dict.

    Parameters
    ----------
    candidate : dict
        Raw candidate record (as loaded from JSONL).

    Returns
    -------
    dict[str, float]
        Keys: ``skills_match``, ``career_trajectory``, ``experience_fit``,
        ``location_fit``, ``education_fit``, ``anti_pattern``.
        All values in [0.0, 1.0].
    """
    return {
        "skills_match": _skills_match_score(candidate),
        "career_trajectory": _career_trajectory_score(candidate),
        "experience_fit": _experience_fit_score(candidate),
        "location_fit": _location_score(candidate),
        "education_fit": _education_score(candidate),
        "anti_pattern": _anti_pattern_score(candidate),
    }
