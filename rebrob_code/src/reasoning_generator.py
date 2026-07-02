"""
reasoning_generator.py — Generates specific, honest, varied reasoning for each ranked candidate.

No LLM calls — uses structured templates with dynamic data insertion.
Each reasoning references ONLY facts from the candidate's actual profile.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MUST_HAVE_SKILLS, NICE_TO_HAVE_SKILLS, CONSULTING_FIRMS,
    PRODUCT_COMPANIES, RELEVANT_TITLES, ANTI_SKILLS,
)


def generate_reasoning(
    candidate: dict,
    rank: int,
    features: dict,
    behavioral_multiplier: float,
    honeypot_flags: list[str] | None = None,
) -> str:
    """
    Generate a 1-2 sentence reasoning string for why this candidate is at this rank.

    Rules (from submission_spec.md Stage 4):
    - Reference specific facts from the candidate's profile
    - Connect to specific JD requirements
    - Acknowledge concerns honestly
    - No hallucination — every claim must exist in the profile
    - Each reasoning must be substantively different
    """
    profile = candidate["profile"]
    signals = candidate.get("redrob_signals", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    # ---- Gather facts ----
    name_parts = []
    years = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "Unknown")
    company = profile.get("current_company", "Unknown")
    location = profile.get("location", "Unknown")
    country = profile.get("country", "Unknown")

    # Relevant skills present
    candidate_skill_names = {s["name"].lower() for s in skills}
    matched_must_have = candidate_skill_names & {s.lower() for s in MUST_HAVE_SKILLS}
    matched_nice_to_have = candidate_skill_names & {s.lower() for s in NICE_TO_HAVE_SKILLS}

    # Top skills by proficiency + duration
    skill_highlights = _get_top_skills(skills, 3)

    # Career highlights
    career_highlights = _get_career_highlights(career)

    # Behavioral signals
    response_rate = signals.get("recruiter_response_rate", 0)
    github_score = signals.get("github_activity_score", -1)
    notice_days = signals.get("notice_period_days", 0)
    open_to_work = signals.get("open_to_work_flag", False)
    last_active = signals.get("last_active_date", "unknown")

    # ---- Build strengths ----
    strengths = []

    # Experience + title
    strengths.append(f"{years:.0f} years as {title} at {company}")

    # Product company experience
    product_cos = _find_product_companies(career)
    if product_cos:
        if len(product_cos) == 1:
            strengths.append(f"product-company experience at {product_cos[0]}")
        else:
            strengths.append(f"product-company experience ({', '.join(product_cos[:2])})")

    # Key skills
    if skill_highlights:
        strengths.append(f"skills include {', '.join(skill_highlights)}")

    # Career system-building evidence
    if career_highlights:
        strengths.append(career_highlights)

    # Behavioral positives
    behavioral_positives = []
    if response_rate > 0.6:
        behavioral_positives.append(f"response rate {response_rate:.0%}")
    if github_score > 50:
        behavioral_positives.append(f"GitHub score {github_score:.0f}")
    if notice_days <= 30 and notice_days > 0:
        behavioral_positives.append(f"{notice_days}-day notice")
    if open_to_work:
        behavioral_positives.append("actively looking")

    # ---- Build concerns ----
    concerns = []

    # Location concern
    if country.lower() != "india":
        concerns.append(f"based in {location}, {country} (not India)")
    elif not any(loc in location.lower() for loc in ["pune", "noida", "hyderabad", "bangalore", "bengaluru", "mumbai", "delhi", "gurgaon", "gurugram"]):
        concerns.append(f"based in {location} (not in preferred cities)")

    # Notice period
    if notice_days > 90:
        concerns.append(f"long notice period ({notice_days} days)")
    elif notice_days > 60:
        concerns.append(f"{notice_days}-day notice period")

    # Behavioral red flags
    if response_rate < 0.2 and response_rate > 0:
        concerns.append(f"low response rate ({response_rate:.0%})")
    if github_score == -1:
        concerns.append("no GitHub linked")
    if not open_to_work:
        concerns.append("not marked open to work")

    # Services-heavy background
    services_cos = _find_services_companies(career)
    if len(services_cos) == len(career) and len(career) > 1:
        concerns.append("entire career at IT services firms")
    elif len(services_cos) > len(career) // 2:
        concerns.append("mostly services-company background")

    # Skills gaps
    if not matched_must_have:
        concerns.append("no explicit must-have JD skills listed")
    elif len(matched_must_have) < 3:
        concerns.append("limited must-have skill coverage")

    # Anti-skills dominance
    matched_anti = candidate_skill_names & {s.lower() for s in ANTI_SKILLS}
    if len(matched_anti) > len(matched_must_have) + len(matched_nice_to_have):
        concerns.append("primarily CV/speech/robotics focus vs. NLP/retrieval")

    # ---- Compose reasoning based on rank tier ----
    if rank <= 10:
        reasoning = _compose_top_tier(strengths, behavioral_positives, concerns)
    elif rank <= 30:
        reasoning = _compose_high_tier(strengths, behavioral_positives, concerns)
    elif rank <= 70:
        reasoning = _compose_mid_tier(strengths, behavioral_positives, concerns)
    else:
        reasoning = _compose_low_tier(strengths, behavioral_positives, concerns)

    return reasoning


def _get_top_skills(skills: list[dict], n: int) -> list[str]:
    """Get top N skills sorted by proficiency level and duration."""
    proficiency_order = {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}

    scored = []
    for s in skills:
        prof_score = proficiency_order.get(s.get("proficiency", "beginner"), 0)
        duration = s.get("duration_months", 0)
        scored.append((prof_score * 100 + duration, s["name"]))

    scored.sort(reverse=True)
    return [name for _, name in scored[:n]]


def _get_career_highlights(career: list[dict]) -> str:
    """Extract career highlight mentioning systems built."""
    highlight_terms = [
        "ranking", "recommendation", "search", "retrieval", "embedding",
        "nlp", "pipeline", "deployed", "production", "shipped",
        "machine learning", "ml", "at scale", "real users",
    ]

    for job in career:
        desc = job.get("description", "").lower()
        matches = [t for t in highlight_terms if t in desc]
        if len(matches) >= 2:
            company = job.get("company", "")
            title = job.get("title", "")
            return f"built {matches[0]}/{matches[1]} systems as {title} at {company}"

    return ""


def _find_product_companies(career: list[dict]) -> list[str]:
    """Find product companies in career history."""
    found = []
    for job in career:
        company_lower = job.get("company", "").lower()
        for pc in PRODUCT_COMPANIES:
            if pc in company_lower:
                found.append(job["company"])
                break
    return list(dict.fromkeys(found))  # dedupe preserving order


def _find_services_companies(career: list[dict]) -> list[str]:
    """Find consulting/services companies in career history."""
    found = []
    for job in career:
        company_lower = job.get("company", "").lower()
        for sc in CONSULTING_FIRMS:
            if sc in company_lower:
                found.append(job["company"])
                break
    return found


def _compose_top_tier(
    strengths: list[str],
    behavioral: list[str],
    concerns: list[str],
) -> str:
    """Compose reasoning for top-10 candidates."""
    parts = []

    # Lead with strongest 2-3 strengths
    parts.append("; ".join(strengths[:3]))

    # Add behavioral if notable
    if behavioral:
        parts.append(", ".join(behavioral[:2]))

    # Minor concern if any (top-tier should still be honest)
    if concerns:
        parts.append(f"minor note: {concerns[0]}")

    return ". ".join(parts) + "."


def _compose_high_tier(
    strengths: list[str],
    behavioral: list[str],
    concerns: list[str],
) -> str:
    """Compose reasoning for rank 11-30."""
    parts = []

    parts.append("; ".join(strengths[:2]))

    if behavioral:
        parts.append(", ".join(behavioral[:2]))

    if concerns:
        parts.append(f"concern: {concerns[0]}")

    return ". ".join(parts) + "."


def _compose_mid_tier(
    strengths: list[str],
    behavioral: list[str],
    concerns: list[str],
) -> str:
    """Compose reasoning for rank 31-70."""
    parts = []

    # Lead with strengths
    parts.append("; ".join(strengths[:2]))

    # Concerns are more prominent at mid-tier
    if concerns:
        concern_text = "; ".join(concerns[:2])
        parts.append(f"concerns: {concern_text}")
    elif behavioral:
        parts.append(", ".join(behavioral[:1]))

    return ". ".join(parts) + "."


def _compose_low_tier(
    strengths: list[str],
    behavioral: list[str],
    concerns: list[str],
) -> str:
    """Compose reasoning for rank 71-100 — borderline candidates."""
    parts = []

    # Brief strength
    parts.append(strengths[0] if strengths else "Limited profile match")

    # Explain why they're borderline
    if concerns:
        concern_text = "; ".join(concerns[:2])
        parts.append(f"included despite: {concern_text}")

    # What redeems them
    redemption = []
    if behavioral:
        redemption.extend(behavioral[:1])
    if len(strengths) > 1:
        redemption.append(strengths[1])
    if redemption:
        parts.append(f"balanced by {', '.join(redemption)}")

    return ". ".join(parts) + "."
