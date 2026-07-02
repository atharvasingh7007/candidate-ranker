"""
behavioral_scorer.py — Compute a behavioral multiplier from redrob platform signals.

The multiplier ranges from 0.5 (worst) to 1.2 (best) and is applied
multiplicatively to the composite feature score.  A candidate who is active,
responsive, and verified gets a boost; a ghost candidate gets penalized.

The function also returns a breakdown dict for debugging / explainability.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BEHAVIORAL


def _parse_date(date_str: str) -> datetime:
    """Parse an ISO-ish date string into a *datetime* object.

    Supports ``YYYY-MM-DD`` and ``YYYY-MM-DDTHH:MM:SS`` (with or without
    timezone offset).  Falls back to a very old date if parsing fails so the
    caller treats the candidate as maximally inactive.
    """
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(date_str, fmt)
        except (ValueError, TypeError):
            continue
    # Fallback: treat as very old
    return datetime(2020, 1, 1)


def compute_behavioral_multiplier(
    signals: dict,
    reference_date: str = "2026-07-01",
) -> tuple[float, dict[str, float]]:
    """Compute a multiplicative behavioral multiplier from platform signals.

    Parameters
    ----------
    signals : dict
        Dictionary of behavioral / platform signals for one candidate.
        Expected keys (all optional — missing keys use conservative defaults):

        - ``last_active_date`` (str)
        - ``open_to_work_flag`` (bool)
        - ``recruiter_response_rate`` (float, 0–1)
        - ``avg_response_time_hours`` (float)
        - ``interview_completion_rate`` (float, 0–1)
        - ``notice_period_days`` (int)
        - ``github_activity_score`` (float)
        - ``profile_completeness_score`` (float, 0–100)
        - ``verified_email`` (bool)
        - ``verified_phone`` (bool)
        - ``linkedin_connected`` (bool)
        - ``saved_by_recruiters_30d`` (int)
        - ``willing_to_relocate`` (bool)

    reference_date : str
        ISO date used as "today" for staleness calculations.

    Returns
    -------
    tuple[float, dict[str, float]]
        ``(final_multiplier, breakdown)`` where *breakdown* maps each
        component name to its individual multiplier contribution.
    """
    ref_dt: datetime = _parse_date(reference_date)
    multiplier: float = 1.0
    breakdown: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 1. Activity recency  (exponential decay)
    # ------------------------------------------------------------------
    last_active_str: str = signals.get("last_active_date", "")
    if last_active_str:
        last_active_dt: datetime = _parse_date(last_active_str)
        days_inactive: float = max(0.0, (ref_dt - last_active_dt).days)
    else:
        days_inactive = 365.0  # unknown → assume stale

    decay_rate: float = BEHAVIORAL["inactive_decay_rate"]
    recency_mult: float = max(0.5, 1.0 - days_inactive * decay_rate)

    # Extra penalty for very stale profiles
    if days_inactive > BEHAVIORAL["inactive_heavy_penalty_days"]:
        recency_mult *= 0.85

    breakdown["activity_recency"] = recency_mult
    multiplier *= recency_mult

    # ------------------------------------------------------------------
    # 2. Open to work
    # ------------------------------------------------------------------
    open_to_work: bool = signals.get("open_to_work_flag", True)
    if not open_to_work:
        otw_mult: float = BEHAVIORAL["not_open_to_work_multiplier"]
    else:
        otw_mult = 1.0
    breakdown["open_to_work"] = otw_mult
    multiplier *= otw_mult

    # ------------------------------------------------------------------
    # 3. Response rate
    # ------------------------------------------------------------------
    response_rate: float | None = signals.get("recruiter_response_rate")
    if response_rate is not None:
        if response_rate > BEHAVIORAL["response_rate_high"]:
            rr_mult: float = BEHAVIORAL["response_rate_boost"]
        elif response_rate < BEHAVIORAL["response_rate_low"]:
            rr_mult = BEHAVIORAL["response_rate_penalty"]
        else:
            rr_mult = 1.0
    else:
        rr_mult = 1.0  # unknown → neutral
    breakdown["response_rate"] = rr_mult
    multiplier *= rr_mult

    # ------------------------------------------------------------------
    # 4. Response time
    # ------------------------------------------------------------------
    avg_response_hours: float | None = signals.get("avg_response_time_hours")
    if (
        avg_response_hours is not None
        and avg_response_hours > BEHAVIORAL["slow_response_hours"]
    ):
        rt_mult: float = BEHAVIORAL["slow_response_penalty"]
    else:
        rt_mult = 1.0
    breakdown["response_time"] = rt_mult
    multiplier *= rt_mult

    # ------------------------------------------------------------------
    # 5. Interview completion
    # ------------------------------------------------------------------
    interview_rate: float | None = signals.get("interview_completion_rate")
    if (
        interview_rate is not None
        and interview_rate < BEHAVIORAL["low_interview_completion"]
    ):
        ic_mult: float = BEHAVIORAL["low_interview_penalty"]
    else:
        ic_mult = 1.0
    breakdown["interview_completion"] = ic_mult
    multiplier *= ic_mult

    # ------------------------------------------------------------------
    # 6. Notice period
    # ------------------------------------------------------------------
    notice_days: int = int(signals.get("notice_period_days", 0))
    if notice_days > 90:
        np_mult: float = BEHAVIORAL["very_long_notice_penalty"]
    elif notice_days > 30:
        np_mult = BEHAVIORAL["long_notice_penalty"]
    else:
        np_mult = 1.0
    breakdown["notice_period"] = np_mult
    multiplier *= np_mult

    # ------------------------------------------------------------------
    # 7. GitHub activity
    # ------------------------------------------------------------------
    github_score: float = float(signals.get("github_activity_score", 0))
    if github_score > BEHAVIORAL["github_high"]:
        gh_mult: float = BEHAVIORAL["github_boost"]
    else:
        gh_mult = 1.0
    breakdown["github_activity"] = gh_mult
    multiplier *= gh_mult

    # ------------------------------------------------------------------
    # 8. Profile completeness
    # ------------------------------------------------------------------
    completeness: float = float(
        signals.get("profile_completeness_score", 100)
    )
    if completeness < BEHAVIORAL["low_completeness"]:
        pc_mult: float = BEHAVIORAL["low_completeness_penalty"]
    else:
        pc_mult = 1.0
    breakdown["profile_completeness"] = pc_mult
    multiplier *= pc_mult

    # ------------------------------------------------------------------
    # 9. Verification
    # ------------------------------------------------------------------
    verified_email: bool = signals.get("verified_email", False)
    verified_phone: bool = signals.get("verified_phone", False)
    linkedin_connected: bool = signals.get("linkedin_connected", False)

    if not all([verified_email, verified_phone, linkedin_connected]):
        ver_mult: float = BEHAVIORAL["unverified_penalty"]
    else:
        ver_mult = 1.0
    breakdown["verification"] = ver_mult
    multiplier *= ver_mult

    # ------------------------------------------------------------------
    # 10. Saved by recruiters (social proof)
    # ------------------------------------------------------------------
    saved_count: int = int(signals.get("saved_by_recruiters_30d", 0))
    if saved_count > 5:
        sp_mult: float = 1.03
    else:
        sp_mult = 1.0
    breakdown["social_proof"] = sp_mult
    multiplier *= sp_mult

    # ------------------------------------------------------------------
    # 11. Willing to relocate
    # ------------------------------------------------------------------
    willing: bool = signals.get("willing_to_relocate", False)
    if willing:
        reloc_mult: float = 1.03
    else:
        reloc_mult = 1.0
    breakdown["willing_to_relocate"] = reloc_mult
    multiplier *= reloc_mult

    # ------------------------------------------------------------------
    # Final clamp
    # ------------------------------------------------------------------
    multiplier = max(0.5, min(1.2, multiplier))
    breakdown["final"] = multiplier

    return multiplier, breakdown
