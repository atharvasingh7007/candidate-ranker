"""
ranker.py — Core ranking orchestrator.

Combines all scoring modules into a single pipeline that produces
the final top-100 ranked candidate list.

This is the main logic module — rank.py (entry point) calls this.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import WEIGHTS, TOP_K, OUTPUT_PRECISION
from src.coarse_filter import passes_coarse_filter
from src.honeypot_detector import detect_honeypot
from src.feature_engineer import extract_features
from src.behavioral_scorer import compute_behavioral_multiplier
from src.reasoning_generator import generate_reasoning

logger = logging.getLogger(__name__)


def compute_composite_score(features: dict, behavioral_multiplier: float) -> float:
    """
    Compute the weighted composite score from structured features and behavioral multiplier.

    Formula:
        composite = (
            W_skills * skills_match +
            W_career * career_trajectory +
            W_semantic * semantic_similarity +
            W_experience * experience_fit +
            W_location * location_fit +
            W_education * education_fit -
            W_anti * anti_pattern
        ) * behavioral_multiplier
    """
    raw = (
        WEIGHTS["skills_match"] * features.get("skills_match", 0.0)
        + WEIGHTS["career_trajectory"] * features.get("career_trajectory", 0.0)
        + WEIGHTS["semantic_similarity"] * features.get("semantic_similarity", 0.0)
        + WEIGHTS["experience_fit"] * features.get("experience_fit", 0.0)
        + WEIGHTS["location_fit"] * features.get("location_fit", 0.0)
        + WEIGHTS["education_fit"] * features.get("education_fit", 0.0)
        - WEIGHTS["anti_pattern_penalty"] * features.get("anti_pattern", 0.0)
    )

    # Apply behavioral multiplier
    score = raw * behavioral_multiplier

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def rank_candidates(
    candidates: list[dict],
    semantic_scores: Optional[dict] = None,
    use_xgboost: bool = False,
    xgb_model=None,
    reference_date: str = "2026-07-01",
) -> list[dict]:
    """
    Full ranking pipeline: filter → score → rank → reason → output.

    Args:
        candidates: List of candidate dicts from candidates.jsonl
        semantic_scores: Pre-computed dict of {candidate_id: semantic_score}
        use_xgboost: Whether to use XGBoost re-ranking (requires trained model)
        xgb_model: Trained XGBoost model (if use_xgboost=True)
        reference_date: Date to compute behavioral recency against

    Returns:
        List of top-100 candidate result dicts, sorted by rank (1 = best)
    """
    t_start = time.time()
    total_candidates = len(candidates)
    logger.info(f"Starting ranking pipeline for {total_candidates} candidates...")

    # ================================================================
    # Stage 1: Coarse Filter
    # ================================================================
    t1 = time.time()
    filtered_candidates = []
    filter_stats = {"passed": 0, "eliminated": 0, "reasons": {}}

    for candidate in candidates:
        passed, reason = passes_coarse_filter(candidate)
        if passed:
            filtered_candidates.append(candidate)
            filter_stats["passed"] += 1
        else:
            filter_stats["eliminated"] += 1
            filter_stats["reasons"][reason] = filter_stats["reasons"].get(reason, 0) + 1

    logger.info(
        f"Stage 1 (Coarse Filter): {filter_stats['passed']} passed, "
        f"{filter_stats['eliminated']} eliminated in {time.time() - t1:.1f}s"
    )
    for reason, count in sorted(filter_stats["reasons"].items(), key=lambda x: -x[1])[:5]:
        logger.info(f"  - {reason}: {count}")

    # ================================================================
    # Stage 2: Honeypot Detection
    # ================================================================
    t2 = time.time()
    clean_candidates = []
    honeypot_count = 0
    honeypot_details = {}

    for candidate in filtered_candidates:
        is_honeypot, flags = detect_honeypot(candidate)
        if is_honeypot:
            honeypot_count += 1
            honeypot_details[candidate["candidate_id"]] = flags
        else:
            clean_candidates.append(candidate)

    logger.info(
        f"Stage 2 (Honeypot Detection): {honeypot_count} honeypots flagged, "
        f"{len(clean_candidates)} clean candidates in {time.time() - t2:.1f}s"
    )

    # ================================================================
    # Stage 3: Feature Engineering + Scoring
    # ================================================================
    t3 = time.time()
    scored_candidates = []

    for candidate in clean_candidates:
        cid = candidate["candidate_id"]

        # Extract structured features
        features = extract_features(candidate)

        # Inject semantic score if available
        if semantic_scores and cid in semantic_scores:
            features["semantic_similarity"] = semantic_scores[cid]
        else:
            # Fallback: use a default based on skills match
            features["semantic_similarity"] = features.get("skills_match", 0.0) * 0.8

        # Compute behavioral multiplier
        signals = candidate.get("redrob_signals", {})
        behavioral_mult, behavioral_breakdown = compute_behavioral_multiplier(
            signals, reference_date=reference_date
        )

        # Compute composite score
        if use_xgboost and xgb_model is not None:
            # Use XGBoost for final scoring
            score = _xgboost_score(features, behavioral_mult, xgb_model)
        else:
            # Use weighted linear composite
            score = compute_composite_score(features, behavioral_mult)

        scored_candidates.append({
            "candidate": candidate,
            "candidate_id": cid,
            "features": features,
            "behavioral_multiplier": behavioral_mult,
            "behavioral_breakdown": behavioral_breakdown,
            "composite_score": score,
        })

    logger.info(
        f"Stage 3 (Feature Engineering): scored {len(scored_candidates)} candidates "
        f"in {time.time() - t3:.1f}s"
    )

    # ================================================================
    # Stage 4: Sort and Select Top-K
    # ================================================================
    t4 = time.time()

    # Sort by composite score descending, then by candidate_id ascending for tiebreak
    scored_candidates.sort(
        key=lambda x: (-x["composite_score"], x["candidate_id"])
    )

    # Select top K
    top_k = scored_candidates[:TOP_K]

    logger.info(
        f"Stage 4 (Top-K Selection): selected top {len(top_k)} from "
        f"{len(scored_candidates)} in {time.time() - t4:.1f}s"
    )

    # ================================================================
    # Stage 5: Generate Reasoning
    # ================================================================
    t5 = time.time()
    results = []

    for rank_idx, entry in enumerate(top_k):
        rank = rank_idx + 1  # 1-indexed
        score = round(entry["composite_score"], OUTPUT_PRECISION)

        reasoning = generate_reasoning(
            candidate=entry["candidate"],
            rank=rank,
            features=entry["features"],
            behavioral_multiplier=entry["behavioral_multiplier"],
        )

        results.append({
            "candidate_id": entry["candidate_id"],
            "rank": rank,
            "score": score,
            "reasoning": reasoning,
            "features": entry["features"],  # kept for debugging, not in CSV
            "behavioral_multiplier": entry["behavioral_multiplier"],
        })

    logger.info(
        f"Stage 5 (Reasoning): generated {len(results)} reasonings "
        f"in {time.time() - t5:.1f}s"
    )

    total_time = time.time() - t_start
    logger.info(f"Total pipeline time: {total_time:.1f}s")

    # ================================================================
    # Sanity Checks
    # ================================================================
    _run_sanity_checks(results, total_candidates)

    return results


def _xgboost_score(features: dict, behavioral_mult: float, model) -> float:
    """Score using XGBoost model instead of linear weights."""
    import xgboost as xgb

    feature_names = [
        "skills_match", "career_trajectory", "semantic_similarity",
        "experience_fit", "location_fit", "education_fit",
        "anti_pattern", "behavioral_multiplier",
    ]

    feature_vector = np.array([
        features.get("skills_match", 0.0),
        features.get("career_trajectory", 0.0),
        features.get("semantic_similarity", 0.0),
        features.get("experience_fit", 0.0),
        features.get("location_fit", 0.0),
        features.get("education_fit", 0.0),
        features.get("anti_pattern", 0.0),
        behavioral_mult,
    ]).reshape(1, -1)

    dmatrix = xgb.DMatrix(feature_vector, feature_names=feature_names)
    score = float(model.predict(dmatrix)[0])

    # Normalize to [0, 1]
    return max(0.0, min(1.0, score))


def _run_sanity_checks(results: list[dict], total_candidates: int) -> None:
    """Run sanity checks on the output."""
    if len(results) != TOP_K:
        logger.warning(f"Expected {TOP_K} results, got {len(results)}")

    # Check scores are non-increasing
    for i in range(len(results) - 1):
        if results[i]["score"] < results[i + 1]["score"]:
            logger.warning(
                f"Score not non-increasing: rank {results[i]['rank']} "
                f"({results[i]['score']}) < rank {results[i+1]['rank']} "
                f"({results[i+1]['score']})"
            )

    # Check for duplicate candidate_ids
    ids = [r["candidate_id"] for r in results]
    if len(ids) != len(set(ids)):
        logger.warning("Duplicate candidate_ids in results!")

    # Check for duplicate ranks
    ranks = [r["rank"] for r in results]
    if len(ranks) != len(set(ranks)):
        logger.warning("Duplicate ranks in results!")

    # Log top-5 for inspection
    logger.info("Top 5 candidates:")
    for r in results[:5]:
        logger.info(
            f"  Rank {r['rank']}: {r['candidate_id']} "
            f"(score={r['score']:.4f}) — {r['reasoning'][:80]}..."
        )

    # Log score distribution
    scores = [r["score"] for r in results]
    logger.info(
        f"Score distribution: min={min(scores):.4f}, max={max(scores):.4f}, "
        f"mean={np.mean(scores):.4f}, median={np.median(scores):.4f}"
    )
