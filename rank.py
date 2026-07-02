#!/usr/bin/env python3
"""
rank.py — Main entry point for producing the submission CSV.

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Must complete in ≤5 minutes on CPU with 16GB RAM and no network access.
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR,
    CANDIDATE_EMBEDDINGS_FILE,
    JD_EMBEDDINGS_FILE,
    XGBOOST_MODEL_FILE,
    TOP_K,
    OUTPUT_PRECISION,
)
from src.candidate_loader import stream_candidates, load_all_candidates
from src.ranker import rank_candidates

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rank")


def load_semantic_scores() -> dict | None:
    """Load pre-computed semantic similarity scores if available."""
    # Option A: Pre-computed per-candidate scores (fastest at runtime)
    scores_file = DATA_DIR / "semantic_scores.json"
    if scores_file.exists():
        logger.info(f"Loading pre-computed semantic scores from {scores_file}")
        with open(scores_file, "r") as f:
            return json.load(f)

    # Option B: Compute from embeddings on the fly
    if CANDIDATE_EMBEDDINGS_FILE.exists() and JD_EMBEDDINGS_FILE.exists():
        logger.info("Computing semantic scores from pre-computed embeddings...")
        from src.semantic_scorer import load_embeddings, compute_semantic_scores

        candidate_data = np.load(str(CANDIDATE_EMBEDDINGS_FILE))
        all_embeddings = candidate_data["embeddings"]
        candidate_ids = list(candidate_data["candidate_ids"])

        jd_data = np.load(str(JD_EMBEDDINGS_FILE))
        jd_embedding = jd_data["jd_embedding"]
        anti_pattern_embedding = jd_data["anti_pattern_embedding"]

        # Build candidate_id -> index map
        id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

        scores = compute_semantic_scores(
            candidate_ids=candidate_ids,
            candidate_embeddings_map=id_to_idx,
            all_embeddings=all_embeddings,
            jd_embedding=jd_embedding,
            anti_pattern_embedding=anti_pattern_embedding,
        )
        return scores

    logger.warning(
        "No pre-computed embeddings found. Semantic scoring will use "
        "skills-based fallback. Run precompute.py on Kaggle first for best results."
    )
    return None


def load_xgb_model():
    """Load pre-trained XGBoost model if available."""
    if XGBOOST_MODEL_FILE.exists():
        try:
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(str(XGBOOST_MODEL_FILE))
            logger.info(f"Loaded XGBoost model from {XGBOOST_MODEL_FILE}")
            return model
        except ImportError:
            logger.warning("xgboost not installed — using weighted scoring instead")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost model: {e}")
    return None


def write_submission_csv(results: list[dict], output_path: Path) -> None:
    """Write the submission CSV in the required format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for r in results:
            # Ensure score precision and clean reasoning
            score_str = f"{r['score']:.{OUTPUT_PRECISION}f}"
            reasoning = r["reasoning"].replace('"', "'").replace("\n", " ").strip()

            writer.writerow([
                r["candidate_id"],
                r["rank"],
                score_str,
                reasoning,
            ])

    logger.info(f"Wrote {len(results)} rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranker — produces top-100 submission CSV"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl (or .json for sample data)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./submission.csv",
        help="Output CSV path (default: ./submission.csv)",
    )
    parser.add_argument(
        "--reference-date",
        type=str,
        default="2026-07-01",
        help="Reference date for behavioral scoring (default: 2026-07-01)",
    )
    parser.add_argument(
        "--no-semantic",
        action="store_true",
        help="Skip semantic scoring (useful if embeddings not yet computed)",
    )
    parser.add_argument(
        "--no-xgboost",
        action="store_true",
        help="Skip XGBoost re-ranking (use weighted linear scoring)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Redrob Intelligent Candidate Ranking System")
    logger.info("=" * 60)

    # ── Step 1: Load candidates ──────────────────────────────────────
    candidates_path = Path(args.candidates)
    logger.info(f"Loading candidates from {candidates_path}...")

    candidates = load_all_candidates(str(candidates_path))
    logger.info(f"Loaded {len(candidates)} candidates")

    # ── Step 2: Load pre-computed artifacts ──────────────────────────
    semantic_scores = None
    if not args.no_semantic:
        semantic_scores = load_semantic_scores()

    xgb_model = None
    if not args.no_xgboost:
        xgb_model = load_xgb_model()

    # ── Step 3: Run ranking pipeline ─────────────────────────────────
    results = rank_candidates(
        candidates=candidates,
        semantic_scores=semantic_scores,
        use_xgboost=(xgb_model is not None),
        xgb_model=xgb_model,
        reference_date=args.reference_date,
    )

    # ── Step 4: Validate output ──────────────────────────────────────
    if len(results) < TOP_K:
        logger.warning(
            f"Only {len(results)} candidates in output (expected {TOP_K}). "
            f"This may indicate overly aggressive filtering."
        )

    # Ensure scores are strictly non-increasing (fix floating point ties)
    for i in range(1, len(results)):
        if results[i]["score"] > results[i - 1]["score"]:
            results[i]["score"] = results[i - 1]["score"]

    # ── Step 5: Write CSV ────────────────────────────────────────────
    output_path = Path(args.out)
    write_submission_csv(results, output_path)

    # ── Done ─────────────────────────────────────────────────────────
    total_time = time.time() - t_start
    logger.info(f"Total execution time: {total_time:.1f}s")

    if total_time > 300:
        logger.error(f"⚠️  Exceeded 5-minute budget! ({total_time:.0f}s)")
    elif total_time > 240:
        logger.warning(f"⚠️  Close to 5-minute budget ({total_time:.0f}s)")
    else:
        logger.info(f"✅ Well within 5-minute budget ({total_time:.0f}s)")

    logger.info(f"Submission written to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
