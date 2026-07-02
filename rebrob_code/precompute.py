#!/usr/bin/env python3
"""
precompute.py — Offline pre-computation script (runs on Kaggle).

This script performs the heavy computation that doesn't fit in the
5-minute ranking budget:
    1. Embed all 100K candidates using sentence-transformers
    2. Embed the JD (requirements + anti-patterns)
    3. Compute and cache semantic scores
    4. Extract features for all candidates
    5. Generate pairwise training data for XGBoost
    6. Train XGBoost ranking model

Output artifacts are saved to data/ and used by rank.py at submission time.

Usage on Kaggle:
    python precompute.py --candidates ./candidates.jsonl --output-dir ./data
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM, DATA_DIR
from src.candidate_loader import stream_candidates, load_all_candidates
from src.jd_parser import get_jd_profile
from src.semantic_scorer import build_candidate_text, embed_texts
from src.feature_engineer import extract_features
from src.coarse_filter import passes_coarse_filter
from src.honeypot_detector import detect_honeypot
from src.behavioral_scorer import compute_behavioral_multiplier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("precompute")


def step1_embed_candidates(candidates: list[dict], output_dir: Path) -> None:
    """Embed all candidate profiles using sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Build texts
    logger.info("Building candidate texts...")
    candidate_ids = []
    texts = []
    for c in candidates:
        candidate_ids.append(c["candidate_id"])
        texts.append(build_candidate_text(c))

    # Embed in batches
    logger.info(f"Embedding {len(texts)} candidates...")
    embeddings = embed_texts(texts, model)

    # Save
    output_file = output_dir / "candidate_embeddings.npz"
    np.savez_compressed(
        str(output_file),
        embeddings=embeddings,
        candidate_ids=np.array(candidate_ids),
    )
    logger.info(f"Saved candidate embeddings to {output_file} ({embeddings.shape})")


def step2_embed_jd(output_dir: Path) -> None:
    """Embed the job description segments."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    jd = get_jd_profile()

    # Embed the main JD requirements text
    logger.info("Embedding JD requirements...")
    jd_texts = [jd.requirements_text, jd.summary_text]
    jd_embeddings = embed_texts(jd_texts, model)
    jd_embedding = jd_embeddings.mean(axis=0)  # Average of requirements + summary

    # Embed anti-pattern text
    logger.info("Embedding JD anti-patterns...")
    anti_embeddings = embed_texts([jd.anti_pattern_text], model)
    anti_pattern_embedding = anti_embeddings[0]

    # Save
    output_file = output_dir / "jd_embeddings.npz"
    np.savez_compressed(
        str(output_file),
        jd_embedding=jd_embedding,
        anti_pattern_embedding=anti_pattern_embedding,
    )
    logger.info(f"Saved JD embeddings to {output_file}")


def step3_compute_semantic_scores(output_dir: Path) -> dict:
    """Compute semantic similarity scores for all candidates."""
    from src.semantic_scorer import compute_semantic_scores

    # Load embeddings
    cand_data = np.load(str(output_dir / "candidate_embeddings.npz"))
    all_embeddings = cand_data["embeddings"]
    candidate_ids = list(cand_data["candidate_ids"])

    jd_data = np.load(str(output_dir / "jd_embeddings.npz"))
    jd_embedding = jd_data["jd_embedding"]
    anti_pattern_embedding = jd_data["anti_pattern_embedding"]

    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    logger.info("Computing semantic scores...")
    scores = compute_semantic_scores(
        candidate_ids=candidate_ids,
        candidate_embeddings_map=id_to_idx,
        all_embeddings=all_embeddings,
        jd_embedding=jd_embedding,
        anti_pattern_embedding=anti_pattern_embedding,
    )

    # Save
    output_file = output_dir / "semantic_scores.json"
    with open(output_file, "w") as f:
        json.dump(scores, f)
    logger.info(f"Saved semantic scores to {output_file}")

    return scores


def step4_extract_features(
    candidates: list[dict], semantic_scores: dict, output_dir: Path
) -> list[dict]:
    """Extract structured features for all candidates."""
    logger.info("Extracting features for all candidates...")

    all_features = []
    for c in candidates:
        cid = c["candidate_id"]

        # Coarse filter pass/fail
        passed, reason = passes_coarse_filter(c)

        # Honeypot detection
        is_honeypot, flags = detect_honeypot(c)

        # Feature extraction
        features = extract_features(c)
        features["semantic_similarity"] = semantic_scores.get(cid, 0.0)

        # Behavioral
        signals = c.get("redrob_signals", {})
        behavioral_mult, _ = compute_behavioral_multiplier(signals)

        all_features.append({
            "candidate_id": cid,
            "passed_coarse_filter": passed,
            "is_honeypot": is_honeypot,
            "honeypot_flags": flags,
            **features,
            "behavioral_multiplier": behavioral_mult,
        })

    # Save as JSON (lighter than parquet for this use case)
    output_file = output_dir / "candidate_features.json"
    with open(output_file, "w") as f:
        json.dump(all_features, f)
    logger.info(f"Saved features for {len(all_features)} candidates to {output_file}")

    return all_features


def step5_generate_training_pairs(all_features: list[dict]) -> list[tuple]:
    """
    Generate pairwise training data for XGBoost LTR.

    Strategy: create obvious preference pairs from common-sense rules:
    - Technical candidate > non-technical candidate
    - Product-company candidate > services-only candidate
    - Active candidate > inactive candidate
    - Non-honeypot > honeypot

    These pairs train XGBoost to learn feature interaction effects.
    """
    logger.info("Generating pairwise training data...")

    # Separate into tiers based on features
    passed = [f for f in all_features if f["passed_coarse_filter"] and not f["is_honeypot"]]
    failed = [f for f in all_features if not f["passed_coarse_filter"] or f["is_honeypot"]]

    pairs = []

    # Pair 1: passed candidates vs failed candidates
    import random
    random.seed(42)

    for good in random.sample(passed, min(1000, len(passed))):
        for bad in random.sample(failed, min(3, len(failed))):
            pairs.append((good, bad, 1))  # good > bad

    # Pair 2: among passed candidates, rank by composite quality
    # High career_trajectory + skills > low
    high_quality = [f for f in passed if f.get("career_trajectory", 0) > 0.6 and f.get("skills_match", 0) > 0.5]
    low_quality = [f for f in passed if f.get("career_trajectory", 0) < 0.3 or f.get("skills_match", 0) < 0.2]

    for good in random.sample(high_quality, min(500, len(high_quality))):
        for bad in random.sample(low_quality, min(3, len(low_quality))):
            pairs.append((good, bad, 1))

    # Pair 3: active vs inactive
    active = [f for f in passed if f.get("behavioral_multiplier", 0) > 1.0]
    inactive = [f for f in passed if f.get("behavioral_multiplier", 0) < 0.7]

    for good in random.sample(active, min(300, len(active))):
        for bad in random.sample(inactive, min(2, len(inactive))):
            pairs.append((good, bad, 1))

    logger.info(f"Generated {len(pairs)} training pairs")
    return pairs


def step6_train_xgboost(pairs: list[tuple], output_dir: Path) -> None:
    """Train XGBoost ranking model on pairwise data."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.warning("xgboost not installed. Skipping XGBoost training.")
        logger.warning("Install with: pip install xgboost")
        return

    logger.info("Training XGBoost ranking model...")

    feature_names = [
        "skills_match", "career_trajectory", "semantic_similarity",
        "experience_fit", "location_fit", "education_fit",
        "anti_pattern", "behavioral_multiplier",
    ]

    # Convert pairs to pairwise format
    X = []
    y = []

    for good, bad, label in pairs:
        good_vec = [good.get(f, 0.0) for f in feature_names]
        bad_vec = [bad.get(f, 0.0) for f in feature_names]

        # Positive pair: good > bad
        X.append(good_vec)
        y.append(1.0)

        # Negative pair: bad < good
        X.append(bad_vec)
        y.append(0.0)

    X = np.array(X)
    y = np.array(y)

    # Shuffle
    perm = np.random.RandomState(42).permutation(len(X))
    X = X[perm]
    y = y[perm]

    # Train
    dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "nthread": -1,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        verbose_eval=50,
    )

    # Save
    model_file = output_dir / "xgb_ranker.json"
    model.save_model(str(model_file))
    logger.info(f"Saved XGBoost model to {model_file}")

    # Feature importance
    importance = model.get_score(importance_type="gain")
    logger.info("Feature importance (gain):")
    for feat, score in sorted(importance.items(), key=lambda x: -x[1]):
        logger.info(f"  {feat}: {score:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute embeddings, features, and model for ranking"
    )
    parser.add_argument(
        "--candidates",
        type=str,
        required=True,
        help="Path to candidates.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Output directory for pre-computed artifacts",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding computation (use existing)",
    )
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip XGBoost training",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Load candidates
    logger.info(f"Loading candidates from {args.candidates}...")
    candidates = load_all_candidates(args.candidates)
    logger.info(f"Loaded {len(candidates)} candidates")

    # Step 1: Embed candidates
    if not args.skip_embeddings:
        step1_embed_candidates(candidates, output_dir)
        step2_embed_jd(output_dir)
    else:
        logger.info("Skipping embedding computation (--skip-embeddings)")

    # Step 3: Compute semantic scores
    semantic_scores = step3_compute_semantic_scores(output_dir)

    # Step 4: Extract features
    all_features = step4_extract_features(candidates, semantic_scores, output_dir)

    # Step 5-6: XGBoost
    if not args.skip_xgboost:
        pairs = step5_generate_training_pairs(all_features)
        step6_train_xgboost(pairs, output_dir)
    else:
        logger.info("Skipping XGBoost training (--skip-xgboost)")

    total_time = time.time() - t_start
    logger.info(f"Pre-computation complete in {total_time:.1f}s")
    logger.info(f"Artifacts saved to {output_dir.resolve()}")
    logger.info("Next: copy the data/ folder to your local machine and run rank.py")


if __name__ == "__main__":
    main()
