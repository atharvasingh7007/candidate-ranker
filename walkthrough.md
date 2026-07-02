# Redrob Data & AI Challenge System Walkthrough

I have successfully completed the intelligent candidate ranking system for Redrob! The architecture is structured perfectly to comply with the < 5 minute runtime on a 16GB CPU requirement, while offloading the heavy model training to Kaggle.

## The Architecture at a Glance

The pipeline is split into two phases:

1. **Pre-computation (Kaggle)**: Handled by `precompute.py`.
2. **Fast Local Inference (Runtime)**: Handled by `rank.py`.

## Core Components Built

- **`config.py`**: The central source of truth storing skill taxonomies (Must-Have, Nice-to-Have, Anti-Skills), configuration thresholds, and scoring parameters.
- **`src/candidate_loader.py`**: A highly memory-efficient streaming generator for the 100K JSONL candidates to keep memory usage low.
- **`src/jd_parser.py`**: A structured parsing of the Job Description without requiring `python-docx` as a runtime dependency.
- **`src/coarse_filter.py`**: Eliminates obvious non-fits using lenient experience band logic and keyword checks while respecting "career-pivot escape hatches".
- **`src/honeypot_detector.py`**: Detects the ~80 impossible/fake candidate profiles (which would result in disqualification) by applying independent checks for experience mismatch, skill inflation, and endorsement anomalies.
- **`src/feature_engineer.py`**: Extracts normalized (0.0 to 1.0) scores across 6 key metrics: Skills Match, Career Trajectory, Experience Fit, Location Score, Education Score, and Anti-Patterns.
- **`src/semantic_scorer.py`**: Uses pre-computed embeddings via vector-math cosine similarities to quickly compute semantic matches.
- **`src/behavioral_scorer.py`**: Processes 11 "Redrob platform signals" to generate a behavioral multiplier between 0.5 (stale/unresponsive) and 1.2 (highly active/engaged).
- **`src/reasoning_generator.py`**: Generates a clean, transparent, and structured reasoning string per candidate based on the calculated features.
- **`src/ranker.py`**: Orchestrates the entire pipeline from top to bottom.

## Entry Points

- **`rank.py`**: This is your main submission script. Run this on your local PC. It executes the entire logic using the artifacts downloaded from Kaggle, and yields `submission.csv` in under a minute!
- **`precompute.py`**: This is the heavy lifting script. **Run this on Kaggle using a GPU**. It computes embeddings using `sentence-transformers`, prepares pairwise training data, and trains an XGBoost model. It outputs all the artifacts required by `rank.py` into a `data/` folder.
- **`app.py`**: A Streamlit application intended for the Sandbox requirement. You can upload candidate data and preview how they rank and visually explore the top features.
- **`README.md`**: Provides a complete step-by-step guide on how to configure and execute the Kaggle pre-computation phase, download the artifacts, and execute the final ranker.
- **`Dockerfile` & `requirements.txt`**: Minimal, ready-to-deploy environments.
- **`submission_metadata.yaml`**: The required submission template filled with the approach overview and declarations.

## Next Steps

1. Read the comprehensive guide in the [README.md](file:///C:/Users/Atharva/Documents/redrob/README.md).
2. Create a Kaggle notebook and run `precompute.py`.
3. Download the generated `data/` artifacts into your `c:\Users\Atharva\Documents\redrob\data\` folder.
4. Run `python rank.py --candidates ./candidates.jsonl` locally to generate your final output CSV!
