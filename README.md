# Redrob Intelligent Candidate Ranking System

This repository contains the complete implementation for the Redrob Data & AI Challenge.

## Architecture Overview

Our system is divided into two phases to meet the strict 5-minute, CPU-only runtime constraint:

1.  **Offline Pre-computation (Kaggle)**: Runs heavy tasks like Sentence-Transformers embeddings, feature extraction across 100k profiles, and training an XGBoost Learning-to-Rank model. Saves artifacts to a `data/` directory.
2.  **Fast Local Ranking (Local/Sandbox)**: A lightning-fast Python script (`rank.py`) that loads the pre-computed artifacts, runs our leniant Coarse Filter, Honeypot Detector, Feature Engineer, and Semantic Scorer, applies Behavioral Multipliers, and runs inference via the XGBoost model to output the `submission.csv` in under 30 seconds.

## Phase 1: Kaggle Pre-computation Guide (Heavy Works)

You must run this phase on Kaggle using a GPU to pre-compute the embeddings and train the model.

### 1. Create a Kaggle Notebook
1. Go to Kaggle, click **Create -> New Notebook**.
2. Upload the `candidates.jsonl` dataset to your Kaggle workspace.
3. In the right sidebar, under **Settings**, ensure the **Accelerator** is set to **GPU T4 x2** or **P100** (for fast embeddings).
4. Make sure **Internet** is toggled **On**.

### 2. Upload Code to Kaggle
Upload the following files to your Kaggle environment (you can zip the folder and upload it as a dataset or drag-and-drop):
- `precompute.py`
- `config.py`
- `src/` directory containing all modules

### 3. Run Pre-computation
Create a cell in your Kaggle notebook and run:

```bash
# Install dependencies
!pip install sentence-transformers xgboost python-docx numpy pandas scikit-learn tqdm

# Run the pre-computation pipeline
!python precompute.py --candidates /kaggle/input/dataset/candidates.jsonl --output-dir /kaggle/working/data
```

This will perform the following steps:
- Embed all 100K candidates using `sentence-transformers/all-MiniLM-L6-v2`.
- Embed the Job Description text and Anti-patterns.
- Compute cosine similarity semantic scores.
- Extract structured features for all candidates.
- Generate preference pairs and train the XGBoost ranking model.

### 4. Download Artifacts
Once the script finishes, you will see a `data/` folder in `/kaggle/working/`. It contains:
- `candidate_embeddings.npz`
- `jd_embeddings.npz`
- `semantic_scores.json`
- `candidate_features.json`
- `xgb_ranker.json`

**Download the entire `data/` folder and place it in your local project root (`c:\Users\Atharva\Documents\redrob\data\`).**

---

## Phase 2: Local Ranking (Final Output generation)

Once you have the `data/` folder from Kaggle, you can generate your `submission.csv`.

### Requirements
Install the local (CPU-only) dependencies:
```bash
pip install -r requirements.txt
```

### Run the Ranker
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

This script will load the pre-computed models/scores, score the candidates using a multi-stage pipeline, and output the top 100 candidates to `submission.csv`. The execution should comfortably take < 5 minutes.

---

## Sandbox App (Streamlit)

You can launch a local sandbox to visualize the ranking pipeline and inspect the Top 10 candidate features breakdown:

```bash
pip install streamlit
streamlit run app.py
```
Upload a subset of your JSON/JSONL candidates to see how the system ranks them.

---

## What's inside `src/`?
- `candidate_loader.py`: Streaming, memory-efficient loader for large JSONL files.
- `coarse_filter.py`: Lenient, fast rules to eliminate obvious non-fits.
- `honeypot_detector.py`: Avoids disqualification by statistically catching >80 fake profiles (flags experience-tenure mismatch, skill inflation, endorsement anomaly).
- `feature_engineer.py`: Generates continuous [0, 1] scores for Skills, Career Trajectory, Experience Fit, Location, Education, and Anti-patterns. 
- `semantic_scorer.py`: Computes cosine similarity of candidate texts against the JD and Anti-patterns using Sentence-Transformers embeddings.
- `behavioral_scorer.py`: Calculates platform intent and recency multipliers.
- `reasoning_generator.py`: Generates the final, structured rationale snippet for the top 100 outputs.
- `ranker.py`: The pipeline orchestrator.
