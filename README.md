# Redrob Intelligent Candidate Ranking System

This repository contains the complete implementation for the Redrob Data & AI Challenge.

## Quick Start (Run Directly)

If the pre-computed `data/` artifacts (the XGBoost model and embeddings) are already present in the repository, you can skip the Kaggle step entirely and generate the final `submission.csv` instantly.

### Requirements
Install the local (CPU-only) dependencies:
```bash
pip install -r requirements.txt
```

### Run the Ranker
```bash
python rank.py --candidates ./India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```
This script will load the pre-computed models/scores, score the candidates using a multi-stage pipeline, and output the top 100 candidates to `submission.csv` in **under 2 minutes on a standard CPU**.

---

## Architecture Overview & Design Decisions

Our system is engineered to strictly adhere to the challenge constraints (< 5-minute runtime, CPU-only, no internet) while maximizing accuracy and dodging the hidden "honeypots." 

To achieve this, we split the architecture into two distinct phases:

### Why an Offline/Online Split?
Traditional LLMs (like GPT-4 or Llama) are far too slow to evaluate 100,000 candidates in under 5 minutes on a CPU, and they are prone to hallucinating reasoning. By offloading the heavy embedding generation and model training to a GPU on Kaggle (Offline), our local execution (Online) only has to perform fast matrix lookups and XGBoost tree inference, taking seconds instead of hours.

### 1. Offline Pre-computation (Kaggle/GPU)
* **Dense Semantic Matching:** We use `sentence-transformers/all-MiniLM-L6-v2`. Instead of rigid keyword matching (which misses context), we embed the Job Description and candidates into vectors. This allows us to accurately match a candidate who says "built large-scale recommender systems" even if they don't explicitly list the exact keywords in the JD.
* **Learning-to-Rank (LTR):** We generate preference pairs based on extracted features and train an **XGBoost** model. XGBoost was chosen because it natively handles non-linear feature interactions (e.g., figuring out how to combine high skills with a high anti-pattern penalty) much better than simple linear weights, and inference takes literally milliseconds.

### 2. Fast Local Ranking (Runtime/CPU)
* **Streaming Loader:** Instead of loading 500MB of JSONL into memory at once and crashing, `candidate_loader.py` acts as a streaming generator, keeping RAM usage almost at zero.
* **Honeypot Detector:** We built a dedicated anomaly detection script to catch the ~80 fake profiles. It mathematically identifies experience-tenure mismatches, skill inflation (expert skills with <12mo duration), and impossible timelines. This ensures we are not disqualified for having >10% honeypots.
* **Deterministic Explainability:** Our `reasoning_generator.py` is template-based, extracting the exact mathematical features (Top skills, Trajectory, Behavioral boosts) and converting them into a human-readable sentence. It is 100% factual and hallucination-free.

---

## Phase 1: Kaggle Pre-computation Guide (Heavy Works)

*(If you are training from scratch)* You must run this phase on Kaggle using a GPU to pre-compute the embeddings and train the model.

1. Create a Kaggle Notebook and attach the `candidates.jsonl` dataset. Enable the **GPU T4 x2** accelerator.
2. Upload this codebase to Kaggle (as a dataset or via the interface).
3. Run the following cell:

```bash
!pip install sentence-transformers xgboost python-docx numpy pandas scikit-learn tqdm

# Run the pre-computation pipeline
!python precompute.py --candidates /kaggle/input/dataset/candidates.jsonl --output-dir /kaggle/working/data
```

4. Once the script finishes (it takes ~6.5 minutes), download the generated `data/` folder and place it in your local project root.

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
