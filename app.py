"""
app.py — Streamlit sandbox app for the Redrob Hackathon.

This app demonstrates the ranking system on a small candidate sample.
Deploy to HuggingFace Spaces or Streamlit Cloud for the sandbox requirement.

Usage:
    streamlit run app.py
"""

import json
import sys
import time
import io
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import TOP_K
from src.candidate_loader import load_all_candidates
from src.ranker import rank_candidates


def main():
    st.set_page_config(
        page_title="Redrob AI Candidate Ranker",
        page_icon="🎯",
        layout="wide",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: #1e1e2e;
            padding: 1.2rem;
            border-radius: 10px;
            border: 1px solid #333;
            color: #e0e0e0;
        }
        .rank-badge {
            display: inline-block;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
        }
        .score-bar {
            height: 8px;
            border-radius: 4px;
            background: linear-gradient(90deg, #f5576c, #fda085, #f6d365, #a8e063, #11998e);
        }
    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Redrob AI Candidate Ranker</h1>
        <p>Intelligent candidate discovery and ranking for the Senior AI Engineer role.
        Upload candidate data to see the top matches.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.header("⚙️ Configuration")

        reference_date = st.date_input(
            "Reference Date",
            value=pd.Timestamp("2026-07-01"),
            help="Date for computing behavioral signal recency",
        )

        num_results = st.slider(
            "Top N Results",
            min_value=10,
            max_value=100,
            value=100,
            step=10,
        )

        st.divider()
        st.markdown("### 📊 Pipeline Stages")
        st.markdown("""
        1. **Coarse Filter** — Eliminate clearly irrelevant
        2. **Honeypot Detection** — Flag impossible profiles
        3. **Feature Engineering** — 6 scoring dimensions
        4. **Behavioral Signals** — Platform engagement
        5. **Composite Ranking** — Weighted scoring
        6. **Reasoning** — Per-candidate explanation
        """)

    # ── File Upload ──
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload candidates file (.json or .jsonl)",
            type=["json", "jsonl"],
            help="Upload a JSON array or JSONL file with candidate profiles",
        )

    with col2:
        use_sample = st.button("📋 Use Sample Data", type="primary")

    # ── Load Data ──
    candidates = None

    if use_sample:
        sample_path = Path(__file__).parent / "India_runs_data_and_ai_challenge" / "sample_candidates.json"
        if sample_path.exists():
            candidates = load_all_candidates(str(sample_path))
            st.success(f"Loaded {len(candidates)} sample candidates")
        else:
            st.error(f"Sample file not found at {sample_path}")

    elif uploaded_file:
        try:
            content = uploaded_file.read().decode("utf-8")
            if uploaded_file.name.endswith(".jsonl"):
                candidates = [json.loads(line) for line in content.strip().split("\n") if line.strip()]
            else:
                candidates = json.loads(content)
                if not isinstance(candidates, list):
                    candidates = [candidates]
            st.success(f"Loaded {len(candidates)} candidates from upload")
        except Exception as e:
            st.error(f"Error parsing file: {e}")

    if candidates is None:
        st.info("👆 Upload a candidates file or use the sample data to get started.")
        return

    # ── Run Ranking ──
    st.divider()
    st.header("🏆 Ranking Results")

    with st.spinner("Running ranking pipeline..."):
        t_start = time.time()
        results = rank_candidates(
            candidates=candidates,
            semantic_scores=None,  # No pre-computed embeddings in sandbox
            reference_date=str(reference_date),
        )
        elapsed = time.time() - t_start

    # Limit results
    results = results[:num_results]

    # ── Metrics ──
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Candidates", len(candidates))
    with col2:
        st.metric("Ranked Output", len(results))
    with col3:
        st.metric("Processing Time", f"{elapsed:.1f}s")
    with col4:
        top_score = results[0]["score"] if results else 0
        st.metric("Top Score", f"{top_score:.4f}")

    st.divider()

    # ── Results Table ──
    df = pd.DataFrame([
        {
            "Rank": r["rank"],
            "Candidate ID": r["candidate_id"],
            "Score": f"{r['score']:.4f}",
            "Reasoning": r["reasoning"],
        }
        for r in results
    ])

    st.dataframe(
        df,
        use_container_width=True,
        height=500,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Score": st.column_config.TextColumn("Score", width="small"),
            "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
        },
    )

    # ── Feature Breakdown for Top 10 ──
    st.divider()
    st.header("🔍 Feature Breakdown — Top 10")

    feature_data = []
    for r in results[:10]:
        f = r.get("features", {})
        feature_data.append({
            "Rank": r["rank"],
            "ID": r["candidate_id"],
            "Skills": f"{f.get('skills_match', 0):.2f}",
            "Career": f"{f.get('career_trajectory', 0):.2f}",
            "Semantic": f"{f.get('semantic_similarity', 0):.2f}",
            "Experience": f"{f.get('experience_fit', 0):.2f}",
            "Location": f"{f.get('location_fit', 0):.2f}",
            "Education": f"{f.get('education_fit', 0):.2f}",
            "Anti-Pattern": f"{f.get('anti_pattern', 0):.2f}",
            "Behavioral": f"{r.get('behavioral_multiplier', 0):.2f}",
        })

    if feature_data:
        st.dataframe(pd.DataFrame(feature_data), use_container_width=True)

    # ── Download CSV ──
    st.divider()
    csv_data = "candidate_id,rank,score,reasoning\n"
    for r in results:
        reasoning = r["reasoning"].replace('"', "'").replace("\n", " ")
        csv_data += f'{r["candidate_id"]},{r["rank"]},{r["score"]:.4f},"{reasoning}"\n'

    st.download_button(
        label="📥 Download Submission CSV",
        data=csv_data,
        file_name="submission.csv",
        mime="text/csv",
        type="primary",
    )


if __name__ == "__main__":
    main()
