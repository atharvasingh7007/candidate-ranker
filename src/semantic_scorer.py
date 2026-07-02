"""
semantic_scorer.py — Embedding-based semantic scoring for candidate ranking.

At **ranking time** this module works entirely with pre-computed embeddings
stored in ``.npz`` files (generated on Kaggle).  All heavy lifting uses
NumPy vectorised operations so 100K candidates can be scored in seconds
on a CPU.

At **pre-computation time** (Kaggle notebook / ``precompute.py``) the helper
functions ``embed_texts`` and ``build_candidate_text`` are used to generate
the embeddings that this module later consumes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Loading pre-computed embeddings
# ============================================================================

def load_embeddings(
    embeddings_path: str | Path,
    jd_embeddings_path: str | Path,
) -> tuple[dict[str, int], np.ndarray, np.ndarray]:
    """Load pre-computed candidate and JD embeddings from ``.npz`` files.

    Parameters
    ----------
    embeddings_path : str | Path
        Path to the candidate embeddings ``.npz`` file.  Expected keys:

        - ``embeddings`` — 2-D float32 array of shape ``(N, dim)``.
        - ``candidate_ids`` — 1-D array of candidate-id strings.

    jd_embeddings_path : str | Path
        Path to the JD embeddings ``.npz`` file.  Expected keys:

        - ``embeddings`` — 2-D float32 array of JD embedding vectors.

    Returns
    -------
    tuple[dict[str, int], np.ndarray, np.ndarray]
        ``(candidate_id_to_index, candidate_embeddings, jd_embeddings)``

        - *candidate_id_to_index* maps each candidate ID string to its row
          index in *candidate_embeddings*.
        - *candidate_embeddings* is the ``(N, dim)`` matrix.
        - *jd_embeddings* is the ``(M, dim)`` matrix of JD vectors.
    """
    cand_data = np.load(str(embeddings_path), allow_pickle=False)
    jd_data = np.load(str(jd_embeddings_path), allow_pickle=False)

    # Build the ID → row-index mapping.  ``candidate_ids`` is stored as a
    # byte/string array inside the npz.
    candidate_ids: np.ndarray = np.load(
        str(embeddings_path), allow_pickle=True
    )["candidate_ids"]
    candidate_id_to_index: dict[str, int] = {
        str(cid): idx for idx, cid in enumerate(candidate_ids)
    }

    candidate_embeddings: np.ndarray = cand_data["embeddings"].astype(
        np.float32
    )
    jd_embeddings: np.ndarray = jd_data["embeddings"].astype(np.float32)

    return candidate_id_to_index, candidate_embeddings, jd_embeddings


# ============================================================================
# Vectorised cosine similarity
# ============================================================================

def compute_cosine_similarity_batch(
    embeddings_matrix: np.ndarray,
    query_vector: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between every row of *embeddings_matrix*
    and a single *query_vector* using fully vectorised NumPy ops.

    Parameters
    ----------
    embeddings_matrix : np.ndarray
        Shape ``(N, dim)``.
    query_vector : np.ndarray
        Shape ``(dim,)`` or ``(1, dim)``.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(N,)`` with cosine similarities in [-1, 1].
    """
    query_vector = query_vector.ravel().astype(np.float32)

    # Dot products  (N,)
    dots: np.ndarray = embeddings_matrix @ query_vector

    # Norms
    emb_norms: np.ndarray = np.linalg.norm(embeddings_matrix, axis=1)
    query_norm: float = float(np.linalg.norm(query_vector))

    # Guard against zero norms
    denom: np.ndarray = emb_norms * query_norm
    denom = np.where(denom == 0, 1e-10, denom)

    return dots / denom


# ============================================================================
# Semantic scoring (ranking time)
# ============================================================================

def compute_semantic_scores(
    candidate_ids: list[str],
    candidate_embeddings_map: dict[str, int],
    all_embeddings: np.ndarray,
    jd_embedding: np.ndarray,
    anti_pattern_embedding: np.ndarray,
) -> dict[str, float]:
    """Compute semantic similarity scores for a batch of candidates.

    For each candidate the score is::

        max(0, cosine(candidate, jd) - 0.3 * cosine(candidate, anti_pattern))

    All operations are batched into a single matrix multiplication for speed.

    Parameters
    ----------
    candidate_ids : list[str]
        Candidate IDs to score.
    candidate_embeddings_map : dict[str, int]
        Mapping from candidate ID → row index in *all_embeddings*.
    all_embeddings : np.ndarray
        Pre-computed embedding matrix, shape ``(N, dim)``.
    jd_embedding : np.ndarray
        JD embedding vector, shape ``(dim,)`` or ``(1, dim)``.
    anti_pattern_embedding : np.ndarray
        Anti-pattern embedding vector, shape ``(dim,)`` or ``(1, dim)``.

    Returns
    -------
    dict[str, float]
        Mapping of candidate_id → semantic score in [0, 1].
    """
    # Resolve indices — skip candidates that have no embedding
    valid_ids: list[str] = []
    indices: list[int] = []
    for cid in candidate_ids:
        idx = candidate_embeddings_map.get(cid)
        if idx is not None:
            valid_ids.append(cid)
            indices.append(idx)

    if not valid_ids:
        return {cid: 0.0 for cid in candidate_ids}

    # Gather embeddings into a contiguous matrix  (batch, dim)
    batch_embeddings: np.ndarray = all_embeddings[indices].astype(np.float32)

    # Vectorised cosine similarities
    positive_scores: np.ndarray = compute_cosine_similarity_batch(
        batch_embeddings, jd_embedding
    )
    negative_scores: np.ndarray = compute_cosine_similarity_batch(
        batch_embeddings, anti_pattern_embedding
    )

    # Final score: clamp to [0, 1]
    raw_scores: np.ndarray = positive_scores - 0.3 * negative_scores
    final_scores: np.ndarray = np.clip(raw_scores, 0.0, 1.0)

    # Build result dict
    result: dict[str, float] = {cid: 0.0 for cid in candidate_ids}
    for cid, score in zip(valid_ids, final_scores):
        result[cid] = float(score)

    return result


# ============================================================================
# Pre-computation helpers  (used by precompute.py on Kaggle)
# ============================================================================

def build_candidate_text(candidate: dict) -> str:
    """Build a single text string from a candidate record for embedding.

    Concatenates headline, summary, all career descriptions, and skill
    names with ``' | '`` as separator.

    Parameters
    ----------
    candidate : dict
        Raw candidate record.

    Returns
    -------
    str
        Flattened text suitable for a sentence-transformer model.
    """
    profile: dict = candidate.get("profile", {})
    headline: str = profile.get("headline", "")
    summary: str = profile.get("summary", "")

    career_descs: list[str] = [
        entry.get("description", "")
        for entry in candidate.get("career_history", [])
        if entry.get("description")
    ]

    skill_names: list[str] = [
        s.get("name", "")
        for s in candidate.get("skills", [])
        if s.get("name")
    ]

    parts: list[str] = [
        headline,
        summary,
        " ".join(career_descs),
        " ".join(skill_names),
    ]

    return " | ".join(p for p in parts if p)


def embed_texts(texts: list[str], model: object) -> np.ndarray:
    """Encode a list of texts into embeddings using a sentence-transformers
    model.  Processes in batches for memory efficiency.

    **This function is only called during pre-computation (on Kaggle).**
    It is NOT used at ranking time.

    Parameters
    ----------
    texts : list[str]
        Raw text strings to embed.
    model : object
        A ``sentence_transformers.SentenceTransformer`` instance (or any
        object with an ``.encode()`` method accepting ``batch_size``).

    Returns
    -------
    np.ndarray
        Embedding matrix of shape ``(len(texts), dim)`` as float32.
    """
    batch_size: int = 256
    embeddings: np.ndarray = model.encode(  # type: ignore[union-attr]
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype(np.float32)
