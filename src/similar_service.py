from pathlib import Path
import pickle
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from joblib import load as joblib_load
import numpy as np
from scipy.sparse import csr_matrix
from difflib import SequenceMatcher
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
        

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"

# ---------- Config: file paths ----------
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
KMEANS_PATH = MODELS_DIR / "kmeans_model.pkl"
TITLES_PATH = MODELS_DIR / "recipe_titles.pkl"
X_PATH = MODELS_DIR / "tfidf_matrix.joblib"
FILENAMES_PATH = MODELS_DIR / "recipe_filenames.pkl"   # <-- new

# ---------- Core search helpers ----------

vectorizer = None
kmeans = None
titles: List[str] = []
filenames: List[str] = []            # <-- new
X: csr_matrix | None = None
cluster_to_indices: dict[int, np.ndarray] = {}

class SimilarResponse(BaseModel):
    query: str
    cluster: int | None
    total_candidates: int
    results: List[Dict[str, Any]]
    matched_title: str | None = None            # <-- new
    matched_filename: str | None = None         # <-- new

def _build_cluster_index(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Map cluster_id -> np.ndarray of row indices."""
    clusters: dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(i)
    return {cid: np.asarray(idxs, dtype=np.int32) for cid, idxs in clusters.items()}

@asynccontextmanager
async def lifespan(app):
    # startup: load artifacts (same logic as previous _load_artifacts)
    global vectorizer, kmeans, titles, filenames, X, cluster_to_indices
    try:
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        with open(KMEANS_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(TITLES_PATH, "rb") as f:
            titles = pickle.load(f)
        with open(FILENAMES_PATH, "rb") as f:
            filenames = pickle.load(f)
        
        X = joblib_load(X_PATH)
        if not hasattr(X, "shape"):
            raise ValueError("Loaded X is not a sparse matrix.")

        # build cluster index from kmeans.labels_
        if hasattr(kmeans, "labels_"):
            labels = np.asarray(kmeans.labels_, dtype=np.int32)
        else:
            raise RuntimeError("kmeans.labels_ not found. Refit or save labels separately.")

        cluster_to_indices = _build_cluster_index(labels)

        # sanity checks
        if len(titles) != labels.shape[0]:
            raise RuntimeError(
                f"titles length ({len(titles)}) != number of samples ({labels.shape[0]}). "
                "Artifacts must be built from the same dataset/order."
            )
        if X is not None and X.shape[0] != labels.shape[0]:
            raise RuntimeError(
                f"X rows ({X.shape[0]}) != number of samples ({labels.shape[0]}). "
                "Artifacts out of sync."
            )
        if filenames and len(filenames) != labels.shape[0]:
            raise RuntimeError(
                f"filenames length ({len(filenames)}) != number of samples ({labels.shape[0]}). "
                "Save filenames in the same order as titles when building artifacts."
            )

        yield  # application runs after this

    finally:
        # optional: shutdown cleanup
        pass

# create app with lifespan handler
app = FastAPI(title="Recipe Similarity API", version="1.0.0", lifespan=lifespan)

def _cosine_sim_rank(q_vec: csr_matrix, cand_matrix: csr_matrix, top_k: int) -> np.ndarray:
    """
    Rank candidates by cosine similarity. With TF-IDF default norm='l2',
    dot product equals cosine similarity.
    Returns array of candidate row indices (relative to cand_matrix) sorted desc.
    """
    # sims shape: (1, n_cands)
    sims = (q_vec @ cand_matrix.T).toarray().ravel()
    # argsort descending, take top_k
    top_local = np.argpartition(-sims, kth=min(top_k, sims.size - 1))[:top_k]
    # sort those top_k by true score
    top_local = top_local[np.argsort(-sims[top_local])]
    return top_local, sims

def _format_results(candidate_global_indices: np.ndarray, sims: np.ndarray, top_local: np.ndarray) -> List[Dict[str, Any]]:
    out = []
    for local_idx in top_local:
        gi = int(candidate_global_indices[local_idx])
        out.append({
            "index": gi,
            "title": titles[gi],
            "filename": (filenames[gi] if filenames and filenames[gi] else None),  # <-- include filename
            "score": float(sims[local_idx])  # cosine similarity (0..1)
        })
    return out

def _best_title_index(name: str, cutoff: float = 0.35) -> tuple[int | None, float]:
    """
    Return (best_index, score). Score is in [0..1]. Uses rapidfuzz if available,
    otherwise falls back to difflib.SequenceMatcher.
    """
    if not titles:
        return None, 0.0

    # try rapidfuzz for better fuzzy matching if installed
    try:
        best = rf_process.extractOne(
            name, {i: t for i, t in enumerate(titles)}, scorer=rf_fuzz.token_sort_ratio
        )
        if best is None:
            return None, 0.0
        match_val, score, idx = best  # rapidfuzz returns (match, score, key)
        return int(idx), float(score) / 100.0
    except Exception:
        # fallback to difflib
        name_l = name.lower()
        best_idx = None
        best_score = 0.0
        for i, t in enumerate(titles):
            score = SequenceMatcher(None, name_l, t.lower()).ratio()
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx, best_score

# ---------- Endpoint ----------

@app.get("/similar_recipes", response_model=SimilarResponse)
def similar_recipes(
    query: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),  # default to 10 results as requested
    fuzzy_cutoff: float = Query(0.35, ge=0.0, le=1.0),
):
    """
    If the query looks like a recipe title, perform a fuzzy title match. If a good title
    match is found, use that recipe's TF-IDF ingredient vector as the query vector and
    return recipes similar by ingredients. If no title match passes `fuzzy_cutoff`,
    fall back to free-text vectorizing of the query (original behavior).
    """
    if vectorizer is None or kmeans is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    if X is None:
        raise HTTPException(
            status_code=503,
            detail="TF-IDF matrix (X) not available on server. Save and deploy tfidf_matrix.joblib.",
        )

    # 1) Try fuzzy title match
    best_idx, match_score = _best_title_index(query, cutoff=fuzzy_cutoff)
    if best_idx is not None and match_score >= fuzzy_cutoff:
        q_idx = int(best_idx)
        q_vec = X[q_idx]

        # Determine cluster for that recipe
        if hasattr(kmeans, "labels_"):
            cluster_id = int(kmeans.labels_[q_idx])
        else:
            cluster_id = int(kmeans.predict(q_vec)[0])

        cand_idxs = cluster_to_indices.get(cluster_id, np.array([], dtype=np.int32))
        if cand_idxs.size == 0:
            cand_idxs = np.arange(X.shape[0], dtype=np.int32)

        cand_idxs = cand_idxs[cand_idxs != q_idx]
        if cand_idxs.size == 0:
            return SimilarResponse(
                query=titles[q_idx],
                cluster=cluster_id,
                total_candidates=0,
                results=[],
                matched_title=titles[q_idx],
                matched_filename=(filenames[q_idx] if filenames and filenames[q_idx] else None),
            )

        cand_matrix = X[cand_idxs]
        top_local, sims = _cosine_sim_rank(q_vec, cand_matrix, top_k=min(top_k, cand_matrix.shape[0]))
        results = _format_results(cand_idxs, sims, top_local)
        return SimilarResponse(
            query=titles[q_idx],
            cluster=cluster_id,
            total_candidates=int(cand_matrix.shape[0]),
            results=results,
            matched_title=titles[q_idx],
            matched_filename=(filenames[q_idx] if filenames and filenames[q_idx] else None),
        )

    # 2) Fallback: free-text query vectorized (original behavior)
    q_vec = vectorizer.transform([query])
    cluster_id = int(kmeans.predict(q_vec)[0])
    cand_idxs = cluster_to_indices.get(cluster_id, np.array([], dtype=np.int32))
    if cand_idxs.size == 0:
        cand_idxs = np.arange(X.shape[0], dtype=np.int32)

    cand_matrix = X[cand_idxs]
    top_local, sims = _cosine_sim_rank(q_vec, cand_matrix, top_k=min(top_k, cand_matrix.shape[0]))
    results = _format_results(cand_idxs, sims, top_local)
    return SimilarResponse(
        query=query,
        cluster=cluster_id,
        total_candidates=int(cand_matrix.shape[0]),
        results=results,
        matched_title=None,
        matched_filename=None,
    )