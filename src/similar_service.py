from pathlib import Path
import pickle
from typing import List, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from joblib import load as joblib_load
import numpy as np
from scipy.sparse import csr_matrix
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from sklearn.metrics.pairwise import cosine_similarity

# Toggle default similarity computation method:
USE_SKLEARN_COSINE = True        

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
STATIC_DIR = PROJECT_DIR / "static"
RECIPES_DIR = PROJECT_DIR / "data" / "potential_labels"

# ---------- Config: file paths ----------
KMEANS_PATH = MODELS_DIR / "kmeans_model.pkl"
TITLES_PATH = MODELS_DIR / "recipe_titles.pkl"
X_PATH = MODELS_DIR / "tfidf_matrix.joblib"
FILENAMES_PATH = MODELS_DIR / "recipe_filenames.pkl"   # <-- new

# ---------- Core search helpers ----------

kmeans = None
titles: List[str] = []
filenames: List[str] = []          
X: csr_matrix | None = None
cluster_to_indices: dict[int, np.ndarray] = {}

class SimilarResponse(BaseModel):
    query: str
    cluster: int | None
    total_candidates: int
    results: List[Dict[str, Any]]
    matched_title: str | None = None            
    matched_filename: str | None = None         

def _build_cluster_index(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Map cluster_id -> np.ndarray of row indices."""
    clusters: dict[int, List[int]] = {}
    for i, c in enumerate(labels):
        clusters.setdefault(int(c), []).append(i)
    return {cid: np.asarray(idxs, dtype=np.int32) for cid, idxs in clusters.items()}

@asynccontextmanager
async def lifespan(app):
    # startup: load artifacts (same logic as previous _load_artifacts)
    global kmeans, titles, filenames, X, cluster_to_indices
    try:
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

# Serve static SPA
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found. Build the SPA under /static.")
    return FileResponse(str(index_path))


def _cosine_sim_rank(query_vector: csr_matrix, candidate_matrix: csr_matrix, top_k: int, use_sklearn: bool ) -> tuple[np.ndarray, np.ndarray]:
    """
    Rank candidates by cosine similarity. With TF-IDF default norm='l2',
    dot product equals cosine similarity.

    Parameters
    - q_vec:      (1 x D) query sparse row
    - cand_matrix:(N x D) candidate sparse matrix
    - top_k:      number of top results to return
    - use_sklearn: if True use sklearn.cosine_similarity (robust, normalizes inside).
                   if False use sparse dot-product (fast, requires pre-normalized vectors).
                   
    Returns (top_local_indices_relative_to_cand_matrix, sims_array)
    """
    
    # Compute similarity scores (1D array length n_cands)
    if use_sklearn:
        # sklearn will handle normalization and safety checks.
        sims = cosine_similarity(query_vector, candidate_matrix).ravel()
    else:
        # Fast sparse dot-product. Correct only if rows are L2-normalized (TF-IDF default).
        sims = (query_vector @ candidate_matrix.T).toarray().ravel()

    n = sims.size

    if n == 0:
        return np.asarray([], dtype=np.int32), sims

    k = min(max(int(top_k), 1), n)  # ensure 1 <= k <= n
    
    # np.argpartition is O(n) and avoids a full sort of all N elements. It returns an unordered partition 
    # where the first k positions contain the top-k items. - We then fully sort only those k items to 
    # produce descending order.
    part = np.argpartition(-sims, kth=k-1)[:k]
    top_local = part[np.argsort(-sims[part])]
    
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
    Return (best_index, score). Score is in [0..1]. 
    """
    if not titles:
        return None, 0.0

    
    best = rf_process.extractOne(
        name, titles, scorer=rf_fuzz.partial_token_sort_ratio
    )
    if best is None:
        return None, 0.0
    match_val, score, idx = best  # rapidfuzz returns (match, score, key)
    return int(idx), float(score) / 100.0

# ---------- Endpoint ----------

@app.get("/similar_recipes", response_model=SimilarResponse)
def similar_recipes(
    recipe_name: str = Query(..., min_length=1),
    top_k: int = Query(10, ge=1, le=100),  # default to 10 results as requested
    fuzzy_cutoff: float = Query(0.35, ge=0.0, le=1.0),
):
    """
    It first performs a fuzzy title match. If a good title match is found, use that recipe's 
    TF-IDF ingredient vector as the query vector and return recipes similar by ingredients. 
    If no good title match is found, return an empty result set.
    """    
    # 1) Try fuzzy title match
    best_idx, match_score = _best_title_index(recipe_name, cutoff=fuzzy_cutoff)
    if best_idx is not None and match_score >= fuzzy_cutoff:
        query_index = int(best_idx)
        query_vector = X[query_index]

        # Determine cluster for that recipe
        cluster_id = int(kmeans.labels_[query_index])
        
        candidate_indexes = cluster_to_indices.get(cluster_id, np.array([], dtype=np.int32))
        if candidate_indexes.size == 0:
            candidate_indexes = np.arange(X.shape[0], dtype=np.int32)

        candidate_indexes = candidate_indexes[candidate_indexes != query_index]
        if candidate_indexes.size == 0:
            return SimilarResponse(
                query=recipe_name,
                cluster=cluster_id,
                total_candidates=0,
                results=[],
                matched_title=titles[query_index],
                matched_filename=(filenames[query_index] if filenames and filenames[query_index] else None),
            )

        candidate_matrix = X[candidate_indexes]
        top_local, sims = _cosine_sim_rank(query_vector, candidate_matrix, top_k=min(top_k, candidate_matrix.shape[0]), use_sklearn=USE_SKLEARN_COSINE)
        results = _format_results(candidate_indexes, sims, top_local)
        return SimilarResponse(
            query=recipe_name,
            cluster=cluster_id,
            total_candidates=int(candidate_matrix.shape[0]),
            results=results,
            matched_title=titles[query_index],
            matched_filename=(filenames[query_index] if filenames and filenames[query_index] else None),
        )

    return SimilarResponse(
        query=recipe_name,
        cluster=0,
        total_candidates=0,
        results=[],
        matched_title=None,
        matched_filename=None,
    )


# ---- Serve recipe JSON by filename (used by SPA modal) ----
@app.get("/recipe/{filename}")
def get_recipe_json(filename: str):
    # Basic safety: prevent path traversal and enforce expected pattern
    if "/" in filename or ".." in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not filename.startswith("recipe_") or not filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Filename must look like recipe_XXXX.json")

    file_path = RECIPES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recipe not found")

    try:
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return JSONResponse(content=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading recipe: {e}")