from pathlib import Path
import os
import json
import pickle
from typing import List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer

class SuggestRequest(BaseModel):
    ingredients: List[str]
    recipe_style: Optional[str] = ""

useOpenAI = False


# locate project root and model file (matches notebook layout)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = PROJECT_ROOT / "models" / "embeddings.pkl"
STATIC_DIR = PROJECT_ROOT / "static"
RECIPES_DIR = PROJECT_ROOT / "data" / "potential_labels"

# Globals populated at startup
embedding_db = []
_mat = None  # (N, D) numpy array of stored vectors (L2-normalized)
model = None  # sentence-transformer for encoding queries


def _load_resources():
    global embedding_db, _mat, model
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Embeddings file not found at {EMBEDDINGS_PATH}")

    with open(EMBEDDINGS_PATH, "rb") as f:
        embedding_db = pickle.load(f)

    if len(embedding_db) == 0:
        _mat = np.empty((0, 0))
    else:
        mat = np.vstack([np.asarray(e["vector"]) for e in embedding_db])
        # L2 normalize rows to make dot product == cosine similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        _mat = mat / norms

    # load sentence-transformer to embed queries the same way as notebook
    model = SentenceTransformer("all-MiniLM-L6-v2")


def find_best_matches(query_ingredients: List[str], top_n: int = 10):
    """Return list of (index, entry_dict, score) for top_n matches."""
    if not embedding_db:
        return []

    q_text = ", ".join(query_ingredients)
    q_vec = model.encode(q_text)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0:
        q_norm = 1.0
    q_vec = q_vec / q_norm

    sims = _mat @ q_vec  # dot product with normalized rows -> cosine similarity

    k = min(int(top_n), sims.size)
    top_idx = np.argsort(sims)[::-1][:k]
    results = []
    for i in top_idx:
        entry = embedding_db[int(i)]
        results.append((int(i), entry, float(sims[int(i)])))
    return results


def build_candidate_list(matches):
    """Return a JSON string of candidate recipes for inclusion in the prompt."""
    candidates = []
    for _, entry, score in matches:
        candidates.append({
            "recipe_name": entry.get("title"),
            "file_name": entry.get("source_file"),
            "ingredients": entry.get("ingredients"),
            "score": score,
        })
    # dump compact JSON array string
    return json.dumps(candidates)

def extract_json_from_text(text: str):
    """
    Try to extract and parse the first JSON object or array found in text.
    Returns the parsed object, or None if nothing valid was found.
    """
    import json

    # 1) quick attempt: whole text is JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    dec = json.JSONDecoder()
    # search for likely start chars
    for start_char in ('{', '['):
        idx = text.find(start_char)
        while idx != -1:
            try:
                obj, end = dec.raw_decode(text[idx:])
                # raw_decode succeeded; return the object
                return obj
            except json.JSONDecodeError:
                # try next occurrence of the same start_char
                idx = text.find(start_char, idx + 1)

    return None


def call_llm(candidate_list_str: str, user_ingredients: List[str], recipe_style: str) -> dict:
    """Call OpenAI or local Ollama as in the notebook and return parsed JSON or raw content."""
    system_msg = (
        "You are a helpful cooking assistant. A user has certain ingredients, and we have some candidate recipes from a database. "
        "Choose which recipe is the best match for the user's ingredients and give a reason for choosing it."
    )

    user_msg = (
        f"The user has the following ingredients: {', '.join(user_ingredients)}.\n"
        f"The candidate recipes are:\n{candidate_list_str}\n"
        f"The user wants a recipe that matches '{recipe_style}'\n"
        "Which recipe from the candidate recipes best matches the user's ingredients and has the style that user wants?\n"
        "Respond with the recipe name and file_name and the reason for picking this recipe in JSON. Like this: "
        "{\"recipe_name\": \"Tomato Soup\", \"file_name\": \"recipe_00031.json\", \"reason\": \"Because soup is good food\"}"
    )

    if useOpenAI and os.getenv("OPENAI_API_KEY"):
        API_KEY = os.getenv("OPENAI_API_KEY")
        api_url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        }
        model_name = "gpt-4.1-nano"
    else:
        api_url = "http://127.0.0.1:11434/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        model_name = "gemma3:1b"

    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    }
    if not model_name.startswith("gpt-5"):
        data["temperature"] = 0.3

    resp = requests.post(api_url, headers=headers, json=data)
    try:
        resp.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e} - {resp.text}")

    resp_json = resp.json()
    # attempt to extract assistant content
    content = None
    try:
        content = resp_json["choices"][0]["message"]["content"].strip()
    except Exception:
        # fallback to raw text if structure differs
        content = resp.text

    # try parse JSON from assistant
    parsed = extract_json_from_text(content)
    if parsed is not None:
        return {"ok": True, "parsed": parsed}            
    return {"ok": False, "text": content}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    try:
        _load_resources()
    except Exception as e:
        # raise so server fails fast if resources missing
        raise RuntimeError(f"Failed to load resources: {e}")

    # Mount static assets if present
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    yield
    # shutdown (no-op)
    return

app = FastAPI(lifespan=lifespan)


@app.post("/suggest_recipe")
def suggest_recipe(req: SuggestRequest):
    # 1) find best matches (use top_n * 2 to be generous to LLM)
    candidates = find_best_matches(req.ingredients, top_n=25)

    # 2) construct candidate JSON string for prompt
    candidate_list_str = build_candidate_list(candidates)

    # 3) call LLM using same prompt structure as notebook
    llm_out = call_llm(candidate_list_str, req.ingredients, req.recipe_style)

    return {
        "suggested_recipe": llm_out,
        "candidates": [
            {"index": idx, "title": entry.get("title"), "file": entry.get("source_file"), "score": score}
            for idx, entry, score in candidates
        ],
    }


@app.get("/", response_class=HTMLResponse)
def serve_index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        # Minimal fallback page if the SPA isn't built yet
        return HTMLResponse("""
        <!doctype html>
        <html><head><meta charset='utf-8'><title>Recipe Suggest</title></head>
        <body><h1>Recipe Suggest</h1><p>The SPA was not found. Create static/index.html.</p></body></html>
        """, status_code=200)
    return FileResponse(str(index_path))


@app.get("/recipe/{filename}")
def get_recipe_file(filename: str):
    """Return the JSON contents of a recipe file from data/potential_labels by filename.
    - Blocks path traversal; only plain filenames like 'recipe_00031.json' are allowed.
    - Returns 404 if the file does not exist or is not a .json file.
    """
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if not safe_name.endswith(".json"):
        raise HTTPException(status_code=400, detail="Filename must end with .json")

    file_path = RECIPES_DIR / safe_name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Recipe file not found")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read recipe: {e}")


