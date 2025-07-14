import re
import numpy as np
import torch
from typing import List, Any
import argparse
import os
import time
import psutil

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

from helpers import clean_html, get_html_path

# ---- SETTINGS ----
CHUNK_SIZE = 100  # Number of words per chunk
TOP_K = 3         # Number of chunks to retrieve
DEFAULT_LLM_MODEL = "microsoft/phi-4-mini-instruct"
EMBED_MODEL = "all-MiniLM-L6-v2"

def chunk_text(text: str, chunk_size: int = 500) -> List[str]:
    """
    Split text into chunks of approximately chunk_size words.
    """
    tokens = re.split(r'(>|\s+)', text)  # Split on '>' or whitespace
    tokens = [token for token in tokens if token.strip()]  # Remove empty tokens
    return [
        ' '.join(tokens[i:i + chunk_size])
        for i in range(0, len(tokens), chunk_size)
    ]

def retrieve_chunks(
    query: str,
    chunk_embeddings: np.ndarray,
    chunks: List[str],
    k: int = 3
) -> List[str]:
    """
    Retrieve the top-k most relevant chunks for a query using cosine similarity.
    """
    query_embedding = embedder.encode([query])[0]
    similarities = np.dot(chunk_embeddings, query_embedding)
    top_k_idx = np.argsort(similarities)[::-1][:k]
    return [chunks[i] for i in top_k_idx]

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG recipe extraction from a web page file.")
    parser.add_argument("html_file", type=str, help="Path to the html text file.")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL, help="HuggingFace model name (default: microsoft/phi-4-mini-instruct)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for generation (default: 0.2)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling for generation (default: 0.95)")
    parser.add_argument("--max_tokens", type=int, default=10000, help="Maximum number of new tokens to generate (default: 256)")
    args = parser.parse_args()

    html_path = get_html_path(args.html_file)

    if not os.path.isfile(html_path):
        print(f"Error: File not found: {html_path}")
        return

    start = time.time()
    with open(html_path, "r", encoding="utf-8") as f:
        your_long_document: str = f.read()
    print(f"Read file in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    query: str =  f"""You are an expert at extracting structured data from HTML.

    INSTRUCTION:
    Extract the recipe title, ingredients, and directions from the provided webpage HTML.
    Respond ONLY with a single JSON object, with the following keys: title, ingredients, directions.
    Do NOT include any explanations, examples, HTML, or extra text—only the JSON object.
    
    FORMAT:
    {{
      "title": "...",
      "ingredients": ["...", "..."],
      "directions": ["...", "..."]
    }}

    ### Example 

    Input: <html><body><h1>Bad Cake</h1><p>Ingredients: 1 cup of flour, 2 eggs, 1/2 cup of sugar</p><p>Directions: Mix the flour and sugar. Add eggs and stir well. Bake at 350°F for 30 minutes.</p></body></html>
    Output: 
    {{
        "title": "Bad Cake",
        "ingredients": [
            "1 cup of flour",
            "2 eggs",
            "1/2 cup of sugar"
        ],
        "directions": [
            "Mix the flour and sugar.",
            "Add eggs and stir well.",
            "Bake at 350°F for 30 minutes."
        ]
    }}
    """

    start = time.time()
    html_document = clean_html(your_long_document)
    print(f"Cleaned HTML in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # ---- STEP 1: CHUNKING ----
    start = time.time()
    chunks: List[str] = chunk_text(html_document, chunk_size=CHUNK_SIZE)
    print(f"Document split into {len(chunks)} chunks in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # ---- STEP 2: EMBED CHUNKS ----
    start = time.time()
    global embedder  # Needed for retrieve_chunks
    embedder = SentenceTransformer(EMBED_MODEL)
    chunk_embeddings: np.ndarray = embedder.encode(chunks, show_progress_bar=True)
    print(f"Embedded chunks in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # ---- STEP 3: RETRIEVE RELEVANT CHUNKS ----
    start = time.time()
    relevant_chunks: List[str] = retrieve_chunks(query, chunk_embeddings, chunks, k=TOP_K)
    print(f"Retrieved relevant chunks for your query in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    # ---- STEP 4: BUILD PROMPT ----
    context: str = "\n\n".join(relevant_chunks)
    prompt: str = f"""Instruction:
{query}

Context:
{context}
"""

    # ---- STEP 5: RUN THE LLM ----
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Loaded model on {device} in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024 ** 2:.2f} MB")

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(device)

    print(f"Generating... (prompt tokens: {inputs.input_ids.numel()}) tensor: {inputs.input_ids.device}")

    start = time.time()
    with torch.no_grad():
        outputs: Any = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"Processed inference in {time.time() - start:.2f} sec, RSS: {psutil.Process().memory_info().rss / 1024**2:.2f} MB")

    answer: str = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ---- STEP 6: SHOW RESULT ----
    print("\n==== RAG Output ====\n")
    print(answer)

if __name__ == "__main__":
    main()
