import argparse
import os
import time

import psutil
import requests
from dotenv import load_dotenv

from helpers import clean_html, get_html_path

# Load environment variables from .env
load_dotenv()

OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434/api/generate')

def count_words(text, model):
    # Placeholder: Ollama models may not provide token counting directly
    # You may want to use tiktoken or similar for compatible models
    return len(text.split())

def get_ollama_completion(prompt, ollama_model, max_tokens, temperature, top_p):
    payload = {
        "model": ollama_model,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_ctx": 20000,
            "num_predict": max_tokens
        },
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("response", "[No response from Ollama]")
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Send a recipe HTML prompt to Ollama and get a response.")
    parser.add_argument("html_file", help="Path to the HTML file or recipe ID")
    parser.add_argument("--model", default="phi4-mini", help="Ollama model name (default: phi4-mini)")
    parser.add_argument("--few-shot", type=bool, default=False)
    parser.add_argument("--max_tokens", type=int, default=1024, help="Max tokens for Ollama response")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    args = parser.parse_args()
    example = ''
    if args.few_shot:
        example = """### Example 

Input: <html><body><h1>Bad Cake</h1><p>Ingredients: 1 cup of flour, 2 eggs, 1/2 cup of sugar</p><p>Directions: Mix the flour and sugar. Add eggs and stir well. Bake at 350°F for 30 minutes.</p></body></html>
Output: 
{{
    \"title\": \"Bad Cake\",
    \"ingredients\": [
        \"1 cup of flour\",
        \"2 eggs\",
        \"1/2 cup of sugar\"
    ],
    \"directions\": [
        \"Mix the flour and sugar.\",
        \"Add eggs and stir well.\",
        \"Bake at 350°F for 30 minutes.\"
    ]
}}
"""

    html_path = get_html_path(args.html_file)
    with open(html_path, 'r', encoding='utf-8') as f:
        raw_html = f.read()
    recipe_html = clean_html(raw_html)

    prompt = f"""Extract the recipe from the webpage HTML.  Respond using JSON format with the following keys: title, ingredients, directions.
{example}    
### Input:
{recipe_html}
### Output:
"""

    model = args.model
    word_count = count_words(prompt, model)
    print(f"Prompt word count: {word_count}")

    print("Sending prompt to Ollama...")
    start_time = time.time()

    result = get_ollama_completion(prompt, ollama_model=model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)

    elapsed = time.time() - start_time

    print(f"\n--- Ollama Response ---\n")
    print(result)
    print(f"\nTime elapsed: {elapsed:.2f} seconds")

if __name__ == "__main__":
    main()

