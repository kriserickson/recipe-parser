import os
from openai import OpenAI
import dotenv
import sys
from pathlib import Path
import tiktoken
import time
import psutil

from helpers import get_html_path, clean_html

# Load environment variables from .env
base_dir = Path(__file__).resolve().parent.parent
dotenv.load_dotenv(base_dir / '.env')

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    print('OPENAI_API_KEY not found in .env')
    sys.exit(1)

client = OpenAI(
  api_key=OPENAI_API_KEY
)

def get_openai_completion(
    open_ai_prompt: str,
    open_ai_model: str,
    max_tokens: int,
    temperature: float,
    top_p: float
) -> str:
    completion = client.responses.create(
        model=open_ai_model,
        input=open_ai_prompt,
        max_output_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p
    )
    return completion.output_text

def count_tokens(token_prompt: str, token_model: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(token_model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(token_prompt))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Send prompt to OpenAI model.")
    parser.add_argument("html_file", type=str, help="HTML file to process (default dir: /data/html_pages)")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="OpenAI model to use (default: gpt-4.1-nano)")
    parser.add_argument("--few-shot", type=bool, default=False)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for OpenAI model (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p for OpenAI model (default: 0.9)")
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

    prompt = f"""Extract the recipe as JSON from the webpage HTML:
{example}    
### Input:
{recipe_html}
### Output:
"""

    model = args.model
    token_count = count_tokens(prompt, model)
    print(f"Prompt token count: {token_count}")
    proceed = input("Continue and send to OpenAI? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Aborted by user.")
        sys.exit(0)

    print("Sending prompt to OpenAI...")
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

    result = get_openai_completion(prompt, open_ai_model=model, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)

    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    elapsed = time.time() - start_time

    print(f"\n--- OpenAI Response ---\n")
    print(result)
    print(f"\nTime elapsed: {elapsed:.2f} seconds")
    print(f"Memory usage: {mem_before:.2f} MB -> {mem_after:.2f} MB (Δ {mem_after - mem_before:.2f} MB)")
