import time
import argparse
import os
from vllm import LLM, SamplingParams

from helpers import get_html_path, clean_html


def main():
    parser = argparse.ArgumentParser(description="Extract recipe JSON from HTML using vLLM (Phi-4-mini-instruct).")
    parser.add_argument("html_file", type=str, help="HTML file to process (default dir: /data/html_pages)")
    args = parser.parse_args()

    start = time.time()
    html_path = get_html_path(args.html_file)
    if not os.path.isfile(html_path):
        print(f"Error: HTML file not found: {html_path}")
        return

    with open(html_path, "r", encoding="utf-8") as f:
        raw_html = f.read()

    print(f"Read file in {time.time() - start:.2f} sec")

    start = time.time()

    recipe_html = clean_html(raw_html)

    print(f"Cleaned HTML in {time.time() - start:.2f} sec")


    prompt = f"""Extract the recipe as JSON from the webpage HTML:

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

### Input:
{recipe_html}
### Output:
"""

    start = time.time()

    model_id = "Qwen/Qwen3-0.6B"

    # Initialize vLLM model (loads to GPU, supports long contexts)
    llm = LLM(
        model=model_id,
        dtype="float16",  # float16 for speed/VRAM
        max_model_len=20000,  # up to 20k context, adjust for your GPU
        gpu_memory_utilization=0.7, # GPU memory utilization (0.7 = 70% of GPU memory, you might have to adjust this based on your GPU's VRAM size)
    )

    print(f"Loaded model {model_id} in {time.time() - start:.2f} sec")

    print(f"Generating... (prompt length: {len(prompt)} characters)")

    # vLLM batch API (can send multiple prompts in a list!)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=2400
    )

    start = time.time()
    outputs = llm.generate([prompt], sampling_params)
    print(f"Processed inference in {time.time() - start:.2f} sec")

    # Each output corresponds to an input prompt
    for output in outputs:
        print(output.outputs[0].text.strip())


if __name__ == "__main__":
    main()
