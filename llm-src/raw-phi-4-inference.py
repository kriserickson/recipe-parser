import time
import torch
import argparse
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationMixin
)

from helpers import get_html_path, clean_html

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract recipe JSON from HTML using Phi-4.")
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

    bnb_config = BitsAndBytesConfig(
    # Quantization config to ensure full GPU load
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    start = time.time()

    model_id = "microsoft/Phi-4-mini-instruct"
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loaded model in {time.time() - start:.2f} sec")

    prompt = f"""Extract the recipe as JSON from the webpage HTML:
    # Prompt
    
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
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"Generating... (prompt tokens: {inputs.input_ids.numel()})")
    # START timing
    start = time.time()

    outputs = model.generate(
        **inputs,
        max_new_tokens=2400,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,

    )

    print(f"Processed inference in {time.time() - start:.2f} sec")

    decode = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decode)

if __name__ == "__main__":
    main()
