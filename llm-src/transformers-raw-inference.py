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

DEFAULT_LLM_MODEL = "microsoft/Phi-4-mini-instruct"

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract recipe JSON from HTML using Phi-4.")
    parser.add_argument("html_file", type=str, help="HTML file to process (default dir: /data/html_pages)")
    parser.add_argument("--model", type=str, default=DEFAULT_LLM_MODEL,
                        help="HuggingFace model name (default: microsoft/phi-4-mini-instruct)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature for generation (default: 0.2)")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p (nucleus) sampling for generation (default: 0.95)")
    parser.add_argument("--max_tokens", type=int, default=2400,
                        help="Maximum number of new tokens to generate (default: 256)")
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

    prompt = f"""Extract the recipe ingredients, directions and title from the webpage HTML.  
    Respond with a single recipe using JSON format with the following keys: title, ingredients, directions.
    Do not return the example or any html, just the JSON response.

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

    bnb_config = BitsAndBytesConfig(
    # Quantization config to ensure full GPU load
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    start = time.time()

    model_id = args.model
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loaded model in {time.time() - start:.2f} sec, model size: {model.get_memory_footprint() / 1024**2:.2f} MB device: {model.device}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print(f"Generating... (prompt tokens: {inputs.input_ids.numel()}) tensor: {inputs.input_ids.device}")

    # START timing
    start = time.time()

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,

        )

    print(f"Processed inference in {time.time() - start:.2f} sec")

    decode = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(decode)

if __name__ == "__main__":
    main()
