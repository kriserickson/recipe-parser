import os
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    base_dir = Path(__file__).resolve().parent.parent
    html_dir = os.path.join(base_dir, 'data', 'html_pages')
    label_dir = os.path.join(base_dir, 'data', 'labels')
    output_dir = os.path.join(base_dir, 'data', 'jsonl')
    output_file = os.path.join(output_dir, 'train.jsonl')

    os.makedirs(output_dir, exist_ok=True)

    html_files = [f for f in os.listdir(html_dir) if f.startswith('recipe_') and f.endswith('.html')]
    count = 0
    total_files = len(html_files)
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for idx, html_file in enumerate(html_files):
            base = os.path.splitext(html_file)[0]
            label_file = base + '.json'
            html_path = os.path.join(html_dir, html_file)
            label_path = os.path.join(label_dir, label_file)
            if not os.path.exists(label_path):
                logging.warning(f'Missing label for {html_file}')
                continue
            try:
                with open(html_path, 'r', encoding='utf-8') as h_f:
                    html_content = h_f.read()
                with open(label_path, 'r', encoding='utf-8') as l_f:
                    label_content = json.load(l_f)
                block = {"text": html_content, "output": label_content}
                out_f.write(json.dumps(block, ensure_ascii=False) + '\n')
                count += 1
                if count % 500 == 0 or count == total_files:
                    percent = (count / total_files) * 100
                    print(f"Processed {count} of {total_files} files ({percent:.2f}%)")
            except Exception as e:
                logging.error(f'Error processing {html_file}: {e}')
    logging.info(f'Wrote {count} records to {output_file}')

if __name__ == '__main__':
    main()
