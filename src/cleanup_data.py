import os
import json
import argparse
from collections import Counter
from pathlib import Path
from bs4 import BeautifulSoup

def find_doubled_ingredients(labels_dir):
    doubled_files = []
    for filename in os.listdir(labels_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(labels_dir, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                ingredients = data.get('ingredients', [])
                counts = Counter(ingredients)
                # Check if any ingredient appears more than once (excluding empty or '0')
                doubled = any(count > 1 for ing, count in counts.items() if ing and ing != '0')
                if doubled:
                    doubled_files.append(filename)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    return doubled_files

def get_html_path(json_filename, html_dir):
    base = os.path.splitext(json_filename)[0]
    html_name = base + '.html'

    html_path = os.path.join(html_dir, html_name)
    if os.path.exists(html_path):
        return html_path

    return None

def html_missing_content(html_path, label_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
        body_text = soup.body.get_text(separator=' ', strip=True).lower() if soup.body else ''
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        name = data.get('name', '').lower()
        ingredients = [i.lower() for i in data.get('ingredients', []) if i]
        directions = [d.lower() for d in data.get('directions', []) if d]
        # Check if any ingredient, name, or direction is in the body text
        found = False
        for item in [name] + ingredients + directions:
            if item and item in body_text:
                found = True
                break
        return not found
    except Exception as e:
        print(f"Error processing {html_path} or {label_path}: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Find and optionally delete files with doubled ingredients.')
    parser.add_argument('--dry-run', action='store_true', help='Only display files, do not delete.')
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / 'data'
    labels_dir = os.path.join(data_dir, 'labels')
    html_dir =os.path.join(data_dir, 'html_pages')

    doubled = find_doubled_ingredients(labels_dir)
    # Find doubled ingredient JSONs
    doubled_jsons = set(doubled)

    # Find JSONs whose HTML is missing content
    html_missing_jsons = set()
    html_missing_pairs = []
    for html_file in os.listdir(html_dir):
        if html_file.endswith('.html'):
            html_path = os.path.join(html_dir, html_file)
            label_file = os.path.splitext(html_file)[0] + '.json'
            label_path = os.path.join(labels_dir, label_file)
            if os.path.exists(label_path):
                if html_missing_content(html_path, label_path):
                    html_missing_jsons.add(label_file)
                    html_missing_pairs.append((label_file, html_path))

    # Prepare output/deletion list with reasons
    to_process = []
    for json_file in doubled_jsons:
        json_path = os.path.join(labels_dir, json_file)
        html_path = get_html_path(json_file, html_dir)
        to_process.append((json_path, html_path, 'doubled ingredients'))
    for json_file, html_path in html_missing_pairs:
        json_path = os.path.join(labels_dir, json_file)
        # Avoid duplicates if already in doubled
        if json_file not in doubled_jsons:
            to_process.append((json_path, html_path, 'html missing recipe'))

    # Output or delete
    doubled_count = sum(1 for _, _, reason in to_process if reason == 'doubled ingredients')
    html_missing_count = sum(1 for _, _, reason in to_process if reason == 'html missing recipe')
    total_count = len(to_process)

    if to_process:
        if args.dry_run:
            print(f"[DRY RUN] {total_count} files to process:")
            for json_path, html_path, reason in to_process:
                print(f"Reason: {reason}  JSON: {json_path} (exists: {os.path.exists(json_path)})  HTML: {html_path} (exists: {os.path.exists(html_path)})")
            print("-" * 40)
            print(f"Total doubled ingredients: {doubled_count}")
            print(f"Total html missing recipe: {html_missing_count}")
            print(f"Total files: {total_count}")
        else:
            print(f"Deleting {total_count} files:")
            for json_path, html_path, reason in to_process:
                print(f"Reason: {reason}")
                if os.path.exists(json_path):
                    os.remove(json_path)
                    print(f"  Deleted JSON: {json_path}")
                else:
                    print(f"  JSON not found: {json_path}")
                if html_path and os.path.exists(html_path):
                    os.remove(html_path)
                    print(f"  Deleted HTML: {html_path}")
                elif html_path:
                    print(f"  HTML not found: {html_path}")
            print(f"Total doubled ingredients: {doubled_count}")
            print(f"Total html missing recipe: {html_missing_count}")
            print(f"Total files: {total_count}")
    else:
        print("No files to process.")
