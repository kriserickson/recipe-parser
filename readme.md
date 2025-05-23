# Recipe Extractor

This project builds a supervised learning system that classifies text segments in raw HTML recipe pages into structured components: **title**, **ingredients**, **directions**, and **image links**.

## 📁 Project Structure

```
recipe_extractor/
├── data/
│   ├── html_pages/           # Raw HTML files
│   ├── labels/               # JSON files with labeled data (title, ingredients, etc.)
│   ├── potential_labels/     # Raw candidates needing validation
│   └── processing_state.json # Tracks state of scraping and validation
├── models/                   # Trained model (.joblib)
├── src/
│   ├── __init__.py
│   ├── html_parser.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── notebooks/
│   └── exploration.ipynb
├── run_train.sh              # CLI runner
└── README.md
```

## 🚀 Getting Started

```bash
pip install -r requirements.txt
chmod +x run_train.sh
./run_train.sh
```

## 🧠 How It Works

- `validate_and_filter_recipes.py` filters usable recipes by confirming if text in the labeled JSON appears in the HTML.
- `html_parser.py` extracts tag, depth, and text blocks from HTML.
- `feature_extraction.py` uses TF-IDF to vectorize.
- `train.py` builds and trains a Logistic Regression model.
- `predict.py` applies the model to classify HTML blocks.
- `evaluate.py` prints metrics on test data.

## 🧪 Example
```bash
python src/predict.py ../data/html_pages/recipe_00001.html
```

## 📝 License
Apache-2.0 License

