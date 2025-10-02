# RAG - Retrieval-Augmented Generation

This branch of the Recipe Parse focus on RAG (Retrieval-Augmented Generation) techniques, for the article [RAG Time: Cooking Up Smart Recipe Suggestions with AI and a Dash of Retrieval](https://agingcoder.com/posts/rag-time-cooking-up-smart-recipe-suggestions/]

## 📁 Project Structure and File Overview

### src/
- `__init__.py`: Marks the directory as a Python package.
- `convert_bson.py`: Script or module for converting BSON data, likely used for data import/export.
- `suggest_service.py`: Provides suggestion or recommendation services, possibly for recipe or ingredient suggestions.

### static/
- `index.html`: Main HTML file for the web frontend.
- `main.js`: JavaScript file for frontend interactivity and logic.
- `styles.css`: CSS file for styling the web interface.

### notebooks/
- `rag.ipynb`: Jupyter notebook focused on Retrieval-Augmented Generation experiments, demonstrations, or documentation.

### data/potential_labels/
- `recipe_00005.json`, `recipe_000XX.json`: JSON files containing potential labels for recipes, such as ingredients, directions, and metadata. Each file corresponds to a specific recipe and includes fields like `title`, `image`, `keywords`, `directions`, and `ingredients`.

---

- The `src/` directory contains backend and data processing code.
- The `static/` directory contains all frontend assets for the web interface.
- The `notebooks/` directory is for interactive development, prototyping, and documentation using Jupyter notebooks.
- The `data/potential_labels/` directory holds structured data for recipes, used for training, evaluation, or demonstration purposes.

For more details on usage, setup, and contributing, see the referenced article.

## 📝 License
Apache-2.0 License
