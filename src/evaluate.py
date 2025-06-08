# evaluate.py
# Evaluate the trained HTML block classifier against labeled JSON + HTML pairs

from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from feature_extraction import build_feature_pipeline, load_labeled_blocks
from config import LABEL_DIR, HTML_DIR

def evaluate() -> None:
    """
    Evaluate the HTML block classifier using labeled data.
    
    Performs the following steps:
    1. Loads labeled blocks
    2. Extracts features
    3. Splits data into training and test sets
    4. Trains a logistic regression model
    5. Prints classification report
    """
    # Load labeled HTML text chunks and their true labels
    X_raw, y = load_labeled_blocks(LABEL_DIR, HTML_DIR, limit=100)

    # Convert list of dicts to DataFrame if needed
    X_features = [str(f) for f in X_raw]  # Ensure X_features is a list of strings

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # Build model pipeline
    model = make_pipeline(
        build_feature_pipeline(),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate()