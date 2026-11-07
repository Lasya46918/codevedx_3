from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

HERE = Path(__file__).resolve().parent
DATA = HERE / "news_dataset.csv"
MODEL = HERE / "fake_news_model.joblib"

def main():
    data = pd.read_csv(DATA).dropna(subset=["text", "label"])
    if data["label"].nunique() < 2 or data["label"].value_counts().min() < 2:
        raise ValueError("Need at least two labels with at least two examples each.")
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"].astype(str), data["label"].astype(str),
        test_size=0.25, random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Holdout accuracy (tiny synthetic dataset; not meaningful as real-world evidence):",
          round(accuracy_score(y_test, pred), 3))
    print(classification_report(y_test, pred, zero_division=0))
    model.fit(data["text"].astype(str), data["label"].astype(str))
    joblib.dump(model, MODEL)
    print(f"Saved {MODEL.name}")

if __name__ == "__main__":
    main()
