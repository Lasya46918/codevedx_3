# Project 3 — AI-Based Fake News Detection Tool

## Objective
Demonstrate an NLP classification pipeline that predicts a label for a news text and shows a model confidence score.

## Features
- Text input.
- TF-IDF vectorization.
- Logistic Regression classifier.
- Stratified train/test split and classification metrics.
- Saved model pipeline and confidence output.

## Run
```bash
pip install -r ../requirements.txt
python train_model.py
python main.py
```

## Critical limitation
The bundled dataset is synthetic and intentionally small for demonstrating code. It is **not** suitable for reliable real-world fake-news detection. The model learns patterns in its training examples and can confidently be wrong. Replace it with a credible, well-documented, properly licensed labelled dataset, evaluate across independent test data, and always verify claims using reliable sources.
