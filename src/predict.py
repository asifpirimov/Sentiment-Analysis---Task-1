import argparse
from pathlib import Path

import joblib

from src.preprocessing import clean_text


def predict_emotion(text: str, vectorizer, model, label_map):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]

    if label_map is not None:
        try:
            pred_int = int(pred)
            return label_map.get(pred_int, str(pred_int))
        except Exception:
            return str(pred)

    return str(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--artifacts", type=str, default="artifacts")
    args = parser.parse_args()

    art = Path(args.artifacts)
    vectorizer = joblib.load(art / "tfidf.joblib")
    model = joblib.load(art / "model.joblib")
    label_map = joblib.load(art / "label_map.joblib")

    emotion = predict_emotion(args.text, vectorizer, model, label_map)
    print("Text:", args.text)
    print("Predicted emotion:", emotion)


if __name__ == "__main__":
    main()
