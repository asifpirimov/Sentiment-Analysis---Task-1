import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import clean_text

DEFAULT_LABEL_MAP = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{csv_path.name} must contain columns: text, label")
    df["text"] = df["text"].astype(str)
    return df


def basic_eda(df: pd.DataFrame, name: str) -> None:
    print(f"\n--- EDA: {name} ---")
    print("Shape:", df.shape)

    print("\nLabel counts:")
    print(df["label"].value_counts(dropna=False))

    lengths = df["text"].astype(str).apply(len)
    print("\nText length stats:")
    print("  min:", int(lengths.min()))
    print("  max:", int(lengths.max()))
    print("  mean:", round(float(lengths.mean()), 2))


def build_label_map(y_train):
    try:
        y_int = pd.Series(y_train).astype(int)
        unique_int = sorted(y_int.unique().tolist())
        if unique_int == [0, 1, 2, 3, 4, 5]:
            return DEFAULT_LABEL_MAP
        return {i: str(i) for i in unique_int}
    except Exception:
        return None 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/training.csv")
    parser.add_argument("--val", type=str, default="data/validation.csv")
    parser.add_argument("--outdir", type=str, default="artifacts")
    args = parser.parse_args()

    train_path = Path(args.train)
    val_path = Path(args.val)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)

    #EDA
    basic_eda(train_df, "TRAIN")
    basic_eda(val_df, "VALIDATION")

    #text preprocessing
    train_df["clean_text"] = train_df["text"].apply(clean_text)
    val_df["clean_text"] = val_df["text"].apply(clean_text)

    #features
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=2
    )

    X_train = vectorizer.fit_transform(train_df["clean_text"])
    X_val = vectorizer.transform(val_df["clean_text"])

    #labels
    y_train_raw = train_df["label"]
    y_val_raw = val_df["label"]

    label_map = build_label_map(y_train_raw)

    if label_map is not None:
        y_train = y_train_raw.astype(int)
        y_val = y_val_raw.astype(int)
        target_names = [label_map[i] for i in sorted(label_map.keys())]
        labels_order = sorted(label_map.keys())
    else:
        y_train = y_train_raw.astype(str)
        y_val = y_val_raw.astype(str)
        target_names = sorted(y_train.unique().tolist())
        labels_order = target_names

    #model
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    #evaluation
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)

    print("\n--- RESULTS (Validation) ---")
    print("Accuracy:", round(acc, 4))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_val, preds, labels=labels_order))

    print("\nClassification Report:")
    # classification_report expects labels and target_names aligned
    if label_map is not None:
        print(classification_report(y_val, preds, labels=labels_order, target_names=target_names))
    else:
        print(classification_report(y_val, preds))

    # saving artifacts for later prediction
    joblib.dump(vectorizer, outdir / "tfidf.joblib")
    joblib.dump(model, outdir / "model.joblib")
    joblib.dump(label_map, outdir / "label_map.joblib")

    print(f"\nSaved to: {outdir.resolve()}")
    print("Artifacts: tfidf.joblib, model.joblib, label_map.joblib")


if __name__ == "__main__":
    main()
