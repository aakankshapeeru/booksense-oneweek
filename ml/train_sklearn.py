
import joblib, os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from preprocess import load_dataset, build_preprocessor, make_xy, label_encoder_fit

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "books.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "sk_model.pkl")

def main():
    df = load_dataset(DATA_PATH)
    X, y = make_xy(df)
    y_idx, to_idx, to_lbl = label_encoder_fit(y)

    pre = build_preprocessor(df)

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X, y_idx)

    # Evaluate (using train set for quickstart; replace with CV)
    y_pred = pipe.predict(X)
    print(classification_report(y_idx, y_pred))
    print("F1 (macro):", f1_score(y_idx, y_pred, average="macro"))

    payload = {
        "pipeline": pipe,
        "label_map": to_lbl
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"Saved scikit-learn model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
