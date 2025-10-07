# ml/update_model.py
import json, os
import pandas as pd

FB = os.path.join(os.path.dirname(__file__), "..", "data", "user_feedback.csv")
OUT = os.path.join(os.path.dirname(__file__), "..", "models", "genre_weights.json")

def main():
    if not os.path.exists(FB) or os.path.getsize(FB) == 0:
        print("No feedback yet.")
        return
    df = pd.read_csv(FB)
    if not {"model_served","prediction","was_helpful"}.issubset(df.columns):
        print("CSV missing columns.")
        return
    # Dummy heuristic: help rate by predicted label (age_range proxy),
    # but you can pivot on genre if you log it in the UI payload.
    weights = df.groupby("prediction")["was_helpful"].mean().to_dict()
    # normalize to [0.5, 1.5] around mean 1.0
    import numpy as np
    mean = np.mean(list(weights.values())) if weights else 0.5
    scaled = {k: 1.0 + (v - mean) for k,v in weights.items()}
    with open(OUT, "w") as f:
        json.dump(scaled, f, indent=2)
    print("Wrote", OUT, "→", scaled)

if __name__ == "__main__":
    main()
