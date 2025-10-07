# api/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import os
import joblib

# --------------------
# App setup
# --------------------
app = FastAPI(title="BookSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Optional model loading (do not crash if missing)
# --------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
SK_PATH = os.path.join(BASE_DIR, "models", "sk_model.pkl")
TORCH_W = os.path.join(BASE_DIR, "models", "torch_model.pt")
TORCH_AUX = os.path.join(BASE_DIR, "models", "torch_aux.joblib")

sk_payload = None
torch_model = None
torch_aux = None

# Torch model class (kept tiny here)
try:
    import torch
    import torch.nn as nn

    class TorchMLP(nn.Module):
        def __init__(self, n_features, n_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, n_classes),
            )
        def forward(self, x):  # noqa: D401
            return self.net(x)

except Exception:
    torch = None
    nn = None
    TorchMLP = None  # type: ignore


def _safe_load_models():
    """Load models if present; ignore failures so routes still register."""
    global sk_payload, torch_model, torch_aux
    try:
        if os.path.exists(SK_PATH):
            sk_payload = joblib.load(SK_PATH)
    except Exception:
        sk_payload = None

    try:
        if torch and os.path.exists(TORCH_W) and os.path.exists(TORCH_AUX):
            torch_aux = joblib.load(TORCH_AUX)
            n_features = torch_aux["n_features"]
            n_classes = torch_aux["n_classes"]
            tm = TorchMLP(n_features, n_classes)
            tm.load_state_dict(torch.load(TORCH_W, map_location="cpu"))
            tm.eval()
            torch_model = tm
    except Exception:
        torch_model = None


_safe_load_models()

# --------------------
# Schemas
# --------------------
class BookFeatures(BaseModel):
    title: Optional[str] = None
    genre: str
    pages: int
    complexity: float
    rating: float


# --------------------
# Helpers
# --------------------
def _predict_sklearn(feat: dict):
    import pandas as pd
    pipe = sk_payload["pipeline"]
    to_lbl = sk_payload["label_map"]
    X = pd.DataFrame([{
        "title": feat.get("title", "") or "",
        "genre": feat["genre"],
        "pages": feat["pages"],
        "complexity": feat["complexity"],
        "rating": feat["rating"],
    }])
    pred_idx = int(pipe.predict(X)[0])
    try:
        proba = float(pipe.predict_proba(X).max())
    except Exception:
        proba = None
    return to_lbl[pred_idx], proba



def _predict_torch(feat: dict):
    import numpy as np
    import pandas as pd
    import torch as T
    pre = torch_aux["preprocessor"]
    to_lbl = torch_aux["label_map"]
    Xdf = pd.DataFrame([{
        "title": feat.get("title", "") or "",
        "genre": feat["genre"],
        "pages": feat["pages"],
        "complexity": feat["complexity"],
        "rating": feat["rating"],
    }])
    X_enc = pre.transform(Xdf)
    X_enc = X_enc.toarray() if hasattr(X_enc, "toarray") else X_enc
    x = T.tensor(np.array(X_enc), dtype=T.float32)
    with T.no_grad():
        logits = torch_model(x)
        probs = T.softmax(logits, dim=1).numpy().squeeze()
        pred = int(probs.argmax())
        conf = float(probs.max())
    return to_lbl[pred], conf



# --------------------
# Routes
# --------------------
@app.get("/")
def root():
    return {"ok": True, "see": ["/health", "/docs", "/predict"]}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sklearn_loaded": sk_payload is not None,
        "torch_loaded": torch_model is not None,
    }


@app.post("/predict")
def predict(features: BookFeatures, model: str = "auto"):
    feat = features.model_dump()
    if model == "sklearn":
        if not sk_payload:
            return {"error": "sklearn model not available. Train first."}
        label, conf = _predict_sklearn(feat)
        return {"model_served": "sklearn", "prediction": label, "confidence": conf}

    if model == "torch":
        if not torch_model or not torch_aux:
            return {"error": "torch model not available. Train first."}
        label, conf = _predict_torch(feat)
        return {"model_served": "torch", "prediction": label, "confidence": conf}

    # Auto: prefer both, otherwise whichever is available
    import random
    options = []
    if sk_payload:
        options.append("sklearn")
    if torch_model and torch_aux:
        options.append("torch")
    if not options:
        return {"error": "No models available. Train first."}
    chosen = random.choice(options)
    return predict(features, model=chosen)  # recurse with explicit choice
