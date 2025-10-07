
# BookSense — One-Week Adaptive Reading Recommender

**Goal:** Build a compact, portfolio-ready ML project in 7 days:
- Train baseline (scikit-learn) and deep (PyTorch) models.
- Serve predictions via **FastAPI**.
- Build an interactive **Streamlit** UI.
- Run an **A/B test** and log **user feedback**.
- Visualize analytics.

## Project Structure
```
BookSense/
├── api/                # FastAPI backend
├── analysis/           # Notebooks: A/B tests + analytics
├── data/               # Datasets + feedback logs
├── ml/                 # Training & preprocessing scripts
├── models/             # Saved models
├── report/             # Write-up / summary
├── scripts/            # Convenience scripts
├── ui/                 # Streamlit app
├── .github/workflows/  # CI (lint/test)
├── .gitignore
├── requirements.txt
└── README.md
```

## Quickstart

```bash
# 1) Create a virtual env (optional)
python3 -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Train baseline + torch model
python ml/train_sklearn.py
python ml/train_torch.py

# 4) Run API
uvicorn api.main:app --reload

# 5) Run UI (in a new terminal)
streamlit run ui/app.py
```

## A/B Testing & Analytics
- Use `analysis/ab_testing.ipynb` to compare **scikit-learn vs PyTorch**.
- Use `analysis/user_analytics.ipynb` to plot feedback trends.

## Notes
- Put your dataset in `data/books.csv` (see sample schema in file header).
- Feedback is appended to `data/user_feedback.csv`.
