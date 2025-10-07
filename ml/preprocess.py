"""
Preprocessing utilities for BookSense.

Expected columns in data/books.csv:
    - title (str): book title (free text)
    - genre (str): categorical (e.g., Fiction, Non-Fiction)
    - pages (int): number of pages
    - complexity (float): readability/complexity score [0,1]
    - age_range (str): target age (label to predict), e.g. "7-9"
    - rating (float): avg rating [0,5]

Outputs:
    - X (features), y (labels), fitted encoders/scalers
"""
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

TEXT = ["title"]
CATEGORICAL = ["genre"]
NUMERIC = ["pages", "complexity", "rating"]
LABEL = "age_range"

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    # Basic cleaning
    df = df.dropna(subset=TEXT + CATEGORICAL + NUMERIC + [LABEL])
    return df

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    # Note: ColumnTransformer can pass a single column (string) to TfidfVectorizer
    pre = ColumnTransformer([
        ("txt", TfidfVectorizer(max_features=200, ngram_range=(1,1)), "title"),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", StandardScaler(), NUMERIC),
    ], remainder="drop", verbose_feature_names_out=False)
    return pre

def make_xy(df: pd.DataFrame):
    X = df[TEXT + CATEGORICAL + NUMERIC]
    y = df[LABEL]
    return X, y

def label_encoder_fit(y):
    # Simple string-to-index mapping
    classes = sorted(y.unique())
    to_idx = {c: i for i, c in enumerate(classes)}
    to_lbl = {i: c for c, i in to_idx.items()}
    y_idx = y.map(to_idx)
    return y_idx, to_idx, to_lbl
