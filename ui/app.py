import os
import uuid
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import streamlit as st

API_URL = os.environ.get("BOOKSENSE_API", "http://127.0.0.1:8000")

st.set_page_config(page_title="BookSense", layout="centered")
st.title("📚 BookSense — Adaptive Reading Recommender")
st.caption(f"API URL: {API_URL}")

# ---------- Paths
DATA_DIR = (Path(__file__).resolve().parent / ".." / "data").resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
FB_PATH = DATA_DIR / "user_feedback.csv"

# ---------- Caching helpers
@st.cache_resource
def get_http_session():
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s

@st.cache_data(ttl=5)  # refresh every 5s when viewing Analytics tab
def load_feedback_csv(path: Path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp","session_id","model_served","prediction","was_helpful"])
    return pd.read_csv(path)

def log_feedback(helpful: bool, served: str, pred: str) -> None:
    row = ",".join([
        datetime.utcnow().isoformat(),
        str(uuid.uuid4()),
        str(served),
        str(pred),
        "1" if helpful else "0",
    ]) + "\n"
    header_needed = (not FB_PATH.exists()) or (FB_PATH.stat().st_size == 0)
    with FB_PATH.open("a", encoding="utf-8") as f:
        if header_needed:
            f.write("timestamp,session_id,model_served,prediction,was_helpful\n")
        f.write(row)

# ---------- Tabs
tab_predict, tab_analytics = st.tabs(["🔮 Predict", "📈 Analytics"])

with tab_predict:
    with st.form("input_form", clear_on_submit=False):
        title = st.text_input("Title (optional, improves accuracy)", value="")
        genre = st.selectbox("Genre", ["Fiction","Non-Fiction","Fantasy","Science","History","Biography"], index=0)
        pages = st.number_input("Pages", min_value=10, max_value=1200, value=200, step=10)
        complexity = st.slider("Complexity (0 easy → 1 hard)", 0.0, 1.0, 0.5, 0.01)
        rating = st.slider("Average Rating (0–5)", 0.0, 5.0, 4.0, 0.1)
        model_choice = st.selectbox("Model", ["auto","sklearn","torch"], index=0)
        submitted = st.form_submit_button("Predict Reading Level", use_container_width=True)

    # API button to verify connectivity (handy while debugging)
    cols = st.columns(2)
    if cols[0].button("Test API /health", use_container_width=True):
        try:
            r = get_http_session().get(f"{API_URL}/health", timeout=3)
            st.info(f"/health → {r.status_code} {r.text[:120]}")
        except Exception as e:
            st.error(f"Could not reach API: {e}")

    if submitted:
        payload = {
            "title": title,
            "genre": genre,
            "pages": int(pages),
            "complexity": float(complexity),
            "rating": float(rating),
        }
        try:
            with st.spinner("Contacting API…"):
                r = get_http_session().post(f"{API_URL}/predict", json=payload, params={"model": model_choice}, timeout=5)
            if r.status_code == 200:
                body = r.json()
                if "prediction" in body:
                    pred = body["prediction"]
                    conf = body.get("confidence")
                    served = body.get("model_served", model_choice)
                    st.session_state["last_pred"] = pred
                    st.session_state["last_served"] = served
                    conf_str = f" ({conf:.3f})" if isinstance(conf, (int, float)) else ""
                    st.success(f"Predicted Level: {pred}{conf_str} — model: {served}")
                else:
                    st.error(f"API returned: {body.get('error','Unknown error')}")
            else:
                st.error(f"API error {r.status_code}: {r.text[:200]}")
        except Exception as e:
            st.error(f"Failed to reach API at {API_URL}. Error: {e}")

    # Feedback (persists until logged)
    if "last_pred" in st.session_state and "last_served" in st.session_state:
        st.markdown("### Was this helpful?")
        c1, c2 = st.columns(2)
        if c1.button("👍 Helpful", key="fb_yes", use_container_width=True):
            log_feedback(True, st.session_state["last_served"], st.session_state["last_pred"])
            st.info("Thanks for the feedback — logged!")
            del st.session_state["last_pred"]; del st.session_state["last_served"]
        if c2.button("👎 Not helpful", key="fb_no", use_container_width=True):
            log_feedback(False, st.session_state["last_served"], st.session_state["last_pred"])
            st.info("Thanks for the feedback — logged!")
            del st.session_state["last_pred"]; del st.session_state["last_served"]

with tab_analytics:
    df = load_feedback_csv(FB_PATH)
    if len(df) == 0:
        st.info("No feedback yet. Use the Predict tab and click 👍/👎.")
    else:
        left, right = st.columns(2)
        with left:
            st.metric("Total Feedback", len(df))
        with right:
            rate = (df["was_helpful"].mean() * 100.0) if "was_helpful" in df else 0.0
            st.metric("Helpfulness Rate", f"{rate:.0f}%")
        if {"model_served","was_helpful"}.issubset(df.columns):
            st.bar_chart(df.groupby("model_served")["was_helpful"].mean())
