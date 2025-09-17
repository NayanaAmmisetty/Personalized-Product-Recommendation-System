# frontend/app.py
import streamlit as st
import pandas as pd
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root
DATA_CSV = ROOT / "data" / "dummy_user_product_interactions.csv"

st.set_page_config(page_title="Personalized Recommender", layout="wide")
st.title("🛍 Personalized Product Recommender")

if not DATA_CSV.exists():
    st.error(f"Data file not found at {DATA_CSV}. Run generate_data.py first.")
    st.stop()

df = pd.read_csv(DATA_CSV)
users = sorted(df["user_id"].unique().tolist())
selected_user = st.selectbox("Select user", users)

col1, col2 = st.columns(2)

with col1:
    st.subheader("User recent history")
    user_hist = df[df["user_id"] == selected_user].sort_index(ascending=False).head(10)
    st.table(user_hist)

with col2:
    st.subheader("Get recommendations")
    if st.button("Get Recommendations"):
        with st.spinner("Fetching recommendations from backend..."):
            try:
                resp = requests.get("http://127.0.0.1:5000/recommend", params={"user": selected_user}, timeout=10)
                if resp.status_code == 200:
                    recs = pd.DataFrame(resp.json())
                    st.subheader("Top Recommendations (re-ranked)")
                    st.table(recs)
                    st.markdown("**Top 3 Products**")
                    for i, row in recs.head(3).iterrows():
                        st.markdown(f"- **{row['product_id']}** (score: {row['score']:.4f}, avg_rating: {row['avg_rating']:.2f}, popularity: {row['popularity']})")
                else:
                    st.error(resp.json().get("error", "Unknown error"))
            except Exception as e:
                st.error(f"Request failed: {e}")
