# backend/app.py
from flask import Flask, request, jsonify
from pathlib import Path
import joblib
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]  # project root
MODEL_DIR = ROOT / "model"
DATA_CSV = ROOT / "data" / "dummy_user_product_interactions.csv"

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"Model directory not found at {MODEL_DIR}. Run training first.")
if not DATA_CSV.exists():
    raise FileNotFoundError(f"Data file not found at {DATA_CSV}. Run generate_data.py first.")

# Load artifacts
knn_artifact = joblib.load(MODEL_DIR / "knn_pipeline.pkl")
knn = knn_artifact["knn"]
product_encoder = knn_artifact["product_encoder"]
product_stats = knn_artifact["product_stats"]
lgb_model = lgb.Booster(model_file=str(MODEL_DIR / "lightgbm_model.txt"))
meta = joblib.load(MODEL_DIR / "meta.pkl")
FEATURES = meta["FEATURES"]

# Load data
df = pd.read_csv(DATA_CSV)

# Build simple mappings
users = sorted(df["user_id"].unique().tolist())
products = sorted(df["product_id"].unique().tolist())

app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    user_id = request.args.get("user")
    if not user_id:
        return jsonify({"error": "Missing 'user' parameter"}), 400
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404

    user_history = df[df["user_id"] == user_id]
    if user_history.empty:
        return jsonify({"error": "No history for user"}), 404

    recent_product = user_history.iloc[-1]["product_id"]
    try:
        prod_idx = product_encoder.transform([recent_product])[0]
    except Exception:
        prod_idx = 0

    # Build feature array from product_stats for knn query
    # product_stats must contain rows indexed by product_idx
    pf = product_stats[["avg_rating", "popularity"]].values
    distances, indices = knn.kneighbors([pf[prod_idx]], n_neighbors=min(10, len(pf)))
    candidate_idxs = indices.flatten().tolist()

    candidates = []
    user_mean = float(user_history["rating"].mean())
    for idx in candidate_idxs:
        try:
            product_id = product_encoder.inverse_transform([idx])[0]
        except Exception:
            product_id = f"P{idx+1}"
        prod_row = product_stats.iloc[idx]
        avg_rating = float(prod_row["avg_rating"])
        popularity = int(prod_row["popularity"])
        fv = pd.DataFrame([{
            "avg_rating": avg_rating,
            "popularity": popularity,
            "user_mean_rating": user_mean
        }])
        score = float(lgb_model.predict(fv[FEATURES]))
        candidates.append({
            "product_id": product_id,
            "avg_rating": avg_rating,
            "popularity": popularity,
            "score": score
        })

    candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    return jsonify(candidates)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
