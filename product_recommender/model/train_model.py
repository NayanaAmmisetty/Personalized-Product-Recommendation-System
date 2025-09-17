# model/train_model.py
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]  # Project root
DATA_PATH = ROOT / "data" / "dummy_user_product_interactions.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Data not found at {DATA_PATH}. Run generate_data.py first.")

print("Loading data...")
df = pd.read_csv(DATA_PATH)

# Encode users/products as integer indices
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()
df["user_idx"] = user_encoder.fit_transform(df["user_id"])
df["product_idx"] = product_encoder.fit_transform(df["product_id"])

# Product-level features: average rating, popularity
product_stats = (
    df.groupby("product_idx")
      .agg(avg_rating=("rating", "mean"),
           popularity=("rating", "count"))
      .reset_index()
      .sort_values("product_idx")
      .reset_index(drop=True)
)

product_features = product_stats[["avg_rating", "popularity"]].values

# Train k-NN for candidate generation
print("Training k-NN (candidate generation)...")
knn = NearestNeighbors(n_neighbors=10, metric="cosine")
knn.fit(product_features)

joblib.dump({
    "knn": knn,
    "product_encoder": product_encoder,
    "product_stats": product_stats
}, MODEL_DIR / "knn_pipeline.pkl")
print("Saved knn_pipeline.pkl")

# Prepare training data for LightGBM (re-ranking)
print("Preparing LightGBM training data...")

# Define positives and negatives
positive = df[df["rating"] >= 4]
all_product_idxs = df["product_idx"].unique()

rows = []
for user in df["user_idx"].unique():
    pos_items = positive[positive["user_idx"] == user]["product_idx"].tolist()
    interacted = df[df["user_idx"] == user]["product_idx"].tolist()
    non_interacted = list(set(all_product_idxs) - set(interacted))
    if pos_items:
        # Negative sampling: up to 2x positives if possible
        neg_count = min(len(pos_items) * 2, len(non_interacted))
        if neg_count > 0:
            neg_sampled = list(np.random.choice(non_interacted, size=neg_count, replace=False))
        else:
            neg_sampled = []
    else:
        neg_sampled = []

    # Add positive labels
    for p in pos_items:
        rows.append([user, p, 1])
    # Add negative labels
    for p in neg_sampled:
        rows.append([user, p, 0])

train_df = pd.DataFrame(rows, columns=["user_idx", "product_idx", "label"])
train_df = train_df.merge(product_stats, on="product_idx", how="left")
user_mean = df.groupby("user_idx")["rating"].mean().reset_index().rename(columns={"rating": "user_mean_rating"})
train_df = train_df.merge(user_mean, on="user_idx", how="left")

FEATURES = ["avg_rating", "popularity", "user_mean_rating"]
X = train_df[FEATURES]
y = train_df["label"]

# Split into train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "seed": 42
}

print("Training LightGBM re-ranker...")
# Use callbacks for early stopping and evaluation logging
callbacks = [
    lgb.early_stopping(stopping_rounds=20),
    lgb.log_evaluation(period=10)
]

model = lgb.train(
    params,
    train_set=lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=200,
    callbacks=callbacks
)

# Save model and metadata
model.save_model(MODEL_DIR / "lightgbm_model.txt")
joblib.dump({
    "user_encoder": user_encoder,
    "product_encoder": product_encoder,
    "FEATURES": FEATURES
}, MODEL_DIR / "meta.pkl")
print("Saved LightGBM model and metadata.")
