# generate_data.py
import pandas as pd
import numpy as np
import os
from pathlib import Path

np.random.seed(42)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

users = [f"U{i}" for i in range(1, 51)]
products = [f"P{i}" for i in range(1, 21)]

rows = []
for u in users:
    n = np.random.randint(5, 11)
    sampled = np.random.choice(products, n, replace=False)
    for p in sampled:
        rating = np.random.randint(1, 6)
        rows.append([u, p, rating])

df = pd.DataFrame(rows, columns=["user_id", "product_id", "rating"])
out_path = DATA_DIR / "dummy_user_product_interactions.csv"
df.to_csv(out_path, index=False)
print(f"Saved {out_path}")
