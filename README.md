# Personalized Product Recommendation System (2025)

> Hybrid recommender: candidate generation via k-NN + LightGBM re-ranking.  
> Includes a Flask API backend and a Streamlit UI frontend. Dummy dataset and training pipeline included so you can run the project end-to-end.

---

## Table of contents

- [Project overview](#project-overview)  
- [Features](#features)  
- [Tech stack & concepts](#tech-stack--concepts)  
- [Repository structure](#repository-structure)  
- [Dataset (dummy)](#dataset-dummy)  
- [Modeling details](#modeling-details)  
  - [Candidate generation — k-NN](#candidate-generation---k-nn)  
  - [Re-ranking — LightGBM](#re-ranking---lightgbm)  
  - [Training & evaluation](#training--evaluation)  
- [API documentation](#api-documentation)  
- [Run locally (development)](#run-locally-development)  
- [Streamlit UI usage](#streamlit-ui-usage)  
- [Production & deployment notes](#production--deployment-notes)  
- [Security & privacy notes](#security--privacy-notes)  
- [Possible improvements](#possible-improvements)  
- [Troubleshooting](#troubleshooting)  
- [Contributing](#contributing)  
- [License](#license)

---

## Project overview

This repository demonstrates a **hybrid recommendation pipeline**:

1. **Candidate generation** using *k-Nearest Neighbors* (k-NN) on product embeddings to quickly propose similar or related products.
2. **Re-ranking** the candidate list using a supervised gradient boosting model (LightGBM) to predict the probability of a user engaging or purchasing — producing the final ordered recommendations.
3. **Serving** recommendations through a small **Flask** API.
4. **Visualization & interaction** via a **Streamlit** UI that calls the API for real-time recommendations.
5. Includes a **dummy dataset generator** so the system is runnable out-of-the-box.

This pattern (fast, approximate candidate generation + learned re-ranking) is common in production recommender systems because it balances scalability and accuracy.

---

## Features

- Synthetic user-product interaction dataset (reproducible)
- k-NN candidate generation (fast nearest neighbors)
- LightGBM re-ranking using simple user & product features
- Flask REST endpoint to serve recommendations
- Streamlit front-end to explore and visualize recommendations
- Instructions and scripts to train & save models locally

---

## Tech stack & concepts

**Backend**
- **Python** — primary language
- **Flask** — lightweight web server to expose recommendation endpoints

**Modeling**
- **scikit-learn** — utilities & k-NN (candidate generation)
- **LightGBM** — fast gradient-boosted trees for re-ranking

**Frontend**
- **Streamlit** — declarative Python UI to interact with the system

**Data**
- CSV file (dummy dataset for reproducibility) — simple, human-readable format for a demo

**Key concepts**
- **Candidate generation**: produce a manageable set of product candidates per user (e.g., ~10-200) using scalable heuristic methods (k-NN, item-to-item, TF-IDF or ANN).
- **Re-ranking**: run a learned model over candidates to produce a final ordering personalized by user features and context.
- **Offline vs Online**: training/test done offline on historical data; Flask API performs online inference per request.

---

## Repository structure (recommended)


---

## Dataset

This repo includes `generate_data.py` which generates `data/dummy_user_product_interactions.csv`. The CSV has the following schema:

- `user_id` — user identifier (e.g., `U1`)
- `product_id` — product identifier (e.g., `P7`)
- `rating` — interaction label (e.g., explicit rating 1–5, or binary 0/1 for purchase)
- Additional features (optional): user age, product price, category, timestamp

**Why a dummy CSV?**  
- Reproducibility for examples, demos and unit tests.
- Easy to replace with your real dataset later.

---

## Modeling details

### Candidate generation — k-NN
**Purpose:** quickly narrow down millions of products to a small candidate set per user or per product.

**How it's implemented in this demo:**
- We build a simple product feature vector (in the toy example this may be `price` or handcrafted features).
- Fit `sklearn.neighbors.NearestNeighbors` on product embeddings.
- For a product that a user recently interacted with, we query `kneighbors()` to obtain `k` nearest products (top-N similar items).

**Conceptual notes:**
- In production, product embeddings come from collaborative filtering (ALS), matrix factorization, item2vec, or neural embeddings.
- For large catalogs use approximate nearest neighbor (ANN) libraries: FAISS, Annoy, HNSWlib.

---

### Re-ranking — LightGBM
**Purpose:** Learn to order candidate items by the likelihood the target user will interact/purchase.

**Features used:**
- User features: age, segment, recency attributes
- Product features: price, category, popularity
- Context features: time of day, device (if available)
- Interaction features: historical user-product interactions (e.g., count, last seen)

**Training approach in demo:**
- Use positive samples (interactions/purchases) and negative sampling (products the user didn't interact with).
- Train LightGBM binary classifier to predict label (1 if interacted, 0 otherwise).
- At inference, produce a `score = model.predict(features)` to rank candidates.

**Why LightGBM?**
- Fast to train & serve, handles categorical features well (with encoding), robust to heterogeneous features.

---

### Training & evaluation

**Offline evaluation metrics to use:**
- **Precision@K / Recall@K** — how many top-K recommendations are relevant
- **NDCG@K** — rank-sensitive metric (gives higher weight to top positions)
- **AUC / Logloss** — for binary predictions

**Typical training loop:**
1. Build dataset with user-product label (1 positive, many negatives).
2. Split by user/time into train/validation/test (avoid leaking future interactions).
3. Fit LightGBM with early stopping on validation.
4. Evaluate with `precision@K` and `ndcg@K` by generating candidate lists and re-ranking them.

**Offline vs online uplift:**  
When you claim improvements (e.g., ~30% accuracy boost), be explicit whether this is relative on your offline metric or on real-world A/B test results. Offline gains don’t always translate online.

---

## API documentation (Flask)

### Endpoint
**GET** `/recommend?user=<USER_ID>`

**Parameters**
- `user` (required): e.g., `U1`

**Response**
```json
[
  {
    "product": "P12",
    "price": 1499,
    "age": 32,
    "score": 0.8123
  },
  ...
]
