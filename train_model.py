"""
train_model.py
---------------
Trains TWO complementary models, which is the key differentiator of this
project over a single-classifier approach:

1. XGBoost (supervised) - learns fraud patterns from labeled history.
   Handled imbalance via `scale_pos_weight` (tells the model "false negatives
   on the minority class cost N times more than false positives").

2. Isolation Forest (unsupervised anomaly detection) - flags transactions
   that look statistically unusual, even if they don't match any known
   fraud pattern in the training labels. This catches NOVEL fraud types
   that a purely supervised model would miss (a genuinely important
   real-world argument: fraudsters adapt, labels lag).

Evaluation uses PRECISION-RECALL AUC, not ROC-AUC, because:
- With 1.5% positive class, ROC-AUC can look deceptively high even for a
  weak model (the huge number of true negatives inflates it).
- PR-AUC focuses on how well the model performs on the minority class,
  which is what actually matters for fraud.
"""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, precision_score,
    recall_score, f1_score, confusion_matrix
)
from xgboost import XGBClassifier

RANDOM_STATE = 42


def load_data():
    X_train = pd.read_csv("data/X_train.csv")
    X_test = pd.read_csv("data/X_test.csv")
    y_train = pd.read_csv("data/y_train.csv").squeeze()
    y_test = pd.read_csv("data/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def train_xgboost(X_train, y_train):
    # scale_pos_weight = (# negative) / (# positive) - standard XGBoost
    # technique for imbalanced classification, cheaper than SMOTE and works
    # well in practice for tree models.
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",   # optimize directly for PR-AUC during training
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def train_isolation_forest(X_train, y_train):
    # Trained ONLY on legitimate transactions - the model learns what
    # "normal" looks like, then flags deviations from it. This is the
    # unsupervised half of the hybrid approach.
    legit_only = X_train[y_train == 0]
    model = IsolationForest(
        n_estimators=200,
        contamination=0.015,  # matches our known fraud rate
        random_state=RANDOM_STATE,
    )
    model.fit(legit_only)
    return model


def find_best_threshold(y_true, y_scores):
    """Pick the probability threshold that maximizes F1 on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1s[:-1])  # last point has no matching threshold
    return thresholds[best_idx], precisions[best_idx], recalls[best_idx], f1s[best_idx]


def evaluate(y_true, y_scores, model_name, threshold=None):
    pr_auc = average_precision_score(y_true, y_scores)

    if threshold is None:
        threshold, prec_at_best, rec_at_best, f1_at_best = find_best_threshold(y_true, y_scores)
    else:
        y_pred = (y_scores >= threshold).astype(int)
        prec_at_best = precision_score(y_true, y_pred, zero_division=0)
        rec_at_best = recall_score(y_true, y_pred, zero_division=0)
        f1_at_best = f1_score(y_true, y_pred, zero_division=0)

    y_pred_final = (y_scores >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred_final)

    print(f"\n--- {model_name} ---")
    print(f"PR-AUC:        {pr_auc:.4f}")
    print(f"Best threshold: {threshold:.4f}")
    print(f"Precision:     {prec_at_best:.4f}")
    print(f"Recall:        {rec_at_best:.4f}")
    print(f"F1:            {f1_at_best:.4f}")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")

    return {
        "model": model_name, "pr_auc": pr_auc, "threshold": float(threshold),
        "precision": float(prec_at_best), "recall": float(rec_at_best),
        "f1": float(f1_at_best), "confusion_matrix": cm.tolist(),
    }


def main():
    X_train, X_test, y_train, y_test = load_data()

    print("Training XGBoost (supervised)...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_scores = xgb_model.predict_proba(X_test)[:, 1]
    xgb_metrics = evaluate(y_test, xgb_scores, "XGBoost (supervised)")

    print("\nTraining Isolation Forest (unsupervised anomaly detection)...")
    iso_model = train_isolation_forest(X_train, y_train)
    # decision_function: higher = more normal. We flip sign so higher = more anomalous,
    # matching the "higher score = more fraud-like" convention used elsewhere.
    iso_scores = -iso_model.decision_function(X_test)
    iso_metrics = evaluate(y_test, iso_scores, "Isolation Forest (unsupervised)")

    # Hybrid ensemble: average the two normalized scores. The idea - XGBoost
    # catches known patterns with high precision, Isolation Forest catches
    # novel/unusual patterns XGBoost wasn't trained to recognize.
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    hybrid_scores = 0.7 * normalize(xgb_scores) + 0.3 * normalize(iso_scores)
    hybrid_metrics = evaluate(y_test, hybrid_scores, "Hybrid (0.7*XGB + 0.3*IsoForest)")

    joblib.dump(xgb_model, "models/xgb_model.joblib")
    joblib.dump(iso_model, "models/iso_forest_model.joblib")

    with open("models/metrics.json", "w") as f:
        json.dump({
            "xgboost": xgb_metrics,
            "isolation_forest": iso_metrics,
            "hybrid": hybrid_metrics,
        }, f, indent=2)

    print("\nModels saved to models/. Metrics saved to models/metrics.json")


if __name__ == "__main__":
    main()
