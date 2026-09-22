"""
generate_data.py
-----------------
Generates a synthetic but REALISTIC transaction dataset for fraud detection.

Why synthetic instead of the standard Kaggle credit-card-fraud CSV?
- That dataset is heavily saturated on fresher resumes (PCA-anonymized, everyone
  uses it identically).
- Here we control the feature semantics (amount, time, location, velocity),
  which lets us do REAL feature engineering and tell a domain story in
  interviews, instead of "V1...V28 are anonymized PCA components."
- Fraud rate is set to 1.5% to mimic real-world class imbalance.

You can later swap this for a real dataset (e.g. IEEE-CIS Fraud Detection,
PaySim) by matching the column schema below in data/transactions.csv.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_SAMPLES = 50_000
FRAUD_RATE = 0.015  # 1.5% fraud - realistic imbalance

n_fraud = int(N_SAMPLES * FRAUD_RATE)
n_legit = N_SAMPLES - n_fraud


def generate_legit(n):
    return pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=3.5, sigma=1.0, size=n).clip(1, 5000),
        "hour_of_day": np.random.choice(range(24), size=n,
                                         p=_hour_weights(peak_hours=(9, 21))),
        "distance_from_home_km": np.random.exponential(scale=15, size=n).clip(0, 500),
        "distance_from_last_txn_km": np.random.exponential(scale=5, size=n).clip(0, 300),
        "txns_last_24h": np.random.poisson(lam=2, size=n).clip(0, 15),
        "avg_txn_amount_last_30d": np.random.lognormal(mean=3.3, sigma=0.6, size=n).clip(1, 3000),
        "is_online": np.random.binomial(1, 0.35, size=n),
        "merchant_risk_score": np.random.beta(2, 8, size=n),  # skewed low = safe merchants
        "card_present": np.random.binomial(1, 0.65, size=n),
        "is_fraud": 0,
    })


def generate_fraud(n):
    # Real fraud is NOT cleanly separable: a meaningful fraction of fraudsters
    # deliberately mimic normal behavior to evade detection. We split fraud
    # into "obvious" (70%) and "sophisticated / mimics legit" (30%) so the
    # classes overlap realistically instead of being trivially separable.
    n_sophisticated = int(n * 0.30)
    n_obvious = n - n_sophisticated

    obvious = pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=4.5, sigma=1.3, size=n_obvious).clip(1, 9000),
        "hour_of_day": np.random.choice(range(24), size=n_obvious,
                                         p=_hour_weights(peak_hours=(0, 5))),
        "distance_from_home_km": np.random.exponential(scale=120, size=n_obvious).clip(0, 3000),
        "distance_from_last_txn_km": np.random.exponential(scale=80, size=n_obvious).clip(0, 2000),
        "txns_last_24h": np.random.poisson(lam=7, size=n_obvious).clip(0, 40),
        "avg_txn_amount_last_30d": np.random.lognormal(mean=3.3, sigma=0.6, size=n_obvious).clip(1, 3000),
        "is_online": np.random.binomial(1, 0.85, size=n_obvious),
        "merchant_risk_score": np.random.beta(5, 3, size=n_obvious),
        "card_present": np.random.binomial(1, 0.15, size=n_obvious),
    })

    # Sophisticated fraud looks statistically much closer to legit behavior -
    # this is what makes real fraud detection hard, and what will pull our
    # PR-AUC down from an unrealistic ~0.99 to a defensible ~0.80-0.90 range.
    sophisticated = pd.DataFrame({
        "transaction_amount": np.random.lognormal(mean=3.6, sigma=1.0, size=n_sophisticated).clip(1, 5000),
        "hour_of_day": np.random.choice(range(24), size=n_sophisticated,
                                         p=_hour_weights(peak_hours=(9, 21))),
        "distance_from_home_km": np.random.exponential(scale=25, size=n_sophisticated).clip(0, 800),
        "distance_from_last_txn_km": np.random.exponential(scale=10, size=n_sophisticated).clip(0, 400),
        "txns_last_24h": np.random.poisson(lam=3, size=n_sophisticated).clip(0, 20),
        "avg_txn_amount_last_30d": np.random.lognormal(mean=3.3, sigma=0.6, size=n_sophisticated).clip(1, 3000),
        "is_online": np.random.binomial(1, 0.55, size=n_sophisticated),
        "merchant_risk_score": np.random.beta(3, 6, size=n_sophisticated),
        "card_present": np.random.binomial(1, 0.45, size=n_sophisticated),
    })

    df = pd.concat([obvious, sophisticated], ignore_index=True)
    df["is_fraud"] = 1
    return df


def _hour_weights(peak_hours):
    """Weighted probability distribution over 24 hours, peaked around given hours."""
    hours = np.arange(24)
    start, end = peak_hours
    if start < end:
        weights = np.where((hours >= start) & (hours <= end), 3.0, 1.0)
    else:  # wraps past midnight
        weights = np.where((hours >= start) | (hours <= end), 3.0, 1.0)
    return weights / weights.sum()


def main():
    legit_df = generate_legit(n_legit)
    fraud_df = generate_fraud(n_fraud)

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
    df.insert(0, "transaction_id", [f"TXN{i:06d}" for i in range(len(df))])

    out_path = "data/transactions.csv"
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} transactions -> {out_path}")
    print(f"Fraud rate: {df['is_fraud'].mean():.3%}")
    print(df.groupby("is_fraud")[["transaction_amount", "distance_from_home_km",
                                   "txns_last_24h"]].mean())


if __name__ == "__main__":
    main()
