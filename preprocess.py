"""
preprocess.py
-------------
Loads raw transactions, engineers additional features, and creates a
stratified train/test split (stratification matters here because of the
1.5% fraud rate - a random split could otherwise leave too few fraud
examples in the test set).
"""

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/transactions.csv"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ratio of this transaction to the user's historical average - a spike
    # relative to your own baseline is a stronger fraud signal than raw amount.
    df["amount_to_avg_ratio"] = df["transaction_amount"] / df["avg_txn_amount_last_30d"].clip(lower=1)

    # Flag transactions in the "risky hours" window (midnight - 5am)
    df["is_night_txn"] = df["hour_of_day"].apply(lambda h: 1 if h <= 5 else 0)

    # Online + card-not-present is a classic high-risk combination
    df["online_and_not_present"] = ((df["is_online"] == 1) & (df["card_present"] == 0)).astype(int)

    # Velocity flag: unusually many transactions in the last 24h
    df["high_velocity"] = (df["txns_last_24h"] > df["txns_last_24h"].quantile(0.90)).astype(int)

    return df


def main():
    df = pd.read_csv(RAW_PATH)
    df = engineer_features(df)

    feature_cols = [
        "transaction_amount", "hour_of_day", "distance_from_home_km",
        "distance_from_last_txn_km", "txns_last_24h", "avg_txn_amount_last_30d",
        "is_online", "merchant_risk_score", "card_present",
        "amount_to_avg_ratio", "is_night_txn", "online_and_not_present", "high_velocity",
    ]

    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print(f"Train: {len(X_train)} rows, fraud rate {y_train.mean():.3%}")
    print(f"Test:  {len(X_test)} rows, fraud rate {y_test.mean():.3%}")
    print(f"Features used ({len(feature_cols)}): {feature_cols}")


if __name__ == "__main__":
    main()
