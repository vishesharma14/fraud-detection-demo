"""
explain_model.py
-----------------
Generates SHAP explanations for the XGBoost model:
1. Global feature importance (which features matter most overall)
2. A per-transaction "waterfall" style explanation (why THIS specific
   transaction was flagged) - this is the part most fresher projects skip,
   and the part interviewers most often ask you to walk through live.
"""

import joblib
import matplotlib
matplotlib.use("Agg")  # no display backend needed
import matplotlib.pyplot as plt
import pandas as pd
import shap

X_test = pd.read_csv("data/X_test.csv")
y_test = pd.read_csv("data/y_test.csv").squeeze()
xgb_model = joblib.load("models/xgb_model.joblib")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_test)

# 1. Global importance - which features drive fraud predictions overall
plt.figure()
shap.summary_plot(shap_values, X_test, show=False, plot_size=(10, 6))
plt.tight_layout()
plt.savefig("plots/shap_summary.png", dpi=120)
plt.close()
print("Saved plots/shap_summary.png (global feature importance)")

# 2. Per-transaction explanation for one true fraud case correctly caught
fraud_indices = y_test[y_test == 1].index
scores = xgb_model.predict_proba(X_test)[:, 1]
# pick a fraud transaction the model scored highly (a true positive) to explain
caught_fraud_idx = fraud_indices[scores[fraud_indices] > 0.9][0] if len(
    fraud_indices[scores[fraud_indices] > 0.9]) > 0 else fraud_indices[0]
row_position = X_test.index.get_loc(caught_fraud_idx)

plt.figure()
shap.plots.waterfall(shap_values[row_position], show=False)
plt.tight_layout()
plt.savefig("plots/shap_example_explanation.png", dpi=120)
plt.close()
print("Saved plots/shap_example_explanation.png (single transaction explained)")

print(f"\nExample explained transaction (row {caught_fraud_idx}):")
print(X_test.loc[caught_fraud_idx])
print(f"Model fraud probability: {scores[row_position]:.3f}")
print(f"Actual label: {'FRAUD' if y_test.loc[caught_fraud_idx] == 1 else 'LEGIT'}")
