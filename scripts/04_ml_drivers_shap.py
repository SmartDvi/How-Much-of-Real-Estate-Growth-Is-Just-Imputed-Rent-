"""
ML driver analysis: what predicts a country-year's imputed-rent share of the
real estate sector, and its year-on-year change?

Model: gradient-boosted trees (XGBoost) on the pooled country-year panel,
evaluated with grouped (by-country) cross-validation so the model is never
tested on a country it trained on. Explained with SHAP.

We deliberately do NOT reach for deep learning here: the panel has ~350-400
usable rows after feature lags, which is adequate for a regularised GBM but
too small for a neural net to learn anything a simpler model can't -- that
comparison is made explicit in script 05.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"

df = pd.read_csv(OUT / "merged_house_prices.csv")
df = df.sort_values(["REF_AREA", "TIME_PERIOD"])

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
df["imputed_share_change"] = df.groupby("REF_AREA")["imputed_share_of_re"].diff()
df["imputed_share_lag1"] = df.groupby("REF_AREA")["imputed_share_of_re"].shift(1)
df["gdp_growth"] = df.groupby("REF_AREA")["gdp_basic_prices"].pct_change() * 100
df["re_total_growth"] = df.groupby("REF_AREA")["real_estate_total"].pct_change() * 100
df["years_since_start"] = df.groupby("REF_AREA")["TIME_PERIOD"].transform(lambda s: s - s.min())

FEATURES = [
    "imputed_share_lag1", "HPI_yoy", "RPI_yoy", "HPI_YDH",
    "gdp_growth", "re_total_growth", "years_since_start",
]
# Primary target: the YEAR-ON-YEAR CHANGE in imputed share, not its level.
# The level is trivially dominated by its own lag (a country's sectoral mix
# barely moves year to year), which would make any model look artificially
# good without identifying real drivers. The change is the harder, more
# informative target for driver / SHAP analysis.
TARGET = "imputed_share_change"

model_df = df.dropna(subset=FEATURES + [TARGET]).copy()
print(f"Model dataset: {len(model_df)} rows, {model_df['REF_AREA'].nunique()} countries")
print(f"Target = {TARGET} (year-on-year change in imputed share, in share points)")
print(f"Naive baseline (predict zero change) MAE = {model_df[TARGET].abs().mean():.4f} "
      f"({model_df[TARGET].abs().mean()*100:.2f} pp)")

X = model_df[FEATURES]
y = model_df[TARGET]
groups = model_df["REF_AREA"]

# ---------------------------------------------------------------------------
# Grouped CV: XGBoost vs a plain linear-regression baseline
# ---------------------------------------------------------------------------
gkf = GroupKFold(n_splits=5)
xgb_scores, lin_scores, naive_scores = [], [], []
xgb_preds = np.full(len(model_df), np.nan)

for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
        random_state=42,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    xgb_preds[te_idx] = pred
    xgb_scores.append({"fold": fold, "mae": mean_absolute_error(y_te, pred), "r2": r2_score(y_te, pred)})

    lin = LinearRegression().fit(X_tr, y_tr)
    lin_pred = lin.predict(X_te)
    lin_scores.append({"fold": fold, "mae": mean_absolute_error(y_te, lin_pred), "r2": r2_score(y_te, lin_pred)})

    naive_pred = np.zeros_like(y_te)
    naive_scores.append({"fold": fold, "mae": mean_absolute_error(y_te, naive_pred)})

xgb_scores_df = pd.DataFrame(xgb_scores)
lin_scores_df = pd.DataFrame(lin_scores)
naive_scores_df = pd.DataFrame(naive_scores)
print("\n--- Naive baseline (predict zero change), same folds ---")
print(f"Mean MAE = {naive_scores_df['mae'].mean():.4f} ({naive_scores_df['mae'].mean()*100:.2f} pp)")

print("\n--- XGBoost, 5-fold grouped-by-country CV ---")
print(xgb_scores_df.to_string(index=False))
print(f"Mean MAE = {xgb_scores_df['mae'].mean():.4f} ({xgb_scores_df['mae'].mean()*100:.2f} pp), "
      f"Mean R2 = {xgb_scores_df['r2'].mean():.3f}")

print("\n--- Linear regression baseline, same folds ---")
print(lin_scores_df.to_string(index=False))
print(f"Mean MAE = {lin_scores_df['mae'].mean():.4f} ({lin_scores_df['mae'].mean()*100:.2f} pp), "
      f"Mean R2 = {lin_scores_df['r2'].mean():.3f}")

skill_vs_naive_pct = (1 - xgb_scores_df["mae"].mean() / naive_scores_df["mae"].mean()) * 100
print(f"\nXGBoost improves on the naive (zero-change) baseline by {skill_vs_naive_pct:.1f}% MAE reduction")

model_df["xgb_pred_oof"] = xgb_preds
model_df.to_csv(OUT / "ml_oof_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# Fit final model on all data for SHAP explanation
# ---------------------------------------------------------------------------
final_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
    random_state=42,
)
final_model.fit(X, y)

explainer = shap.TreeExplainer(final_model)
shap_values = explainer(X)

shap_importance = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)
shap_importance.to_csv(OUT / "shap_feature_importance.csv", index=False)
print("\n--- SHAP mean |contribution| by feature (pp of imputed share) ---")
print(shap_importance.to_string(index=False))

# Save raw SHAP values + feature values for plotting in the report
shap_df = pd.DataFrame(shap_values.values, columns=[f"shap_{c}" for c in FEATURES])
feat_df = X.reset_index(drop=True)
shap_export = pd.concat([
    model_df[["REF_AREA", "TIME_PERIOD", TARGET]].reset_index(drop=True),
    feat_df, shap_df,
], axis=1)
shap_export["shap_base_value"] = explainer.expected_value
shap_export.to_csv(OUT / "shap_values_full.csv", index=False)

# A concrete, named example for the narrative: the single largest positive
# and negative SHAP contribution of HPI_yoy / RPI_yoy for a flagship country
for feat in ["HPI_yoy", "RPI_yoy", "imputed_share_lag1"]:
    idx = shap_export[f"shap_{feat}"].abs().idxmax()
    row = shap_export.loc[idx]
    print(f"\nLargest |SHAP| for {feat}: {row['REF_AREA']} {int(row['TIME_PERIOD'])}, "
          f"{feat}={row[feat]:.2f}, shap={row[f'shap_{feat}']:.4f}")

with open(OUT / "ml_summary.json", "w") as f:
    json.dump({
        "n_rows": len(model_df),
        "n_countries": int(model_df["REF_AREA"].nunique()),
        "features": FEATURES,
        "naive_cv_mae_mean": float(naive_scores_df["mae"].mean()),
        "xgb_cv_mae_mean": float(xgb_scores_df["mae"].mean()),
        "xgb_cv_r2_mean": float(xgb_scores_df["r2"].mean()),
        "linear_cv_mae_mean": float(lin_scores_df["mae"].mean()),
        "linear_cv_r2_mean": float(lin_scores_df["r2"].mean()),
        "xgb_skill_vs_naive_pct_mae_reduction": float(skill_vs_naive_pct),
        "shap_importance": shap_importance.to_dict(orient="records"),
    }, f, indent=2)

print("\nML driver analysis complete. Outputs in data/processed/")
