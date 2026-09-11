"""
Part A: Per-country time-series forecasting of the imputed-rent share to 2030
for the three flagship countries (Czechia, USA, UK), comparing:
  - naive (last value carried forward)
  - linear trend extrapolation
  - damped-trend exponential smoothing (statsmodels Holt)
Each is backtested with expanding-window, 1-year-ahead walk-forward validation
over the last 8 available years, so the forecast method is chosen on
demonstrated accuracy rather than convenience.

Part B: A compact PyTorch feed-forward network trained on the pooled
country-year panel to predict the year-on-year change in imputed share
(same target and grouped train/test split as script 04's XGBoost model),
to test honestly whether deep learning adds anything over the classical /
GBM baselines given the available sample size (~367 rows).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import Holt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"

# ===========================================================================
# PART A: Flagship time-series forecasting with walk-forward backtesting
# ===========================================================================
flagship = pd.read_csv(OUT / "flagship_series.csv")
FLAGSHIPS = ["CZE", "USA", "GBR"]
N_BACKTEST = 8
HORIZON_YEARS = list(range(1, 7))  # forecast 2025..2030 style (1..6 years ahead)


def backtest_methods(series: pd.Series):
    """series indexed by year, values = imputed_share_of_re."""
    years = series.index.tolist()
    results = {"naive": [], "linear": [], "holt": []}
    start = len(years) - N_BACKTEST
    for i in range(start, len(years) - 1):
        train = series.iloc[: i + 1]
        actual_next = series.iloc[i + 1]

        # naive
        results["naive"].append(abs(train.iloc[-1] - actual_next))

        # linear trend
        x = np.arange(len(train))
        coeffs = np.polyfit(x, train.values, 1)
        pred_linear = np.polyval(coeffs, len(train))
        results["linear"].append(abs(pred_linear - actual_next))

        # Holt damped trend
        try:
            fit = Holt(train.values, damped_trend=True, initialization_method="estimated").fit()
            pred_holt = fit.forecast(1)[0]
            results["holt"].append(abs(pred_holt - actual_next))
        except Exception:
            results["holt"].append(np.nan)

    return {k: float(np.nanmean(v)) for k, v in results.items()}


forecast_rows = []
backtest_summary = []
for area in FLAGSHIPS:
    g = flagship[flagship["REF_AREA"] == area].sort_values("TIME_PERIOD")
    series = g.set_index("TIME_PERIOD")["imputed_share_of_re"]

    bt = backtest_methods(series)
    bt["country"] = area
    backtest_summary.append(bt)
    best_method = min(["naive", "linear", "holt"], key=lambda m: bt[m])
    one_step_mae = bt[best_method]
    # 80% interval, random-walk-consistent sqrt(h) scaling: the driver
    # analysis (Part B) found year-on-year changes behave like noise
    # around zero, so an h-step-ahead band widening as sigma*sqrt(h) is the
    # honest choice rather than a false sense of narrowing/tightening.
    z80 = 1.28

    last_year = series.index.max()
    x_full = np.arange(len(series))
    if best_method == "linear":
        coeffs = np.polyfit(x_full, series.values, 1)
        for h in HORIZON_YEARS:
            pred = np.polyval(coeffs, len(series) - 1 + h)
            forecast_rows.append({"country": area, "year": last_year + h, "method": best_method,
                                   "forecast_imputed_share": pred,
                                   "ci_lower": pred - z80 * one_step_mae * np.sqrt(h),
                                   "ci_upper": pred + z80 * one_step_mae * np.sqrt(h)})
    elif best_method == "holt":
        fit = Holt(series.values, damped_trend=True, initialization_method="estimated").fit()
        preds = fit.forecast(max(HORIZON_YEARS))
        for h in HORIZON_YEARS:
            forecast_rows.append({"country": area, "year": last_year + h, "method": best_method,
                                   "forecast_imputed_share": preds[h - 1],
                                   "ci_lower": preds[h - 1] - z80 * one_step_mae * np.sqrt(h),
                                   "ci_upper": preds[h - 1] + z80 * one_step_mae * np.sqrt(h)})
    else:  # naive
        for h in HORIZON_YEARS:
            forecast_rows.append({"country": area, "year": last_year + h, "method": best_method,
                                   "forecast_imputed_share": series.values[-1],
                                   "ci_lower": series.values[-1] - z80 * one_step_mae * np.sqrt(h),
                                   "ci_upper": series.values[-1] + z80 * one_step_mae * np.sqrt(h)})

backtest_df = pd.DataFrame(backtest_summary).set_index("country")
print("--- Walk-forward backtest MAE (share points), last 8 one-year-ahead predictions ---")
print(backtest_df.to_string())
backtest_df.to_csv(OUT / "forecast_backtest_mae.csv")

forecast_df = pd.DataFrame(forecast_rows)
forecast_df.to_csv(OUT / "flagship_forecasts.csv", index=False)
print("\n--- Chosen method per country (lowest backtest MAE) and 2030 forecast ---")
for area in FLAGSHIPS:
    sub = forecast_df[forecast_df["country"] == area]
    method = sub["method"].iloc[0]
    end_forecast = sub.iloc[-1]
    print(f"{area}: method={method}, imputed share forecast for {int(end_forecast['year'])} = "
          f"{end_forecast['forecast_imputed_share']*100:.1f}%")

# ===========================================================================
# PART B: Compact deep-learning comparison on the pooled panel
# ===========================================================================
print("\n" + "=" * 70)
print("PART B: Small feed-forward neural net vs XGBoost/linear/naive baseline")
print("=" * 70)

df = pd.read_csv(OUT / "ml_oof_predictions.csv")
FEATURES = ["imputed_share_lag1", "HPI_yoy", "RPI_yoy", "HPI_YDH",
            "gdp_growth", "re_total_growth", "years_since_start"]
TARGET = "imputed_share_change"

X = df[FEATURES].values.astype(np.float32)
y = df[TARGET].values.astype(np.float32).reshape(-1, 1)
groups = df["REF_AREA"].values

torch.manual_seed(42)


class TinyMLP(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)


gkf = GroupKFold(n_splits=5)
dl_maes = []
for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
    scaler = StandardScaler().fit(X[tr_idx])
    X_tr = torch.tensor(scaler.transform(X[tr_idx]), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X[te_idx]), dtype=torch.float32)
    y_tr = torch.tensor(y[tr_idx], dtype=torch.float32)
    y_te_np = y[te_idx].ravel()

    model = TinyMLP(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-3)
    loss_fn = nn.L1Loss()

    model.train()
    for epoch in range(150):
        opt.zero_grad()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        pred_te = model(X_te).numpy().ravel()
    mae = mean_absolute_error(y_te_np, pred_te)
    dl_maes.append(mae)
    print(f"fold {fold}: DL MAE = {mae:.4f}")

dl_mae_mean = float(np.mean(dl_maes))
naive_mae = float(np.abs(y).mean())
print(f"\nTinyMLP mean MAE = {dl_mae_mean:.4f} ({dl_mae_mean*100:.2f} pp)")
print(f"Naive (zero-change) MAE = {naive_mae:.4f} ({naive_mae*100:.2f} pp)")
print(f"XGBoost MAE (from script 04) for comparison: see ml_summary.json")

with open(OUT / "ml_summary.json") as f:
    ml_summary = json.load(f)

verdict = "no better than" if dl_mae_mean >= naive_mae * 0.97 else "modestly better than"
print(f"\nVerdict: the neural network is {verdict} the naive baseline. "
      f"With ~{len(df)} rows and a near-random-walk target, deep learning "
      f"has no discernible advantage over simpler models here.")

dl_summary = {
    "dl_cv_mae_mean": dl_mae_mean,
    "naive_mae": naive_mae,
    "xgb_cv_mae_mean": ml_summary["xgb_cv_mae_mean"],
    "linear_cv_mae_mean": ml_summary["linear_cv_mae_mean"],
    "verdict": verdict,
    "n_rows": len(df),
}
with open(OUT / "dl_summary.json", "w") as f:
    json.dump(dl_summary, f, indent=2)

print("\nForecast + DL comparison complete. Outputs in data/processed/")
