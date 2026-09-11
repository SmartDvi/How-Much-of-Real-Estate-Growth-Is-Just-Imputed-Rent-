"""Consolidate all analysis outputs into one compact JSON for the report/artifact."""
import json
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"


def r(x, nd=4):
    if pd.isna(x):
        return None
    return round(float(x), nd)


flagship = pd.read_csv(OUT / "flagship_series.csv")
flagship_ts = {}
for area, g in flagship.groupby("REF_AREA"):
    g = g.sort_values("TIME_PERIOD")
    flagship_ts[area] = {
        "years": g["TIME_PERIOD"].astype(int).tolist(),
        "imputed_share_of_re": [r(v, 4) for v in g["imputed_share_of_re"]],
        "re_share_of_gdp": [r(v, 4) for v in g["re_share_of_gdp"]],
        "imputed_rent": [r(v, 0) for v in g["imputed_rent"]],
        "real_estate_excl_imputed": [r(v, 0) for v in g["real_estate_excl_imputed"]],
        "real_estate_total": [r(v, 0) for v in g["real_estate_total"]],
        "gdp_basic_prices": [r(v, 0) for v in g["gdp_basic_prices"]],
    }

volume = pd.read_csv(OUT / "volume_indices.csv")
volume_ts = {}
for area, g in volume.groupby("country"):
    g = g.sort_values("year").dropna(subset=["volume_index_imputed_rent"])
    volume_ts[area] = {
        "years": g["year"].astype(int).tolist(),
        "volume_index_imputed_rent": [r(v, 2) for v in g["volume_index_imputed_rent"]],
        "volume_index_excl_imputed": [r(v, 2) for v in g["volume_index_excl_imputed"]],
    }

cross = pd.read_csv(OUT / "cross_section_latest.csv").sort_values("imputed_share_of_re", ascending=False)
cross_section = [{
    "country": row.REF_AREA, "label": row.REF_AREA_LABEL, "year": int(row.TIME_PERIOD),
    "imputed_share_of_re": r(row.imputed_share_of_re, 4),
    "re_share_of_gdp": r(row.re_share_of_gdp, 4),
} for row in cross.itertuples()]

trends = pd.read_csv(OUT / "trend_regressions.csv")
trend_list = [{
    "country": row.country, "label": row.label, "n_years": int(row.n_years),
    "slope_pp_per_year": r(row.slope_pp_per_year, 3), "r_squared": r(row.r_squared, 3),
    "p_value": r(row.p_value, 5),
} for row in trends.itertuples()]

decomp = pd.read_csv(OUT / "growth_decomposition.csv")
decomp_list = decomp.round(2).to_dict(orient="records")

forecasts = pd.read_csv(OUT / "flagship_forecasts.csv")
forecast_list = {}
for area, g in forecasts.groupby("country"):
    g = g.sort_values("year")
    forecast_list[area] = {
        "method": g["method"].iloc[0],
        "years": g["year"].astype(int).tolist(),
        "forecast": [r(v, 4) for v in g["forecast_imputed_share"]],
        "ci_lower": [r(v, 4) for v in g["ci_lower"]],
        "ci_upper": [r(v, 4) for v in g["ci_upper"]],
    }

backtest_mae = pd.read_csv(OUT / "forecast_backtest_mae.csv").round(4).to_dict(orient="records")

shap_imp = pd.read_csv(OUT / "shap_feature_importance.csv").round(5).to_dict(orient="records")
correlations = pd.read_csv(OUT / "correlations.csv").round(4).to_dict(orient="records")

svk = pd.read_csv(OUT / "panel_reliable_current_prices.csv")
svk_full = pd.read_csv(OUT / "flagship_series.csv")  # not SVK; get raw from fact table via duckdb below

import duckdb
con = duckdb.connect(str(OUT / "real_estate.duckdb"), read_only=True)
svk_raw = con.execute("""
    SELECT TIME_PERIOD, imputed_rent, real_estate_excl_imputed, imputed_share_of_re
    FROM fact_real_estate WHERE PRICE_BASE='V' AND REF_AREA='SVK' ORDER BY TIME_PERIOD
""").fetchdf()
con.close()
svk_case = {
    "years": svk_raw["TIME_PERIOD"].astype(int).tolist(),
    "imputed_rent": [r(v, 0) for v in svk_raw["imputed_rent"]],
    "real_estate_excl_imputed": [r(v, 0) for v in svk_raw["real_estate_excl_imputed"]],
    "imputed_share_of_re": [r(v, 4) for v in svk_raw["imputed_share_of_re"]],
}

with open(OUT / "headline_numbers.json") as f:
    headline = json.load(f)
with open(OUT / "ml_summary.json") as f:
    ml_summary = json.load(f)
with open(OUT / "dl_summary.json") as f:
    dl_summary = json.load(f)

bundle = {
    "flagship_ts": flagship_ts,
    "volume_ts": volume_ts,
    "cross_section": cross_section,
    "trend_regressions": trend_list,
    "growth_decomposition": decomp_list,
    "forecasts": forecast_list,
    "backtest_mae": backtest_mae,
    "shap_importance": shap_imp,
    "correlations": correlations,
    "slovakia_case": svk_case,
    "headline": headline,
    "ml_summary": ml_summary,
    "dl_summary": dl_summary,
}

with open(OUT / "report_bundle.json", "w") as f:
    json.dump(bundle, f)

print(f"Report bundle written: {OUT / 'report_bundle.json'} ({(OUT / 'report_bundle.json').stat().st_size:,} bytes)")
