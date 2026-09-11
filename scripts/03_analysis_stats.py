"""
Core statistical analysis:
  - data-quality flagging (zero/not-separately-reported rows)
  - flagship long-run country deep dives (Czechia, USA, UK)
  - cross-sectional distribution of imputed share (latest year, all reliable countries)
  - per-country trend regressions (imputed share ~ year)
  - nominal growth decomposition (contribution of imputed vs non-imputed to RE growth)
  - chain-linked volume ("real") decomposition using previous-year-price series
  - correlation with OECD house price / rent price indicators
  - paired hypothesis test: early vs late period imputed share
All outputs are written to data/processed/ as CSV/JSON for reuse by the ML
stage and the report, and key numbers are printed for direct inspection.
"""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "real_estate.duckdb"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH), read_only=False)

# ---------------------------------------------------------------------------
# 1. Reliable panel: rows where the imputed/non-imputed split is genuinely
#    reported (both components > 0). Zero on either side means the country
#    had not yet introduced / back-cast the split for that year.
# ---------------------------------------------------------------------------
raw = con.execute("SELECT * FROM fact_real_estate WHERE PRICE_BASE = 'V'").fetchdf()
raw["reliable"] = (raw["imputed_rent"] > 0) & (raw["real_estate_excl_imputed"] > 0)

reliable = raw[raw["reliable"]].copy()
reliable.to_csv(OUT / "panel_reliable_current_prices.csv", index=False)

n_countries_reliable = reliable["REF_AREA"].nunique()
print(f"Reliable (V-price) rows: {len(reliable)} across {n_countries_reliable} countries")

# Countries with a methodology break (some zero years, some non-zero years)
break_countries = (
    raw.groupby(["REF_AREA", "REF_AREA_LABEL"])["reliable"]
    .agg(["sum", "count"])
    .reset_index()
)
break_countries = break_countries[
    (break_countries["sum"] > 0) & (break_countries["sum"] < break_countries["count"])
]
break_countries.to_csv(OUT / "methodology_break_countries.csv", index=False)
print("\nCountries with an apparent methodology break (partial zero years):")
print(break_countries.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. Flagship long-run deep dives
# ---------------------------------------------------------------------------
FLAGSHIPS = ["CZE", "USA", "GBR"]
flagship_df = reliable[reliable["REF_AREA"].isin(FLAGSHIPS)].sort_values(["REF_AREA", "TIME_PERIOD"])
flagship_df.to_csv(OUT / "flagship_series.csv", index=False)

print("\n--- Flagship coverage ---")
print(flagship_df.groupby("REF_AREA")["TIME_PERIOD"].agg(["min", "max", "count"]))


def cagr(v0, v1, n_years):
    if v0 <= 0 or n_years <= 0:
        return np.nan
    return (v1 / v0) ** (1 / n_years) - 1


flagship_summary = []
for area, g in flagship_df.groupby("REF_AREA"):
    g = g.sort_values("TIME_PERIOD")
    y0, y1 = g["TIME_PERIOD"].iloc[0], g["TIME_PERIOD"].iloc[-1]
    n = y1 - y0
    row0, row1 = g.iloc[0], g.iloc[-1]
    flagship_summary.append({
        "country": area,
        "start_year": y0, "end_year": y1,
        "imputed_share_start": row0["imputed_share_of_re"],
        "imputed_share_end": row1["imputed_share_of_re"],
        "imputed_share_change_pp": (row1["imputed_share_of_re"] - row0["imputed_share_of_re"]) * 100,
        "re_total_cagr_nominal": cagr(row0["real_estate_total"], row1["real_estate_total"], n),
        "imputed_cagr_nominal": cagr(row0["imputed_rent"], row1["imputed_rent"], n),
        "excl_imputed_cagr_nominal": cagr(row0["real_estate_excl_imputed"], row1["real_estate_excl_imputed"], n),
        "gdp_cagr_nominal": cagr(row0["gdp_basic_prices"], row1["gdp_basic_prices"], n),
        "re_share_of_gdp_start": row0["re_share_of_gdp"],
        "re_share_of_gdp_end": row1["re_share_of_gdp"],
    })
flagship_summary = pd.DataFrame(flagship_summary)
flagship_summary.to_csv(OUT / "flagship_summary.csv", index=False)
print("\n--- Flagship summary (full period) ---")
print(flagship_summary.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Growth decomposition: how much of nominal RE growth (in levels, not %)
#    came from imputed rent vs the non-imputed component, in defined
#    sub-periods.
# ---------------------------------------------------------------------------
PERIODS = {
    "CZE": [(1995, 2007), (2007, 2015), (2015, 2019), (2019, 2024)],
    "USA": [(1997, 2007), (2007, 2015), (2015, 2019), (2019, 2024)],
    "GBR": [(1997, 2007), (2007, 2015), (2015, 2019), (2019, 2023)],
}
decomp_rows = []
for area, periods in PERIODS.items():
    g = flagship_df[flagship_df["REF_AREA"] == area].set_index("TIME_PERIOD")
    for (y0, y1) in periods:
        if y0 not in g.index or y1 not in g.index:
            continue
        d_total = g.loc[y1, "real_estate_total"] - g.loc[y0, "real_estate_total"]
        d_imputed = g.loc[y1, "imputed_rent"] - g.loc[y0, "imputed_rent"]
        d_excl = g.loc[y1, "real_estate_excl_imputed"] - g.loc[y0, "real_estate_excl_imputed"]
        decomp_rows.append({
            "country": area, "period": f"{y0}-{y1}",
            "re_total_growth_pct": d_total / g.loc[y0, "real_estate_total"] * 100,
            "imputed_contribution_pct_of_growth": (d_imputed / d_total * 100) if d_total != 0 else np.nan,
            "excl_imputed_contribution_pct_of_growth": (d_excl / d_total * 100) if d_total != 0 else np.nan,
            "imputed_cagr_pct": cagr(g.loc[y0, "imputed_rent"], g.loc[y1, "imputed_rent"], y1 - y0) * 100,
            "excl_imputed_cagr_pct": cagr(g.loc[y0, "real_estate_excl_imputed"], g.loc[y1, "real_estate_excl_imputed"], y1 - y0) * 100,
        })
decomp = pd.DataFrame(decomp_rows)
decomp.to_csv(OUT / "growth_decomposition.csv", index=False)
print("\n--- Growth decomposition by sub-period ---")
print(decomp.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Chain-linked volume ("real") decomposition using previous-year prices
# ---------------------------------------------------------------------------
py = con.execute("SELECT * FROM fact_real_estate WHERE PRICE_BASE = 'Y'").fetchdf()
py = py.rename(columns={
    "imputed_rent": "imputed_rent_py",
    "real_estate_excl_imputed": "excl_imputed_py",
    "real_estate_total": "real_estate_total_py",
})[["REF_AREA", "TIME_PERIOD", "imputed_rent_py", "excl_imputed_py", "real_estate_total_py"]]

vol_rows = []
for area in FLAGSHIPS:
    cur = reliable[reliable["REF_AREA"] == area].sort_values("TIME_PERIOD").set_index("TIME_PERIOD")
    pyc = py[py["REF_AREA"] == area].set_index("TIME_PERIOD")
    years = sorted(cur.index)
    idx_imputed, idx_excl = {}, {}
    base_year = years[0]
    idx_imputed[base_year] = 100.0
    idx_excl[base_year] = 100.0
    for t in years[1:]:
        t_prev = t - 1
        if t_prev not in cur.index or t not in pyc.index:
            continue
        v_prev_imputed = cur.loc[t_prev, "imputed_rent"]
        p_t_imputed = pyc.loc[t, "imputed_rent_py"]
        v_prev_excl = cur.loc[t_prev, "real_estate_excl_imputed"]
        p_t_excl = pyc.loc[t, "excl_imputed_py"]
        if pd.isna(p_t_imputed) or v_prev_imputed in (0, None) or pd.isna(v_prev_imputed):
            continue
        growth_imputed = p_t_imputed / v_prev_imputed
        growth_excl = p_t_excl / v_prev_excl if (v_prev_excl not in (0, None) and not pd.isna(p_t_excl)) else np.nan
        prev_idx_imputed = idx_imputed.get(t_prev)
        prev_idx_excl = idx_excl.get(t_prev)
        if prev_idx_imputed is not None:
            idx_imputed[t] = prev_idx_imputed * growth_imputed
        if prev_idx_excl is not None and not pd.isna(growth_excl):
            idx_excl[t] = prev_idx_excl * growth_excl
    for t in years:
        vol_rows.append({
            "country": area, "year": t,
            "volume_index_imputed_rent": idx_imputed.get(t),
            "volume_index_excl_imputed": idx_excl.get(t),
        })
volume_df = pd.DataFrame(vol_rows)
volume_df.to_csv(OUT / "volume_indices.csv", index=False)

print("\n--- Real (volume) growth vs nominal growth, full available window ---")
for area in FLAGSHIPS:
    v = volume_df[volume_df["country"] == area].dropna()
    if len(v) < 2:
        continue
    y0, y1 = v["year"].iloc[0], v["year"].iloc[-1]
    n = y1 - y0
    vi0, vi1 = v["volume_index_imputed_rent"].iloc[0], v["volume_index_imputed_rent"].iloc[-1]
    ve0, ve1 = v["volume_index_excl_imputed"].iloc[0], v["volume_index_excl_imputed"].iloc[-1]
    print(f"{area} ({y0}-{y1}, n={n}): "
          f"real CAGR imputed rent = {cagr(vi0, vi1, n)*100:.2f}%, "
          f"real CAGR non-imputed RE = {cagr(ve0, ve1, n)*100:.2f}%")

# ---------------------------------------------------------------------------
# 5. Cross-sectional distribution (latest reliable year per country)
# ---------------------------------------------------------------------------
latest = (
    reliable.sort_values("TIME_PERIOD")
    .groupby("REF_AREA")
    .tail(1)
    .sort_values("imputed_share_of_re", ascending=False)
)
latest.to_csv(OUT / "cross_section_latest.csv", index=False)

desc = latest["imputed_share_of_re"].describe(percentiles=[.1, .25, .5, .75, .9])
print("\n--- Cross-sectional distribution of imputed_share_of_re (latest year per country) ---")
print(desc)
print(f"\nCoefficient of variation: {desc['std']/desc['mean']:.3f}")

# ---------------------------------------------------------------------------
# 6. Per-country trend regression: imputed_share_of_re ~ year (OLS)
#    only for countries with >=8 reliable years
# ---------------------------------------------------------------------------
trend_rows = []
for area, g in reliable.groupby("REF_AREA"):
    g = g.sort_values("TIME_PERIOD")
    if len(g) < 8:
        continue
    x = g["TIME_PERIOD"].values.astype(float)
    y = g["imputed_share_of_re"].values.astype(float)
    slope, intercept, r, p, se = stats.linregress(x, y)
    trend_rows.append({
        "country": area, "label": g["REF_AREA_LABEL"].iloc[0], "n_years": len(g),
        "slope_pp_per_year": slope * 100, "r_squared": r ** 2, "p_value": p,
    })
trends = pd.DataFrame(trend_rows).sort_values("slope_pp_per_year", ascending=False)
trends.to_csv(OUT / "trend_regressions.csv", index=False)
print(f"\n--- Trend regressions (n={len(trends)} countries with >=8 reliable years) ---")
print(trends.to_string(index=False))
n_sig_positive = ((trends["p_value"] < 0.05) & (trends["slope_pp_per_year"] > 0)).sum()
print(f"\n{n_sig_positive}/{len(trends)} countries show a statistically significant (p<0.05) RISING imputed share")

# ---------------------------------------------------------------------------
# 7. Paired hypothesis test: early vs late period imputed share, by country
# ---------------------------------------------------------------------------
paired_rows = []
for area, g in reliable.groupby("REF_AREA"):
    g = g.sort_values("TIME_PERIOD")
    if len(g) < 10:
        continue
    k = len(g) // 3
    early = g["imputed_share_of_re"].iloc[:k].mean()
    late = g["imputed_share_of_re"].iloc[-k:].mean()
    paired_rows.append({"country": area, "early_mean": early, "late_mean": late})
paired = pd.DataFrame(paired_rows)
paired.to_csv(OUT / "paired_early_late.csv", index=False)

if len(paired) >= 5:
    wstat, wp = stats.wilcoxon(paired["late_mean"], paired["early_mean"])
    tstat, tp = stats.ttest_rel(paired["late_mean"], paired["early_mean"])
    print(f"\n--- Paired early-vs-late imputed share test (n={len(paired)} countries) ---")
    print(f"Mean early={paired['early_mean'].mean():.4f}, mean late={paired['late_mean'].mean():.4f}")
    print(f"Paired t-test: t={tstat:.3f}, p={tp:.5f}")
    print(f"Wilcoxon signed-rank: W={wstat:.3f}, p={wp:.5f}")
    hyp_test = {"n": len(paired), "mean_early": paired["early_mean"].mean(),
                "mean_late": paired["late_mean"].mean(), "t_stat": tstat, "t_p": tp,
                "wilcoxon_stat": wstat, "wilcoxon_p": wp}
else:
    hyp_test = {}

# ---------------------------------------------------------------------------
# 8. Correlation with house price / rent price indicators
# ---------------------------------------------------------------------------
hp = con.execute("""
    SELECT REF_AREA, TIME_PERIOD, MEASURE, OBS_VALUE
    FROM house_prices_clean
    WHERE ADJUSTMENT LIKE 'S%' OR ADJUSTMENT IS NULL
""").fetchdf()
hp_piv = hp.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns="MEASURE", values="OBS_VALUE", aggfunc="mean").reset_index()

merged = reliable.merge(hp_piv, on=["REF_AREA", "TIME_PERIOD"], how="inner")
merged = merged.sort_values(["REF_AREA", "TIME_PERIOD"])
for col in ["imputed_rent", "real_estate_excl_imputed", "HPI", "RPI"]:
    if col in merged.columns:
        merged[f"{col}_yoy"] = merged.groupby("REF_AREA")[col].pct_change() * 100

merged.to_csv(OUT / "merged_house_prices.csv", index=False)

corr_pairs = [
    ("imputed_rent_yoy", "RPI_yoy", "Imputed rent (nominal) growth vs official Rent Price Index growth"),
    ("real_estate_excl_imputed_yoy", "HPI_yoy", "Non-imputed RE value-added growth vs House Price Index growth"),
    ("imputed_share_of_re", "HPI_YDH", "Imputed share level vs House-Price-to-Income ratio (affordability)"),
]
corr_results = []
for c1, c2, desc_txt in corr_pairs:
    if c1 in merged.columns and c2 in merged.columns:
        sub = merged[[c1, c2]].dropna()
        if len(sub) >= 10:
            r, p = stats.pearsonr(sub[c1], sub[c2])
            corr_results.append({"pair": desc_txt, "var1": c1, "var2": c2, "n": len(sub), "pearson_r": r, "p_value": p})
corr_df = pd.DataFrame(corr_results)
corr_df.to_csv(OUT / "correlations.csv", index=False)
print("\n--- Correlations with house/rent price indicators ---")
print(corr_df.to_string(index=False))

# ---------------------------------------------------------------------------
# Save a JSON bundle of headline numbers for the report
# ---------------------------------------------------------------------------
headline = {
    "n_countries_reliable_panel": int(n_countries_reliable),
    "cross_section_latest_year_stats": desc.to_dict(),
    "n_trend_countries": int(len(trends)),
    "n_significant_rising": int(n_sig_positive),
    "hypothesis_test_early_vs_late": hyp_test,
    "flagship_summary": flagship_summary.to_dict(orient="records"),
}
with open(OUT / "headline_numbers.json", "w") as f:
    json.dump(headline, f, indent=2, default=str)

con.close()
print("\nAll analysis outputs written to data/processed/")
