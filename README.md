# How Much of "Real Estate" Growth Is Just Imputed Rent?

A quantitative study of **imputed owner-occupier rent** vs. **genuine real-estate market
activity**, built on OECD national accounts and house-price data, DuckDB, applied statistics,
and machine learning.

> National accounts count the "rent" a homeowner would notionally pay themselves as output of
> the real estate industry — even though no transaction, tenant, or job is involved. This
> project measures how large that notional component actually is, whether it's rising, what
> drives it, and whether it can be forecast — then translates the findings into concrete,
> audience-specific recommendations.

---

## Two deliverables, two audiences

| | For | Contents |
|---|---|---|
| **[`Real_estate_vs_econonic_active.ipynb`](./Real_estate_vs_econonic_active.ipynb)** | Analysts, researchers, technical readers | The full, executed, reproducible analysis: data acquisition, DuckDB pipeline, statistics, ML/SHAP, deep learning, forecasting — with a plain-language glossary and "in everyday terms" callouts woven through every section |
| **[`report/phantom_rent_explainer.html`](./report/phantom_rent_explainer.html)** ("Phantom Rent") | Renters, homeowners, property traders, everyday readers | A standalone, jargon-free page built from the same numbers, with a glossary and an action plan grouped by who you are |

Both use identical underlying figures — nothing was re-derived or approximated for the
plain-language version.

## The question

Split **Real estate activities (ISIC L68)** into its two OECD-reported components:

- **L68A — Imputed rents of owner-occupied dwellings** (notional, non-market)
- **L68B — Real estate activities excluding imputed rents** (brokerage, letting, property
  management, development — genuine market services)

and ask: how large is the imputed share of the sector, is it rising, what statistically explains
its movement, and how much of "real estate GDP growth" is actually a valuation exercise rather
than market activity?

## Headline findings

1. **Across 43 countries with a reliable split, imputed rent is a median ~65% of "real estate
   activities" value added.** In a typical country, roughly two-thirds of reported real-estate
   sector output is rent nobody actually paid — a structural fact, not a recent drift.
2. **The trend is country-specific, not universal.** Only 8 of 33 long-series countries show a
   statistically significant *rising* imputed share (p<0.05); 16 show a significant *fall*. A
   cross-country paired test finds no significant average movement (t-test p≈0.235, Wilcoxon
   p≈0.057). Czechia's share rose from 61%→70% (1990–2024); the UK's fell from 80%→73%
   (1997–2023).
3. **But recent marginal growth is one-sided.** In every flagship country, 74–86% of nominal
   real-estate-sector growth since 2019 has come from the imputed component.
4. **Real (volume) decomposition is the sharpest result.** In Czechia, non-imputed real-estate
   services *shrank* in real terms (−1.19%/yr, 1990–2023) while imputed rent grew in real terms
   (+2.03%/yr) — nominal sector growth sits on top of a contracting genuine service sector. The
   UK shows the reverse (non-imputed +3.08%/yr vs. imputed +1.30%/yr).
5. **Large single-year moves are usually statistical events, not market events.** Slovakia's
   imputed share jumped from 0% to ~80% in 2020 purely because its statistical office began
   separately reporting it — a classification artifact indistinguishable, in the aggregate, from
   a genuine boom.
6. **Year-to-year changes in the imputed share are not predictable from housing-market
   cyclicality.** Linear regression, XGBoost (explained with SHAP), and a compact neural network
   all failed to beat a naive "no change" baseline in grouped cross-validation — an honest
   negative result consistent with #5.

Full statistical detail — exact coefficients, p-values, confidence bands, and every chart — is in
the notebook.

## Data sources

Both are public OECD SDMX REST endpoints; no API key required.

| Source | Dataflow | Used for |
|---|---|---|
| OECD National Accounts — Use table incl. value added (Table 1600) | `OECD.SDD.NAD:DSD_NASU@DF_USEVA_T1600` | Value added (`B1G`) by activity (`_T`, `L`, `L68A`, `L68B`), current and previous-year prices, all countries/years |
| OECD Analytical House Price Indicators | `OECD.ECO.MPD:DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES` | Nominal/real house price index, rent price index, price-to-income ratio — used as independent cross-checks and ML features |

Raw pulls are cached in `data/raw/`; re-running the fetch step will not re-hit the API if those
files already exist.

## Methodology

| Stage | Tool |
|---|---|
| Data acquisition | OECD SDMX REST API via `requests` |
| Cleaning & analytical tables | **DuckDB** — splits OECD's `"CODE: Label"` cell format, pivots activities into columns, builds the country/year fact table |
| Data quality | Flags rows where a component is exactly zero as "not yet separately reported" rather than a real economic zero; documents reporting breaks (Slovakia, Ireland) as a finding in their own right |
| Statistical inference | Per-country OLS trend regressions, a paired early-vs-late hypothesis test (t-test + Wilcoxon), Pearson correlations against house/rent price indices — `scipy`, `statsmodels` |
| Growth decomposition | Nominal contribution-to-growth by sub-period, plus a chain-linked **volume** (real) decomposition using previous-year-price series to isolate price effects from genuine activity |
| Machine learning | **XGBoost** vs. linear regression vs. a naive baseline, predicting the *year-on-year change* in imputed share (deliberately not the level, which is trivially dominated by its own lag), evaluated with grouped 5-fold cross-validation (grouped by country) |
| Explainability | **SHAP** (TreeExplainer) on the XGBoost model |
| Deep learning | A compact PyTorch feed-forward network, benchmarked honestly against the above — reported as a negative result rather than oversold |
| Forecasting | Per-country walk-forward backtesting (naive / linear trend / damped-trend Holt) choosing the best method by demonstrated accuracy, with confidence bands scaled to the observed random-walk-like behavior |

## Repository structure

```
.
├── README.md
├── Real_estate_vs_econonic_active.ipynb   # the full, executed analysis notebook
├── data/
│   ├── raw/                                # cached OECD SDMX pulls (CSV)
│   └── processed/                          # DuckDB database + all derived CSV/JSON outputs
├── report/
│   └── phantom_rent_explainer.html         # plain-language standalone page ("Phantom Rent")
└── scripts/                                # the pipeline as standalone, runnable modules
    ├── 01_fetch_data.py                    # OECD SDMX data acquisition
    ├── 02_build_duckdb.py                  # raw -> clean DuckDB analytical tables
    ├── 03_analysis_stats.py                # trend regressions, decomposition, hypothesis tests
    ├── 04_ml_drivers_shap.py               # XGBoost vs. linear vs. naive + SHAP
    ├── 05_forecast_and_dl.py               # walk-forward forecasting + PyTorch comparison
    ├── 06_export_for_report.py             # consolidates results into report_bundle.json
    └── build_notebook.py                   # programmatically builds & executes the .ipynb
```

`scripts/` and the notebook implement the *same* analysis — the scripts are the fast-iteration
pipeline, and `build_notebook.py` assembles the narrated, reproducible notebook (with markdown
methodology notes and plain-language callouts) and executes it end-to-end so every number and
chart in the `.ipynb` is real output, not hand-typed.

## Reproducing this

```bash
# from the project root
python3 -m venv .venv && source .venv/bin/activate
pip install duckdb pandas numpy scipy statsmodels scikit-learn xgboost shap \
            torch --index-url https://download.pytorch.org/whl/cpu \
            matplotlib seaborn plotly requests nbformat nbclient jupyter ipykernel

# run the pipeline stage by stage
python3 scripts/01_fetch_data.py
python3 scripts/02_build_duckdb.py
python3 scripts/03_analysis_stats.py
python3 scripts/04_ml_drivers_shap.py
python3 scripts/05_forecast_and_dl.py
python3 scripts/06_export_for_report.py

# or rebuild + execute the full notebook in one step
python3 -m ipykernel install --user --name python3   # once, if no kernel is registered
python3 scripts/build_notebook.py
```

Tested with Python 3.12. `data/raw/` is cached after the first fetch, so re-running is fast and
does not depend on network access.

## Limitations

- The nominal/real distinction relies on OECD "previous-year prices" series, a reasonable
  SNA-consistent approximation — not an official chain-linked volume series.
- Country coverage of the L68A/L68B split is uneven: 43 of ~60 countries in the underlying
  dataflow report it at all, with span lengths from 2 to 35 years.
- House-price and rent-price indicators are national aggregates, not real-estate-segment-specific
  — a proxy relationship, not a direct one.
- The ML/DL results are a genuine negative finding, not a "modest" one: no model tested here beat
  a naive no-change baseline, and the write-up says so rather than reporting a weak model as a
  result.

## Attribution

Data: OECD.Stat (National Accounts and Analytical House Price Indicators), publicly available
under OECD's terms of use. All analysis, code, and narrative in this repository are original work
produced for this project.
