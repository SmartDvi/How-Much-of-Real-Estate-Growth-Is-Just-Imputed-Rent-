"""
Fetch raw data from OECD SDMX APIs for the imputed-rent vs. real-estate-activity study.

Sources
-------
1. OECD National Accounts, Use table incl. value added (T1600)
   dataflow: OECD.SDD.NAD:DSD_NASU@DF_USEVA_T1600
   TRANSACTION = B1G (Value added, gross)
   ACTIVITY    = _T (total, all activities -> GDP at basic prices)
                 L    (Real estate activities, total)
                 L68A (Imputed rents of owner-occupied dwellings)
                 L68B (Real estate activities excluding imputed rents)
   PRICE_BASE  = V (current prices) and Y (previous year's prices, used to
                 chain-link a volume/"real growth" measure)

2. OECD Analytical House Price Indicators
   dataflow: OECD.ECO.MPD:DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES
   MEASURE = HPI (nominal house price index), RHP (real house price index),
             RPI (rent price index), HPI_YDH (price-to-income ratio)

Both are official, publicly documented SDMX endpoints; no API key required.
"""
import sys
import time
import requests
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"Accept": "application/vnd.sdmx.data+csv;file=true;labels=both"}
TIMEOUT = 180


def fetch(url: str, params: dict, out_path: Path, label: str):
    print(f"[fetch] {label} -> {out_path.name}")
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
            r.raise_for_status()
            out_path.write_text(r.text, encoding="utf-8")
            n_lines = r.text.count("\n")
            print(f"        OK  ({n_lines} lines, {len(r.text):,} bytes)")
            return
        except Exception as e:
            print(f"        attempt {attempt+1} failed: {e}")
            time.sleep(3)
    print(f"        FAILED after retries: {label}", file=sys.stderr)


def main():
    # --- 1. National accounts: value added by activity, all countries, all years ---
    key = ".".join([
        "A",                        # FREQ = Annual
        "",                         # REF_AREA = all
        "B1G",                      # TRANSACTION = Value added, gross
        "L68A+L68B+L+_T",           # ACTIVITY
        "",                         # PRODUCT
        "",                         # UNIT_MEASURE
        "",                         # VALUATION
        "V+Y",                      # PRICE_BASE = current + previous-year prices
        "",                         # TABLE_IDENTIFIER
    ])
    url = f"https://sdmx.oecd.org/public/rest/data/OECD.SDD.NAD,DSD_NASU@DF_USEVA_T1600,/{key}"
    fetch(url, {"dimensionAtObservation": "AllDimensions"},
          RAW_DIR / "oecd_nad_value_added.csv",
          "OECD National Accounts value added (real estate breakdown)")

    # --- 2. House price indicators, all countries, all years ---
    key2 = ".".join(["", "A", "", ""])  # REF_AREA.FREQ.MEASURE.UNIT_MEASURE
    url2 = f"https://sdmx.oecd.org/public/rest/data/OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,/{key2}"
    fetch(url2, {"dimensionAtObservation": "AllDimensions"},
          RAW_DIR / "oecd_house_prices.csv",
          "OECD Analytical House Price Indicators")


if __name__ == "__main__":
    main()
