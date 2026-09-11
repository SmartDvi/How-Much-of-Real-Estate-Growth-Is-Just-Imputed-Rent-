"""
Load raw OECD CSV pulls into a clean DuckDB database and build analytical views.

Output: data/processed/real_estate.duckdb
"""
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
DB_PATH = ROOT / "data" / "processed" / "real_estate.duckdb"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect(str(DB_PATH))

# ---------------------------------------------------------------------------
# 1. Raw ingestion — split "CODE: Label" cells that OECD's labels=both CSV
#    format produces, for both column headers and cell values.
# ---------------------------------------------------------------------------
con.execute("PRAGMA disable_progress_bar")

con.execute(f"""
    CREATE OR REPLACE TABLE raw_value_added AS
    SELECT * FROM read_csv_auto('{RAW / "oecd_nad_value_added.csv"}', header=True, all_varchar=True)
""")
con.execute(f"""
    CREATE OR REPLACE TABLE raw_house_prices AS
    SELECT * FROM read_csv_auto('{RAW / "oecd_house_prices.csv"}', header=True, all_varchar=True)
""")


def split_code_label(con, table, out_table, code_label_pairs):
    """code_label_pairs: list of (raw_col_prefix, clean_code_col)"""
    cols = con.execute(f"PRAGMA table_info('{table}')").fetchdf()["name"].tolist()

    select_parts = []
    for col in cols:
        base = col.split(":")[0].strip()
        select_parts.append(f'"{col}" AS "{base}_RAW"')
    con.execute(f'CREATE OR REPLACE TABLE {out_table}_stage AS SELECT {", ".join(select_parts)} FROM {table}')
    return [c.split(":")[0].strip() for c in cols]


# Value added table -----------------------------------------------------
cols = con.execute("PRAGMA table_info('raw_value_added')").fetchdf()["name"].tolist()
base_names = [c.split(":")[0].strip() for c in cols]
select_sql = []
for raw_col, base in zip(cols, base_names):
    code_expr = f'split_part("{raw_col}", \':\', 1)'
    select_sql.append(f'trim({code_expr}) AS {base}')
    if base in ("REF_AREA", "ACTIVITY", "TRANSACTION", "PRICE_BASE", "VALUATION", "UNIT_MEASURE"):
        label_expr = f'trim(split_part("{raw_col}", \':\', 2))'
        select_sql.append(f'{label_expr} AS {base}_LABEL')

con.execute(f"""
    CREATE OR REPLACE TABLE value_added_clean AS
    SELECT {", ".join(select_sql)}
    FROM raw_value_added
""")
con.execute("""
    ALTER TABLE value_added_clean ALTER COLUMN TIME_PERIOD TYPE INTEGER
""")
con.execute("""
    ALTER TABLE value_added_clean ALTER COLUMN OBS_VALUE TYPE DOUBLE
""")
con.execute("""
    ALTER TABLE value_added_clean ALTER COLUMN UNIT_MULT TYPE INTEGER
""")
con.execute("""
    CREATE OR REPLACE TABLE value_added_clean AS
    SELECT *, OBS_VALUE * POWER(10, UNIT_MULT) AS VALUE_ACTUAL
    FROM value_added_clean
""")

# House prices table ------------------------------------------------------
cols2 = con.execute("PRAGMA table_info('raw_house_prices')").fetchdf()["name"].tolist()
base_names2 = [c.split(":")[0].strip() for c in cols2]
select_sql2 = []
for raw_col, base in zip(cols2, base_names2):
    code_expr = f'split_part("{raw_col}", \':\', 1)'
    select_sql2.append(f'trim({code_expr}) AS {base}')
    if base in ("REF_AREA", "MEASURE", "UNIT_MEASURE", "ADJUSTMENT"):
        label_expr = f'trim(split_part("{raw_col}", \':\', 2))'
        select_sql2.append(f'{label_expr} AS {base}_LABEL')

con.execute(f"""
    CREATE OR REPLACE TABLE house_prices_clean AS
    SELECT {", ".join(select_sql2)}
    FROM raw_house_prices
""")
con.execute("ALTER TABLE house_prices_clean ALTER COLUMN TIME_PERIOD TYPE INTEGER")
con.execute("ALTER TABLE house_prices_clean ALTER COLUMN OBS_VALUE TYPE DOUBLE")

print("value_added_clean:", con.execute("SELECT COUNT(*) FROM value_added_clean").fetchone()[0], "rows")
print("house_prices_clean:", con.execute("SELECT COUNT(*) FROM house_prices_clean").fetchone()[0], "rows")

# ---------------------------------------------------------------------------
# 2. Analytical fact table: one row per country/year/price_base with the
#    four activity values pivoted into columns.
# ---------------------------------------------------------------------------
con.execute("""
    CREATE OR REPLACE TABLE fact_real_estate AS
    WITH piv AS (
        SELECT
            REF_AREA,
            REF_AREA_LABEL,
            TIME_PERIOD,
            PRICE_BASE,
            PRICE_BASE_LABEL,
            MAX(CASE WHEN ACTIVITY = '_T'   THEN VALUE_ACTUAL END) AS gdp_basic_prices,
            MAX(CASE WHEN ACTIVITY = 'L'    THEN VALUE_ACTUAL END) AS real_estate_total,
            MAX(CASE WHEN ACTIVITY = 'L68A' THEN VALUE_ACTUAL END) AS imputed_rent,
            MAX(CASE WHEN ACTIVITY = 'L68B' THEN VALUE_ACTUAL END) AS real_estate_excl_imputed
        FROM value_added_clean
        GROUP BY 1,2,3,4,5
    )
    SELECT
        *,
        imputed_rent + real_estate_excl_imputed AS real_estate_check_sum,
        CASE WHEN real_estate_total > 0 THEN imputed_rent / real_estate_total END AS imputed_share_of_re,
        CASE WHEN gdp_basic_prices > 0 THEN real_estate_total / gdp_basic_prices END AS re_share_of_gdp,
        CASE WHEN gdp_basic_prices > 0 THEN imputed_rent / gdp_basic_prices END AS imputed_share_of_gdp,
        CASE WHEN gdp_basic_prices > 0 THEN real_estate_excl_imputed / gdp_basic_prices END AS re_excl_imputed_share_of_gdp
    FROM piv
    WHERE real_estate_total IS NOT NULL
    ORDER BY REF_AREA, TIME_PERIOD, PRICE_BASE
""")

n = con.execute("SELECT COUNT(*) FROM fact_real_estate").fetchone()[0]
n_countries_current = con.execute(
    "SELECT COUNT(DISTINCT REF_AREA) FROM fact_real_estate WHERE PRICE_BASE='V'"
).fetchone()[0]
print(f"fact_real_estate: {n} rows, {n_countries_current} countries (current prices)")

# Coverage summary for later reference
coverage = con.execute("""
    SELECT REF_AREA, REF_AREA_LABEL,
           MIN(TIME_PERIOD) AS min_year, MAX(TIME_PERIOD) AS max_year,
           COUNT(*) AS n_years
    FROM fact_real_estate
    WHERE PRICE_BASE = 'V'
    GROUP BY 1,2
    ORDER BY n_years DESC
""").fetchdf()
coverage.to_csv(ROOT / "data" / "processed" / "country_coverage.csv", index=False)
print(coverage.head(15).to_string(index=False))

con.close()
print(f"\nDuckDB database written to {DB_PATH}")
