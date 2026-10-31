import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 03 Explore Sliver Data and Build Features
    """)
    return


@app.cell
def _():
    import marimo as mo
    import duckdb
    import altair as alt

    return alt, duckdb, mo


@app.cell
def _(duckdb, mo):
    SILVER_GLOB = str(mo.notebook_dir().parent / "data" / "silver" / "**" / "*.parquet")
    silver = duckdb.read_parquet(SILVER_GLOB, hive_partitioning=True)
    return (silver,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## EDA
    """)
    return


@app.cell
def _(mo, silver):
    type_counts = mo.sql(
        f"""
        SELECT Type, count(*) AS n
        FROM silver
        GROUP BY Type
        ORDER BY n DESC
        """
    )
    return


@app.cell
def _(silver):
    HOME_TYPES = ("中古マンション等", "宅地(土地と建物)")
    homes = silver.filter(f"Type IN {HOME_TYPES}")
    return (homes,)


@app.cell
def _(homes, mo):
    homes_summary = mo.sql(
        f"""
        SUMMARIZE homes
        """
    )
    return


@app.cell
def _(homes, mo):
    _df = mo.sql(
        f"""
        SELECT
            year,
            count(*) FILTER (WHERE PriceCategory = '不動産取引価格情報') AS n_transaction,
            count(*) FILTER (WHERE PriceCategory = '成約価格情報')       AS n_contract,
            count(*)                                                      AS n_total
        FROM homes
        GROUP BY year
        ORDER BY year
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are two types of data.
    Survey based prices (不動産取引価格情報) and actual contract contract prices(成約価格情報).
    Contract prices only 2020.
    """)
    return


@app.cell
def _(homes, mo):
    _df = mo.sql(
        f"""
        WITH groups AS (
            SELECT
                count(*)                      AS n_rows,
                count(DISTINCT PriceCategory) AS n_categories
            FROM homes
            WHERE year >= 2021          -- contract prices only exist from 2021
            GROUP BY Type, pref, Municipality, DistrictName, year, quarter,
                     Area, FloorPlan, BuildingYear, TradePrice
        )
        SELECT
            count(*)                                     AS n_groups,
            count(*) FILTER (WHERE n_categories > 1)     AS n_cross_category,
            sum(n_rows) FILTER (WHERE n_categories > 1)  AS rows_in_cross
        FROM groups
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Check if there are duplicates within the two categories. There are about 6,651 rows. We cannot garantee that they are duplicates so they are left in the dataset. It is a very small amount of data anyway
    """)
    return


@app.cell
def _(homes, mo):
    _df = mo.sql(
        f"""
        SELECT Remarks, count(*) AS n
        FROM homes
        WHERE Remarks IS NOT NULL
        GROUP BY Remarks
        ORDER BY n DESC
        """
    )
    return


@app.cell
def _(homes, mo):
    _df = mo.sql(
        f"""
        WITH priced AS (
            -- price per m², skipping rows with no usable area
            SELECT Type, Remarks, TradePrice / Area AS price_per_m2
            FROM homes
            WHERE Area > 0
        ),
        baseline AS (
            -- the "normal" price: rows with no remarks at all
            SELECT Type, median(price_per_m2) AS baseline_median
            FROM priced
            WHERE Remarks IS NULL
            GROUP BY Type
        ),
        by_remark AS (
            -- one row per (sale, remark), so combined remarks count toward each one
            SELECT Type, unnest(string_split(Remarks, '、')) AS remark, price_per_m2
            FROM priced
            WHERE Remarks IS NOT NULL
        )
        SELECT
            r.Type,
            r.remark,
            count(*)                                                        AS n,
            round(median(r.price_per_m2))                                   AS remark_median,
            round(any_value(b.baseline_median))                             AS baseline_median,
            round(median(r.price_per_m2) / any_value(b.baseline_median), 2) AS ratio
        FROM by_remark r
        JOIN baseline b USING (Type)
        GROUP BY r.Type, r.remark
        ORDER BY r.Type, ratio
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    調停・競売等, 関係者間取引, 瑕疵有りの可能性, 他の権利・負担付き, その他事情有り, 古屋付き・取壊し前提. drop these
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q4: TradePrice distribution
    """)
    return


@app.cell
def _(homes):
    DROP_REMARKS = [
        "調停・競売等", "関係者間取引", "瑕疵有りの可能性",
        "他の権利・負担付き", "その他事情有り", "古屋付き・取壊し前提",
    ]
    # Drop a row if ANY of its 、-separated remarks is on the list.
    # coalesce(..., false): rows with no remarks give NULL, which would otherwise be dropped too.
    clean = homes.filter(
        f"NOT coalesce(list_has_any(string_split(Remarks, '、'), {DROP_REMARKS}), false)"
    )
    return DROP_REMARKS, clean


@app.cell
def _(clean, mo):
    raw_bins = mo.sql(
        f"""
        SELECT
            Type,
            floor(TradePrice / 5e6) * 5 AS price_bin_m,
            count(*) AS n
        FROM clean
        WHERE TradePrice < 200e6
        GROUP BY ALL
        ORDER BY Type, price_bin_m
        """
    )
    return (raw_bins,)


@app.cell
def _(clean, mo):
    price_bins = mo.sql(
        f"""
        SELECT
            Type,
            floor(log10(TradePrice) * 10) / 10 AS log10_bin,  -- bins 0.1 wide in log10 space
            count(*) AS n
        FROM clean
        WHERE TradePrice > 0
        GROUP BY ALL
        ORDER BY Type, log10_bin
        """
    )
    return (price_bins,)


@app.cell
def _(alt, price_bins, raw_bins):
    _raw = alt.Chart(raw_bins, title="Raw price (< ¥200M)").mark_bar().encode(
        x=alt.X("price_bin_m:Q", title="TradePrice (¥M)"),
        y=alt.Y("n:Q", title="transactions"),
        row=alt.Row("Type:N", title=None),
    ).properties(width=350, height=180)

    _log = alt.Chart(price_bins, title="log10 price").mark_bar().encode(
        x=alt.X("log10_bin:Q", title="log10(TradePrice)  —  7 = ¥10M, 8 = ¥100M"),
        y=alt.Y("n:Q", title="transactions"),
        row=alt.Row("Type:N", title=None),
    ).properties(width=350, height=180)

    _raw | _log
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Predict log(TradePrice) since TradePrice is skewed with long right tail.gtiv
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q5: What does Area mean?

    - Condos: `Area` = unit floor area, `TotalFloorArea` always null.
    - Houses: `Area` = **land** area, `TotalFloorArea` = building floor area.
    - `9999` is the API's code for "2,000 m² or more", not a real size.
    """)
    return


@app.cell
def _(clean, mo):
    area_sentinels = mo.sql(
        f"""
        -- How many 9999 codes (= ≥2,000 m²) and nulls per type and column?
        SELECT
            Type,
            count(*)                                           AS n,
            count(*) FILTER (WHERE Area = 9999)                AS area_9999,
            count(*) FILTER (WHERE TotalFloorArea = 9999)      AS floor_9999,
            count(*) FILTER (WHERE TotalFloorArea IS NULL)     AS floor_null,
            round(100 * count(*) FILTER (WHERE TotalFloorArea IS NULL) / count(*), 1) AS floor_null_pct
        FROM clean
        GROUP BY Type
        """
    )
    return


@app.cell
def _(clean, mo):
    area_quantiles = mo.sql(
        f"""
        -- Real sizes only: NULLIF turns 9999 into NULL, and aggregates skip NULLs.
        SELECT
            Type,
            quantile_cont(NULLIF(Area, 9999),           [0.001, 0.01, 0.5, 0.99, 0.999]) AS area_q,
            max(NULLIF(Area, 9999))                                                    AS area_max,
            quantile_cont(NULLIF(TotalFloorArea, 9999), [0.001, 0.01, 0.5, 0.99, 0.999]) AS floor_q,
            max(NULLIF(TotalFloorArea, 9999))                                          AS floor_max
        FROM clean
        GROUP BY Type
        """
    )
    return


@app.cell
def _(clean, mo):
    missing_floor_sample = mo.sql(
        f"""
        -- Houses with no building floor area: what do they look like?
        SELECT * FROM (
            SELECT
                pref, Municipality, year, Area, TotalFloorArea, BuildingYear,
                Structure, FloorPlan, Use, TradePrice, Remarks
            FROM clean
            WHERE Type = '宅地(土地と建物)' AND TotalFloorArea IS NULL
        )
        USING SAMPLE 20 ROWS
        """
    )
    return


@app.cell
def _(DROP_REMARKS, homes, mo):
    drop_funnel = mo.sql(
        f"""
        -- How many rows each gold drop rule removes, applied in order, per type.
        -- Each kept_* column is the rows still left AFTER that step (and all steps before it).
        WITH flagged AS (
            SELECT
                Type,
                coalesce(list_has_any(string_split(Remarks, '、'), {DROP_REMARKS}), false) AS bad_remark,
                Area = 9999                                                    AS area_9999,
                coalesce(TotalFloorArea = 9999, false)                         AS floor_9999,
                -- the split: condos take floor area from Area, houses from TotalFloorArea
                CASE WHEN Type = '中古マンション等' THEN Area ELSE TotalFloorArea END IS NULL AS no_floor_area
            FROM homes
        )
        SELECT
            Type,
            count(*)                                                                  AS n_start,
            count(*) FILTER (WHERE NOT bad_remark)                                    AS kept_after_remarks,
            count(*) FILTER (WHERE NOT bad_remark AND NOT area_9999)                  AS kept_after_area_9999,
            count(*) FILTER (WHERE NOT bad_remark AND NOT area_9999 AND NOT floor_9999) AS kept_after_floor_9999,
            count(*) FILTER (WHERE NOT bad_remark AND NOT area_9999 AND NOT floor_9999
                                   AND NOT no_floor_area)                             AS n_final,
            round(100 * n_final / n_start, 1)                                         AS pct_kept
        FROM flagged
        GROUP BY Type
        ORDER BY Type
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q6: FloorPlan

    Format is `<rooms><L/D/K/R>[+S]`, e.g. `3LDK+S` = 3 rooms + Living, Dining, Kitchen + Storage room (納戸).
    """)
    return


@app.cell
def _(clean, mo):
    floorplan_fill = mo.sql(
        f"""
        -- How often is FloorPlan filled, and how many distinct values, per type?
        SELECT
            Type,
            count(*)                                       AS n,
            round(100 * count(FloorPlan) / count(*), 1)    AS pct_filled,
            count(DISTINCT FloorPlan)                      AS n_distinct
        FROM clean
        GROUP BY Type
        """
    )
    return


@app.cell
def _(clean, mo):
    floorplan_counts = mo.sql(
        f"""
        -- Every condo FloorPlan value with its share of rows and median price
        SELECT
            FloorPlan,
            count(*)                                    AS n,
            round(100 * count(*) / sum(count(*)) OVER (), 2) AS pct,
            median(TradePrice)                          AS median_price
        FROM clean
        WHERE Type = '中古マンション等'
        GROUP BY FloorPlan
        ORDER BY n DESC
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Need to convert FloorPlan to n_rooms, has_L, has_D and has_K, has_S. 1R should just be a one room. for スタジオ treat as 1R, オープンフロア should have a flag is_open_floor with null n_rooms. メゾネット should have null rooms and no flags
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Q7: Building age

    `age = year - BuildingYear`. Prewar buildings were coded as 1945 in silver (`prewar` flag), so their age is a lower bound.
    """)
    return


@app.cell
def _(clean, mo):
    age_check = mo.sql(
        f"""
        -- Building-age problems that remain AFTER the Q3/Q5 drop rules.
        WITH kept AS (
            SELECT
                Type, year, BuildingYear, prewar,
                year - BuildingYear AS age
            FROM clean                               -- Q3: remarks already dropped
            WHERE Area != 9999                       -- Q5: area 9999 code
              AND coalesce(TotalFloorArea, 0) != 9999
              AND CASE WHEN Type = '中古マンション等' THEN Area ELSE TotalFloorArea END IS NOT NULL
        )
        SELECT
            Type,
            count(*)                                   AS n,
            count(*) FILTER (WHERE BuildingYear IS NULL) AS year_missing,
            round(100 * year_missing / n, 2)           AS year_missing_pct,
            count(*) FILTER (WHERE prewar)             AS n_prewar,
            count(*) FILTER (WHERE age = -1)           AS age_minus_1,
            count(*) FILTER (WHERE age < -1)           AS age_below_minus_1,
            count(*) FILTER (WHERE age = 0)            AS age_0
        FROM kept
        GROUP BY Type
        ORDER BY Type
        """
    )
    return


if __name__ == "__main__":
    app.run()
