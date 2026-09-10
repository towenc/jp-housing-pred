import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # The MLIT Real Estate Information Library API

    **不動産情報ライブラリ** (Real Estate Information Library) is the open data portal
    that Japan's Ministry of Land, Infrastructure, Transport and Tourism (国土交通省 /
    MLIT) launched in April 2024. It replaced the older `land.mlit.go.jp/webland` API.

    It is the source of ground truth for this project: every recorded land and
    building transaction in Japan since 2005, self-reported by buyers via a
    post-registration survey, then anonymised and published quarterly.

    This notebook covers:

    1. Authenticating and calling the API
    2. `XIT002` — looking up municipality codes
    3. `XIT001` — pulling transaction prices, the main dataset
    4. Turning the response into a tidy, numeric `DataFrame`
    5. `XPT001` — the same data as geocoded points
    6. The data quality traps that will bite a price model

    > **Prerequisite.** An API key, requested from the
    > [application form](https://www.reinfolib.mlit.go.jp/help/apiManual/). Approval is
    > by email and takes a day or two. This notebook reads it from `MLIT_API_KEY` in `.env`.
    """)
    return


@app.cell
def _():
    import json
    import os

    import altair as alt
    import httpx
    import marimo as mo
    import pandas as pd
    from dotenv import load_dotenv

    load_dotenv()
    API_KEY = os.environ["MLIT_API_KEY"]
    BASE_URL = "https://www.reinfolib.mlit.go.jp/ex-api/external"
    return API_KEY, BASE_URL, alt, httpx, json, mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. How a request is shaped

    Every endpoint is a plain HTTPS `GET` at `{BASE_URL}/{API_ID}`, authenticated with
    the key in an **`Ocp-Apim-Subscription-Key` header** (it is an Azure API Management
    front end — hence the header name). A missing or wrong key returns `401`.

    Responses are gzip-encoded; `httpx` decompresses transparently. Non-spatial
    endpoints return `{"status": "OK", "data": [...]}`, so the real payload is always
    under `data`.

    | API ID | Returns | Used below |
    |---|---|---|
    | `XIT001` | Transaction & contract prices | ✅ the main event |
    | `XIT002` | Municipality list for a prefecture | ✅ to get city codes |
    | `XPT001` | Transaction prices as GeoJSON points | ✅ for mapping |
    | `XCT001` | Appraisal reports (鑑定評価書) | — |
    | `XPT002` | Official land price surveys (地価公示) | — |
    | `XKT###` | Zoning, schools, hazard zones, demographics | — worth revisiting as features |

    The `XKT` family is the interesting long tail: flood hazard zones, station
    passenger counts, school districts. All of them are keyed by XYZ map tile rather
    than by municipality, so they join to `XPT001` points rather than to `XIT001` rows.
    """)
    return


@app.cell
def _(API_KEY, BASE_URL, httpx):
    def reinfolib(api_id: str, **params) -> dict:
        """Call one Real Estate Information Library endpoint and return parsed JSON."""
        response = httpx.get(
            f"{BASE_URL}/{api_id}",
            params=params,
            headers={"Ocp-Apim-Subscription-Key": API_KEY},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    return (reinfolib,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. `XIT002` — finding a municipality code

    `XIT001` is queried by **prefecture** (`area`, 2 digits), **municipality**
    (`city`, 5 digits) or **station** (`station`, 6 digits) — never nationwide. So the
    first job is code lookup.

    Prefecture codes are the JIS X 0401 standard ones (`01` Hokkaido … `47` Okinawa)
    and are not served by any endpoint, so they are hardcoded here. Municipality codes
    are the first 5 digits of the 全国地方公共団体コード and *are* served, by `XIT002`.
    """)
    return


@app.cell
def _():
    PREFECTURES = {
        "01": "北海道", "02": "青森県", "03": "岩手県", "04": "宮城県",
        "05": "秋田県", "06": "山形県", "07": "福島県", "08": "茨城県",
        "09": "栃木県", "10": "群馬県", "11": "埼玉県", "12": "千葉県",
        "13": "東京都", "14": "神奈川県", "15": "新潟県", "16": "富山県",
        "17": "石川県", "18": "福井県", "19": "山梨県", "20": "長野県",
        "21": "岐阜県", "22": "静岡県", "23": "愛知県", "24": "三重県",
        "25": "滋賀県", "26": "京都府", "27": "大阪府", "28": "兵庫県",
        "29": "奈良県", "30": "和歌山県", "31": "鳥取県", "32": "島根県",
        "33": "岡山県", "34": "広島県", "35": "山口県", "36": "徳島県",
        "37": "香川県", "38": "愛媛県", "39": "高知県", "40": "福岡県",
        "41": "佐賀県", "42": "長崎県", "43": "熊本県", "44": "大分県",
        "45": "宮崎県", "46": "鹿児島県", "47": "沖縄県",
    }
    return (PREFECTURES,)


@app.cell
def _(PREFECTURES, mo):
    prefecture = mo.ui.dropdown(
        options={f"{name} ({code})": code for code, name in PREFECTURES.items()},
        value="東京都 (13)",
        label="Prefecture",
    )
    prefecture
    return (prefecture,)


@app.cell
def _(pd, prefecture, reinfolib):
    municipalities = pd.DataFrame(reinfolib("XIT002", area=prefecture.value)["data"])
    municipalities
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `XIT001` — the transaction price data

    The core endpoint. Parameters:

    | Parameter | Required | Notes |
    |---|---|---|
    | `year` | ✅ | `YYYY`. Transaction prices from 2005 (Q3 onward), contract prices from 2021 |
    | `quarter` | ✅ | `1`–`4`, calendar quarters: Q1 = Jan–Mar |
    | `area` / `city` / `station` | one of | prefecture / municipality / station code |
    | `priceClassification` | optional | `01` transaction prices, `02` contract prices, omit for both |
    | `language` | optional | `ja` (default) or `en` — translates the *values*, not the keys |

    Note the granularity ceiling: one call is one quarter. Building a training set
    means looping over `(year, quarter)` pairs. Below, the query is scoped to a
    prefecture — a whole quarter of Tokyo is ~15,000 rows and about one second.
    """)
    return


@app.cell
def _(mo):
    year = mo.ui.slider(2005, 2025, value=2024, label="Year", show_value=True)
    quarter = mo.ui.dropdown(
        options={"Q1 (Jan–Mar)": 1, "Q2 (Apr–Jun)": 2, "Q3 (Jul–Sep)": 3, "Q4 (Oct–Dec)": 4},
        value="Q1 (Jan–Mar)",
        label="Quarter",
    )
    language = mo.ui.radio(
        options={"Japanese": "ja", "English": "en"}, value="Japanese", label="Value language", inline=True
    )
    mo.hstack([year, quarter, language], justify="start", gap=2)
    return language, quarter, year


@app.cell
def _(language, prefecture, quarter, reinfolib, year):
    payload = reinfolib(
        "XIT001",
        year=year.value,
        quarter=quarter.value,
        area=prefecture.value,
        language=language.value,
    )
    records = payload["data"]
    return payload, records


@app.cell(hide_code=True)
def _(json, mo, payload, records):
    mo.md(f"""
    `status: {payload["status"]}` — **{len(records):,} rows**. One raw record:

    ```json
    {json.dumps(records[0], ensure_ascii=False, indent=2) if records else "{}"}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Every value is a string

    That is the single most important thing about this response. Numbers, dates and
    categories all arrive as strings, and missing values are `"\"` rather than `null`.
    Nothing can be summed or plotted until it is coerced.

    Three fields need more than `pd.to_numeric`:

    - **`Period`** — `"2024年第1四半期"`. Parse out year and quarter.
    - **`BuildingYear`** — `"1969年"`, or the literal **`"戦前"`** ("pre-war") for
      anything before 1945. That value is a string in a numeric field and will silently
      become `NaN` if you coerce without noticing.
    - **`Area`** / **`TotalFloorArea`** — top-coded. Large parcels are published as
      **`9999`**, which means "≥5,000 m²", not 9,999 m². Left as-is it produces
      impossible price-per-m² values, so it is masked to `NaN` below.

    Prices themselves are in **yen**, unrounded in the field but bucketed by MLIT
    before publication (typically to the nearest 10,000 or 100,000 yen).
    """)
    return


@app.cell
def _(pd, records):
    NUMERIC_COLUMNS = [
        "TradePrice", "PricePerUnit", "UnitPrice", "Area", "TotalFloorArea",
        "Frontage", "Breadth", "CoverageRatio", "FloorAreaRatio",
    ]
    TOP_CODED = 9999  # sentinel for "5,000 m² or more"

    def tidy(raw: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(raw)

        for column in NUMERIC_COLUMNS:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        # Top-coded areas are censored, not measured — drop them rather than model them.
        df["AreaCapped"] = df["Area"].eq(TOP_CODED)
        df.loc[df["AreaCapped"], "Area"] = pd.NA
        df.loc[df["TotalFloorArea"].eq(TOP_CODED), "TotalFloorArea"] = pd.NA

        # "戦前" survives as a flag; the year itself is genuinely unknown.
        df["IsPrewar"] = df["BuildingYear"].isin(["戦前", "before the war"])
        df["BuildingYear"] = pd.to_numeric(
            df["BuildingYear"].str.extract(r"(\d{4})", expand=False), errors="coerce"
        )

        period = df["Period"].str.extract(r"(?P<Year>\d{4}).*?(?P<Quarter>[1-4１-４])")
        df["TradeYear"] = pd.to_numeric(period["Year"], errors="coerce")
        df["TradeQuarter"] = pd.to_numeric(
            period["Quarter"].str.translate(str.maketrans("１２３４", "1234")), errors="coerce"
        )

        # The field the model actually wants, derived rather than trusted.
        df["PricePerSqm"] = df["TradePrice"] / df["Area"]
        return df

    transactions = tidy(records)
    transactions.head(10)
    return (transactions,)


@app.cell(hide_code=True)
def _(mo, transactions):
    mo.md(f"""
    Coverage of the derived columns, which is the honest measure of how usable a
    quarter is:

    | Column | Non-null | Share |
    |---|---|---|
    | `TradePrice` | {transactions["TradePrice"].notna().sum():,} | {transactions["TradePrice"].notna().mean():.1%} |
    | `Area` | {transactions["Area"].notna().sum():,} | {transactions["Area"].notna().mean():.1%} |
    | `BuildingYear` | {transactions["BuildingYear"].notna().sum():,} | {transactions["BuildingYear"].notna().mean():.1%} |
    | `PricePerSqm` | {transactions["PricePerSqm"].notna().sum():,} | {transactions["PricePerSqm"].notna().mean():.1%} |
    | `UnitPrice` (as supplied) | {transactions["UnitPrice"].notna().sum():,} | {transactions["UnitPrice"].notna().mean():.1%} |

    Note the last row. MLIT's own `UnitPrice` is populated for only a small minority of
    records, which is why `PricePerSqm` is derived above instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What the prices look like
    """)
    return


@app.cell
def _(alt, mo, transactions):
    priced = transactions.dropna(subset=["PricePerSqm"])
    priced = priced[priced["PricePerSqm"].between(*priced["PricePerSqm"].quantile([0.01, 0.99]))]

    price_distribution = (
        alt.Chart(priced, height=260)
        .mark_bar(color="#4a6fa5", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            alt.X("PricePerSqm:Q", bin=alt.Bin(maxbins=60), title="Price per m² (¥)"),
            alt.Y("count():Q", title="Transactions"),
            tooltip=[
                alt.Tooltip("count():Q", title="Transactions"),
                alt.Tooltip("PricePerSqm:Q", bin=alt.Bin(maxbins=60), title="Price per m² (¥)", format=","),
            ],
        )
        .properties(title="Price per m², 1st–99th percentile")
        .configure_axis(grid=True, gridColor="#e8e8e8", domainColor="#c8c8c8", tickColor="#c8c8c8")
        .configure_view(stroke=None)
    )

    # mo.ui.altair_chart renders through marimo's own Vega bundle and makes the
    # selection reactive: drag across bars and `price_distribution.value` becomes
    # the selected rows, usable in any downstream cell.
    price_distribution = mo.ui.altair_chart(price_distribution)
    price_distribution
    return price_distribution, priced


@app.cell(hide_code=True)
def _(mo, price_distribution):
    mo.md(f"""
    Drag across the histogram to select a price band — marimo re-runs every dependent
    cell automatically. Currently selected: **{len(price_distribution.value):,} rows**.
    """)
    return


@app.cell
def _(alt, mo, priced):
    by_type = (
        priced.groupby("Type", as_index=False)
        .agg(MedianPricePerSqm=("PricePerSqm", "median"), Transactions=("PricePerSqm", "size"))
        .sort_values("MedianPricePerSqm", ascending=False)
    )

    type_chart = (
        alt.Chart(by_type, height=28 * len(by_type) + 40)
        .mark_bar(color="#4a6fa5", cornerRadiusTopRight=4, cornerRadiusBottomRight=4, height=18)
        .encode(
            alt.X("MedianPricePerSqm:Q", title="Median price per m² (¥)"),
            alt.Y("Type:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("Type:N", title="Property type"),
                alt.Tooltip("MedianPricePerSqm:Q", title="Median ¥/m²", format=","),
                alt.Tooltip("Transactions:Q", title="Transactions", format=","),
            ],
        )
        .properties(title="Median price per m² by property type")
        .configure_axis(grid=True, gridColor="#e8e8e8", domainColor="#c8c8c8", tickColor="#c8c8c8")
        .configure_view(stroke=None)
    )
    mo.ui.altair_chart(type_chart, chart_selection=False, legend_selection=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. `XPT001` — the same transactions, geocoded

    `XIT001` locates a sale no more precisely than a district name (`DistrictName`,
    e.g. 岩本町). For anything spatial — distance to a station, hazard zone overlap,
    neighbour-based features — use `XPT001`, which serves the same transactions as
    **GeoJSON points**, addressed by **XYZ map tile** rather than by municipality:

    | Parameter | Notes |
    |---|---|
    | `response_format` | `geojson` or `pbf` |
    | `z`, `x`, `y` | Tile coordinates. `z` must be **11–15** |
    | `from`, `to` | Period as `YYYYQ`, e.g. `20241` |
    | `priceClassification` | as `XIT001` |
    | `landTypeCode` | `01` land, `02` land+building, `07` used condo, `10` farmland, `11` forest |

    Coordinates are in **JGD2011 (EPSG:6668)**, not WGS84. The difference is
    sub-metre, but reproject rather than assume if you join against other sources.

    The tile constraint means a bounding box has to be converted to a tile list first —
    that is what `deg2tile` does below.
    """)
    return


@app.cell
def _(mo):
    import math

    def deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
        """Web Mercator tile containing a lat/lon at zoom z."""
        lat_rad = math.radians(lat)
        n = 2**z
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y

    tile_z = 13
    tile_x, tile_y = deg2tile(35.6812, 139.7671, tile_z)  # Tokyo Station
    mo.md(f"Tokyo Station sits in tile `z={tile_z}, x={tile_x}, y={tile_y}`.")
    return tile_x, tile_y, tile_z


@app.cell
def _(pd, reinfolib, tile_x, tile_y, tile_z):
    geojson = reinfolib(
        "XPT001",
        response_format="geojson",
        z=tile_z,
        x=tile_x,
        y=tile_y,
        **{"from": "20241", "to": "20244"},
    )

    points = pd.DataFrame(
        {
            **feature["properties"],
            "lon": feature["geometry"]["coordinates"][0],
            "lat": feature["geometry"]["coordinates"][1],
        }
        for feature in geojson["features"]
    )
    points[
        [
            "lat", "lon", "point_in_time_name_ja", "land_type_name_ja", "district_name_ja",
            "u_transaction_price_total_ja", "u_area_ja", "u_construction_year_ja",
        ]
    ].head(10)
    return (points,)


@app.cell(hide_code=True)
def _(mo, points):
    mo.md(f"""
    **{len(points):,} geocoded transactions** in that one tile for 2024. Note the
    property naming convention: fields are suffixed `_ja` / `_en`, and numeric-ish
    fields are prefixed `u_`. The schema does *not* match `XIT001`'s CamelCase keys, so
    the two need an explicit mapping if you use both.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Traps worth writing down

    Things that will quietly corrupt a price model:

    1. **Self-reported, not a census.** Rows come from a voluntary survey mailed to
       buyers after registration. Response rates vary by region and property type, so
       the data is a biased sample of transactions, not the population of them.
    2. **Prices are bucketed.** MLIT rounds before publishing. Treat `TradePrice` as
       having roughly ±5% of granularity noise, and don't chase precision below that.
    3. **`Area = 9999` means "≥5,000 m²".** Handled above; the same censoring applies
       to `TotalFloorArea`. Never feed the raw sentinel to a model.
    4. **`戦前` in `BuildingYear`.** A string in a numeric column. Kept as the
       `IsPrewar` flag above.
    5. **Two price types in one feed.** `PriceCategory` distinguishes 不動産取引価格情報
       (registered transactions) from 成約価格情報 (broker contract prices, 2021+).
       They are different populations — filter with `priceClassification` or split on
       the column rather than pooling them silently.
    6. **`DistrictCode` is not stable.** MLIT's own docs warn that district codes can
       change between data refreshes, so they make a poor join key across vintages.
    7. **One quarter per call.** A national panel is 47 prefectures × 4 quarters × N
       years of requests. Cache to disk; the data for a closed quarter never changes.
    8. **Latest quarter is incomplete.** Recent quarters keep gaining rows for months
       as surveys come back. Don't compare a fresh quarter against a settled one.

    ## Next steps for this project

    - A cached fetch layer over `XIT001` that loops `(prefecture, year, quarter)` and
      writes parquet, so the model trains offline.
    - Decide on the target: `TradePrice`, or the derived `PricePerSqm`? The latter
      normalises away the biggest single driver but discards every row where `Area` is
      censored.
    - Join `XKT` layers (zoning, hazard zones, station passenger counts) onto `XPT001`
      points for features that the transaction record itself does not carry.
    """)
    return


if __name__ == "__main__":
    app.run()
