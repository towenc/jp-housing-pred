import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 02 · Raw → bronze transform

    Exploring how to turn `data/raw/**/*.json.gz` (gzipped JSONL) into parquet,
    one step at a time. Anything that works here graduates into
    `src/jp_housing_pred/transform.py`.

    **Step 1:** read *one* file and understand what pyarrow gives back.
    """)
    return


@app.cell
def _():
    import marimo as mo
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.json as pa_json
    import pyarrow.dataset as ds
    import pyarrow.compute as pc

    RAW_FOLDER = Path(__file__).resolve().parents[1] / "data" / "raw"
    sample_path = RAW_FOLDER / "pref=13" / "year=2023" / "quarter=1" / "data.json.gz"
    return RAW_FOLDER, ds, mo, pa, pc


@app.cell
def _(RAW_FOLDER, ds, pa):
    part = ds.partitioning(
        pa.schema([
            ("pref", pa.string()),
            ("year", pa.int16()),
            ("quarter", pa.int8())
        ]),
        flavor="hive"
    )

    data = ds.dataset(
        RAW_FOLDER,
        format="json",
        partitioning=part,
    )
    table = data.to_table(filter=ds.field("pref") == "13")
    #table.schema
    data.head(5)
    return (table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Check
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check Numeric Columns
    """)
    return


@app.cell
def _(pc, table):
    numeric_cols = ["TradePrice", "Area", "TotalFloorArea", "Frontage", "BuildingYear"]

    for col_name in numeric_cols:
        col = table[col_name]

        is_number = pc.match_substring_regex(col, r"^[0-9]+(\.[0-9]+)?$")
        not_number = pc.invert(is_number)
        bad_values = pc.filter(col, not_number)

        print(col_name, len(bad_values))
        print(pc.value_counts(bad_values).to_pylist()[:10])

    buildingyear_values = pc.match_substring_regex(table["BuildingYear"], r"^([0-9]{4}年)?$")
    not_valid = pc.invert(buildingyear_values)
    bad_values = pc.filter(table["BuildingYear"], not_valid)

    print(pc.value_counts(bad_values).to_pylist()[:10])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    TotalFloorArea and Frontage have missing values as empty strings.

    BuildingYear have 年 attached to the values and also 戦前 values.
    """)
    return


@app.cell
def _(pa, pc, table):
    missing = pa.table({
        "column": table.column_names,
        "empty_string": [
            pc.sum(pc.equal(table[c], "")).as_py() or 0
            if pa.types.is_string(table[c].type) else 0
            for c in table.column_names
        ],
        "null": [table[c].null_count for c in table.column_names],
        
    })
    missing
    return


if __name__ == "__main__":
    app.run()
