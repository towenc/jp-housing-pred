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


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the bronze layer
    """)
    return


@app.cell
def _(ds):
    from jp_housing_pred.transform import OUTPUT_FOLDER as BRONZE_FOLDER, PARTITIONING

    bronze_data = ds.dataset(
        BRONZE_FOLDER,
        format="parquet",
        partitioning=PARTITIONING,  # explicit schema: "hive" alone infers pref as int32
    )
    bronze_table = bronze_data.to_table(filter=ds.field("pref") == "13")
    bronze_data.head(5)
    return PARTITIONING, bronze_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Data Check for Bronze -> Silver
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Distinct Values for each Column
    """)
    return


@app.cell
def _(bronze_data, pa, pc):
    rows = []
    for name in bronze_data.schema.names:
        col = bronze_data.to_table(columns=[name])[name]
        rows.append({
            "column": name,
            "distinct": pc.count_distinct(col).as_py(),
            "empty_string": (pc.sum(pc.equal(col, "")).as_py() or 0)
                            if pa.types.is_string(col.type) else 0,
            "null": col.null_count,  
        })

    column_summary = pa.Table.from_pylist(rows)
    column_summary
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
def _():
    # missing = pa.table({
    #     "column": table.column_names,
    #     "empty_string": [
    #         pc.sum(pc.equal(table[c], "")).as_py() or 0
    #         if pa.types.is_string(table[c].type) else 0
    #         for c in table.column_names
    #     ],
    #     "null": [table[c].null_count for c in table.column_names],

    # })
    # missing
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Empty strings by `Type`

    Fraction of rows with `"\"` for each (column, Type) pair. Near 0% or 100% means the
    field always / never applies to that kind of transaction, rather than being
    genuinely missing.
    """)
    return


@app.cell
def _(PARTITIONING, bronze_data, pa, pc):
    _partition_cols = set(PARTITIONING.schema.names)
    _rows = []
    for _name in bronze_data.schema.names:
        if _name in _partition_cols or _name == "Type":
            continue
        _t = bronze_data.to_table(columns=["Type", _name])
        _t = _t.append_column("is_empty", pc.equal(_t[_name], ""))
        _by_type = _t.group_by("Type").aggregate([("is_empty", "mean")])
        for _r in _by_type.to_pylist():
            _rows.append({"column": _name, "Type": _r["Type"], "frac_empty": _r["is_empty_mean"]})

    empty_by_type = pa.Table.from_pylist(_rows).to_pandas()
    return (empty_by_type,)


@app.cell
def _(empty_by_type):
    import altair as alt

    alt.Chart(empty_by_type).mark_rect(stroke="white", strokeWidth=1).encode(
        x=alt.X("Type:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("column:N", title=None, sort=list(empty_by_type["column"].unique())),
        color=alt.Color(
            "frac_empty:Q",
            title="Share empty",
            scale=alt.Scale(scheme="blues", domain=[0, 1]),
            legend=alt.Legend(format="%"),
        ),
        tooltip=[
            "column:N",
            "Type:N",
            alt.Tooltip("frac_empty:Q", title="Share empty", format=".1%"),
        ],
    ).properties(width=420, height=alt.Step(18), title="Share of empty strings by column and Type")
    return


if __name__ == "__main__":
    app.run()
