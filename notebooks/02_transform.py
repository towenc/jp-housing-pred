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

    RAW_FOLDER = Path(__file__).resolve().parents[1] / "data" / "raw"
    sample_path = RAW_FOLDER / "pref=30" / "year=2026" / "quarter=1.json.gz"
    return RAW_FOLDER, ds, mo, pa_json, sample_path


@app.cell
def read_sample(pa_json, sample_path):
    table = pa_json.read_json(sample_path)
    return (table,)


@app.cell
def _():
    # Your turn: inspect `table`.
    #   - how many rows?
    #   - what's the schema? (look at the types of TradePrice, Area, BuildingYear)
    #   - peek at the first 3 rows as Python dicts

    return


@app.cell
def _(table):
    table.schema
    return


@app.cell
def _(table):
    table.num_rows
    return


@app.cell
def _(table):
    table.slice(0, 3).to_pylist()
    return


@app.cell
def _(RAW_FOLDER, ds):
    pref30 = ds.dataset(
        RAW_FOLDER / "pref=30",
        format="json",
        partitioning="hive",
    )
    pref30_table = pref30.to_table()
    pref30_table.schema
    return


if __name__ == "__main__":
    app.run()
