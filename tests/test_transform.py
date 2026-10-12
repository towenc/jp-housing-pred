from jp_housing_pred import ingest, transform
import pyarrow.dataset as ds

# ------------ raw_to_bronze() ------------
def write_raw(raw_folder, pref, year, quarter, records):
    filename = raw_folder / f"pref={pref}" / f"year={year}" / f"quarter={quarter}" / "data.json.gz"
    ingest.save_records(records, filename)

def make_record(**kwargs):
    record = {col: "" for col in transform.RAW_COLUMNS}
    record.update(kwargs)
    return record

def test_raw_to_bronze_round_trip(tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    bronze = tmp_path / "bronze"
    hokkaido = [
        make_record(Prefecture="北海道", TradePrice="10000", BuildingYear="戦前"),
        make_record(Prefecture="北海道", TradePrice="25000", BuildingYear="1997年"),
        make_record(Prefecture="北海道", TradePrice="4000", FloorPlan="１Ｋ"),
    ]
    tokyo = [
        make_record(Prefecture="東京都", TradePrice="30000", BuildingYear="2000年"),
        make_record(Prefecture="東京都", TradePrice="35000", FloorPlan="１ＬＤＫ"),
    ]
    write_raw(raw, "01", "2024", 1, hokkaido)
    write_raw(raw, "13", "2025", 3, tokyo)

    expected = (
        [{**r, "pref":"01", "year":2024, "quarter":1} for r in hokkaido]
        + [{**r, "pref":"13", "year":2025, "quarter":3} for r in tokyo]
    )

    # Act
    transform.raw_to_bronze(raw, bronze)

    # Assert
    result = (
        ds.dataset(bronze, format="parquet", partitioning=transform.PARTITIONING)
        .to_table()
        .sort_by("TradePrice")
        .to_pylist()
    )

    assert result == sorted(expected, key=lambda r: r["TradePrice"])

