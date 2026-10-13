from jp_housing_pred import ingest, transform
import pyarrow as pa
import pyarrow.dataset as ds

# ------------ raw_to_bronze() ------------
def write_raw(raw_folder, pref, year, quarter, records):
    filename = raw_folder / f"pref={pref}" / f"year={year}" / f"quarter={quarter}" / "data.json.gz"
    ingest.save_records(records, filename)

def make_record(**kwargs):
    record = {col: "" for col in transform.RAW_COLUMNS}
    record.update(kwargs)
    return record

def test_raw_to_bronze(tmp_path):
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

# ------------ empty_to_null() ------------
def test_empty_to_null():
    # Arrange
    arr = pa.chunked_array([["", "東京都", None, " "]])

    # Act 
    result = transform.empty_to_null(arr)

    # Assert
    assert result.to_pylist() == [None, "東京都", None, " "]
    assert result.type == pa.string()

# ------------ empty_strings_to_null() ------------
def test_empty_strings_to_null_only_touches_string_cols():
    # Arrange
    table = pa.table({
        "Prefecture": ["", "北海道", "東京都"],
        "year": [2023, 2024, 2025],
    })

    # Act
    result = transform.empty_strings_to_null(table)

    # Assert
    assert result.column("Prefecture").to_pylist() == [None, "北海道", "東京都"]
    assert result.column("year").to_pylist() == [2023, 2024, 2025]
    assert result.schema == table.schema

# ------------ normalize_text() ------------
def test_normalize_text():
    # Arrange
    table = pa.table({
        "FloorPlan": [
            "３ＬＤＫ",                  # full-width ASCII
            "第２種低層住居専用地域",      # full-width digit inside Japanese text
            "東京都",                    # already normal
            None,
        ],
        "year": [2022, 2023, 2024, 2025],
    })

    expected = ["3LDK", "第2種低層住居専用地域", "東京都", None]

    # Act
    result = transform.normalize_text(table)

    # Assert
    assert result.column("FloorPlan").to_pylist() == expected
    assert result.column("year").equals(table.column("year"))
    assert result.schema == table.schema

# ------------ cast_numeric() ------------
def test_cast_numeric():
    # Arrange
    table = pa.table({
        "Prefecture": [None, "北海道", "東京都"],
        "year": [2023, 2024, 2025],
        "TradePrice": [None, "10000", "500000"],
        "Area": ["400", "150.5", "340"]
    })

    columns = {
        "TradePrice": pa.int64(),
        "Area": pa.float64()
    }

    # Act
    result = transform.cast_numeric(table, columns)

    # Assert
    assert result.schema.field("TradePrice").type == pa.int64()
    assert result.schema.field("Area").type == pa.float64()
    assert result.column("TradePrice").to_pylist() == [None, 10000, 500000]
    assert result.column("Area").to_pylist() == [400.0, 150.5, 340.0]
    assert result.column("Prefecture").equals(table.column("Prefecture"))
    assert result.column("year").equals(table.column("year"))

# ------------ parse_building_year() ------------
def test_parse_building_year():
    # Arrange
    table = pa.table({
        "BuildingYear": ["1974年", "戦前", None, "2001年"],
        "Prefecture": ["東京都", "北海道", "大阪府", "福島県"], 
    })

    # Act
    result = transform.parse_building_year(table)

    # Assert
    assert result.schema.field("BuildingYear").type == pa.int16()
    assert result.schema.field("prewar").type == pa.bool_()
    assert result.column("BuildingYear").to_pylist() == [1974, transform.PREWAR_YEAR, None, 2001]
    assert result.column("prewar").to_pylist() == [False, True, None, False]
    assert result.column("Prefecture").equals(table.column("Prefecture"))

# ------------ bronze_to_silver() ------------
def read_silver(silver):
    return ds.dataset(silver, format="parquet", partitioning=transform.PARTITIONING).to_table()

def test_bronze_to_silver(tmp_path):
    # Arrange
    raw = tmp_path / "raw"
    bronze = tmp_path / "bronze"
    silver = tmp_path / "silver"

    write_raw(raw, "01", "2024", 1, [
        make_record(TradePrice="10000", BuildingYear="戦前"),
        make_record(TradePrice="25000", BuildingYear="1997年"),
    ])
    write_raw(raw, "13", "2025", 3, [
        make_record(TradePrice="30000", FloorPlan="１ＬＤＫ"),
    ])
    transform.raw_to_bronze(raw, bronze)

    # Act
    transform.bronze_to_silver(bronze, silver)

    # Assert
    result = read_silver(silver).sort_by("TradePrice")

    assert result.schema.field("TradePrice").type == pa.int64()
    assert result.schema.field("BuildingYear").type == pa.int16()
    assert result.schema.field("prewar").type == pa.bool_()

    columns = ["pref", "year", "quarter", "TradePrice", "BuildingYear", "prewar", "FloorPlan"]
    assert result.select(columns).to_pylist() == [
        {"pref": "01", "year": 2024, "quarter": 1, "TradePrice": 10000,
         "BuildingYear": transform.PREWAR_YEAR, "prewar": True, "FloorPlan": None},
        {"pref": "01", "year": 2024, "quarter": 1, "TradePrice": 25000,
         "BuildingYear": 1997, "prewar": False, "FloorPlan": None},
        {"pref": "13", "year": 2025, "quarter": 3, "TradePrice": 30000,
         "BuildingYear": None, "prewar": None, "FloorPlan": "1LDK"},
    ]