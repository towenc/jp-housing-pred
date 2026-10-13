"""
Transforms raw data in .json.gz files to .parquet
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# ----- Constants -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "bronze"
SILVER_FOLDER = PROJECT_ROOT / "data" / "silver"

# ----- Raw to Bronze -----
RAW_COLUMNS = [
    "PriceCategory", "Type", "Region", "MunicipalityCode", "Prefecture",
    "Municipality", "DistrictName", "TradePrice", "PricePerUnit", "FloorPlan",
    "Area", "UnitPrice", "LandShape", "Frontage", "TotalFloorArea",
    "BuildingYear", "Structure", "Use", "Purpose", "Direction",
    "Classification", "Breadth", "CityPlanning", "CoverageRatio",
    "FloorAreaRatio", "Period", "Renovation", "Remarks", "DistrictCode",
]

PARTITIONING = ds.partitioning(
    pa.schema([
        ("pref", pa.string()),
        ("year", pa.int16()),
        ("quarter", pa.int8())
    ]),
    flavor="hive",
)

RAW_SCHEMA = pa.schema(
    [(col, pa.string()) for col in RAW_COLUMNS] + list(PARTITIONING.schema)
)

def raw_to_bronze(raw_folder: Path, output_folder: Path) -> None:
    """Converts raw gzipped JSONL to parquet, keeping hive partitions"""
    files = sorted(
        str(p) for p in Path(raw_folder).glob("pref=*/year=*/quarter=*/data.json.gz")
    )

    raw = ds.dataset(
        files,
        format="json",
        schema=RAW_SCHEMA,
        partitioning=PARTITIONING,
        partition_base_dir=str(raw_folder),
    )

    ds.write_dataset(
        raw,
        output_folder,
        format="parquet",
        partitioning=PARTITIONING,
        existing_data_behavior="delete_matching",
    )


# ----- Bronze to Silver -----
NUMERIC_COLUMNS = {
    "TradePrice": pa.int64(),
    "Area": pa.float64(),
    "UnitPrice": pa.float64(),
    "PricePerUnit": pa.float64(),
    "Frontage": pa.float64(),
    "TotalFloorArea": pa.float64(),
    "Breadth": pa.float64(),
    "CoverageRatio": pa.float64(),
    "FloorAreaRatio": pa.float64(),
}

# BuildingYear "戦前"
PREWAR_YEAR = 1945

def empty_to_null(arr: pa.ChunkedArray) -> pa.ChunkedArray:
    """Turns empty strings "" to real null value"""
    return pc.if_else(pc.equal(arr, ""), pa.scalar(None, arr.type), arr)

def empty_strings_to_null(table: pa.Table) -> pa.Table:
    """Apply empty_to_null to every string column; leave other types alone"""
    for idx, field in enumerate(table.schema):
        if pa.types.is_string(field.type):
            table = table.set_column(idx, field.name, empty_to_null(table[field.name]))
    return table

def normalize_text(table: pa.Table) -> pa.Table:
    """Normalize full-width text to half-width, e.g. ３ＬＤＫ -> 3LDK, 第２種 -> 第2種.
    Note: Arrow's NFKC doesn't recompose kana (ひ + U+3099 stays split); MLIT data has none."""
    for idx, field in enumerate(table.schema):
        if pa.types.is_string(field.type):
            normalized = pc.utf8_normalize(table[field.name], form="NFKC")
            table = table.set_column(idx, field.name, normalized)
    return table

def cast_numeric(table: pa.Table, columns: dict[str, pa.DataType]) -> pa.Table:
    """Cast string columns to numeric types. Run empty_strings_to_null first"""
    for col, dtype in columns.items():
        idx = table.schema.get_field_index(col)
        table = table.set_column(idx, col, pc.cast(table[col], dtype))
    return table

def parse_building_year(table: pa.Table) -> pa.Table:
    """BuildingYear "1974年" -> 1974 as int16; "戦前" -> PREWAR_YEAR plus a `prewar` flag.
    Run empty_strings_to_null first so missing years stay null."""
    col = table["BuildingYear"]
    prewar = pc.equal(col, "戦前")
    digits = pc.if_else(prewar, str(PREWAR_YEAR), pc.replace_substring(col, "年", ""))
    idx = table.schema.get_field_index("BuildingYear")
    table = table.set_column(idx, "BuildingYear", pc.cast(digits, pa.int16()))
    return table.append_column("prewar", prewar)

def clean(table: pa.Table) -> pa.Table:
    """Bronze -> silver cleaning steps, in the order they must run"""
    table = empty_strings_to_null(table)
    table = normalize_text(table)
    table = cast_numeric(table, NUMERIC_COLUMNS)
    table = parse_building_year(table)
    return table

def bronze_to_silver(bronze_folder: Path, output_folder: Path) -> None:
    """Cleans bronze parquet batch by batch and writes silver, keeping hive partitions"""
    bronze = ds.dataset(bronze_folder, format="parquet", partitioning=PARTITIONING)

    # bronze data is streamed in batches so whole dataset isn't loaded in data at once
    silver_schema = clean(bronze.head(0)).schema
    batches = (clean(batch) for batch in bronze.to_batches())

    ds.write_dataset(
        batches,
        output_folder,
        schema=silver_schema,
        format="parquet",
        partitioning=PARTITIONING,
        existing_data_behavior="delete_matching",
    )

if __name__ == "__main__":
    print("Raw Data -> Bronze Conversion...")
    raw_to_bronze(RAW_FOLDER, OUTPUT_FOLDER)
    print("Bronze -> Silver Conversion...")
    bronze_to_silver(OUTPUT_FOLDER, SILVER_FOLDER)
    print("Done.")