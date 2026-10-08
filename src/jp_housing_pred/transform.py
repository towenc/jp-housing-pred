"""
Transforms raw data in .json.gz files to .parquet
"""
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_FOLDER = PROJECT_ROOT / "data" / "raw"
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "bronze"

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

if __name__ == "__main__":
    raw_to_bronze(RAW_FOLDER, OUTPUT_FOLDER)
    