"""
Converts silver data to gold layer which is ready for modelling
"""

from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

from jp_housing_pred.transform import PARTITIONING

# ----- Constants -----
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_FOLDER = PROJECT_ROOT / "data" / "gold"
SILVER_FOLDER = PROJECT_ROOT / "data" / "silver"

KEEP_TYPES = ["中古マンション等", "宅地(土地と建物)"]

# A row is dropped if Remarks contains ANY of these (Remarks is "、"-joined)
DROP_REMARKS = [
    "調停・競売等",
    "関係者間取引",
    "瑕疵有りの可能性",
    "他の権利・負担付き",
    "その他事情有り",
    "古屋付き・取壊し前提",
]
 
MIN_TRADE_PRICE = 1000000 

# Area being 9999 means area is actually greater than 2000m2
AREA_CAP = 9999.0  

GOLD_COLUMNS: list[str] = ["log_price", 
                            "Type", 
                            "PriceCategory", 
                            "pref", 
                            "year",
                            "quarter",
                            "MunicipalityCode",
                            "DistrictCode",
                            "Region",
                            "floor_area",
                            "land_area",
                            "n_rooms", "has_L", "has_D", "has_K", "has_S", "is_open_floor",
                            "age", "new_buid", "prewar", "Structure", "Renovation",
                            "CityPlanning", "CoverageRatio", "FloorAreaRatio",
                            "LandShape", "Frontage", "Direction", "Classification", "Breadth"
]   

def filter_types(table: pa.Table) -> pa.Table:
    """Keep only condos and land+building (houses)."""
    condition = pc.field("Type").isin(KEEP_TYPES)
    return table.filter(condition)


def drop_bad_remarks(table: pa.Table) -> pa.Table:
    """Drop rows whose Remarks mention any of DROP_REMARKS. Null Remarks are kept."""
    has_bad = pc.match_substring(pc.field("Remarks"), DROP_REMARKS[0])
    for remark in DROP_REMARKS[1:]:
        has_bad = has_bad | pc.match_substring(pc.field("Remarks"), remark)

    # Null value also fails the conditions so it would be dropped. coalesce turns Nulls to False
    has_bad = pc.coalesce(has_bad, False)
    return table.filter(~has_bad)
    
def drop_cheap(table: pa.Table) -> pa.Table:
    """Drop rows with TradePrice below MIN_TRADE_PRICE."""
    condition = pc.field("TradePrice") >= MIN_TRADE_PRICE
    return table.filter(condition)

def drop_big_area(table: pa.Table) -> pa.Table:
    """Drop rows where Area or TotalFloorArea is AREA_CAP."""
    condition = (pc.field("Area") == AREA_CAP) | (pc.field("TotalFloorArea") == AREA_CAP)

    condition = pc.coalesce(condition, False)
    return table.filter(~condition)

# ----- Derived columns -----
def split_areas(table: pa.Table) -> pa.Table:
    """Add floor_area and land_area.

    condo: floor_area = Area,           land_area = null
    house: floor_area = TotalFloorArea, land_area = Area
    """

    is_condo = pc.equal(table["Type"], "中古マンション等")
  
    floor_area = pc.if_else(is_condo, table["Area"], table["TotalFloorArea"])
    land_area  = pc.if_else(is_condo, None, table["Area"])

    table = table.append_column("floor_area", floor_area)
    table = table.append_column("land_area", land_area)

    return table

def drop_missing_floor_area(table: pa.Table) -> pa.Table:
    """Drop rows with null floor_area (houses with no building info).
    Must run AFTER split_areas, or every condo gets dropped."""

    condition = pc.field("floor_area").is_valid()
    return table.filter(condition)


def parse_floor_plan(table: pa.Table) -> pa.Table:
    """FloorPlan "3LDK" -> n_rooms=3, has_L/has_D/has_K/has_S booleans.

    スタジオ      -> treat as 1R
    オープンフロア -> is_open_floor=True, n_rooms null
    メゾネット     -> n_rooms null
    Null FloorPlan (most houses) -> everything stays null, no coalescing.
    """
    plan = pc.if_else(pc.equal(table["FloorPlan"], "スタジオ"), "1R", table["FloorPlan"])
    odd_plan = pc.is_in(plan, pa.array(["オープンフロア", "メゾネット"]))

    cols = {}
    for letter in "LDKS":
        cols[f"has_{letter}"] = pc.if_else(odd_plan, None, pc.match_substring(plan, letter))

    cols["n_rooms"] = pc.extract_regex(plan,  r"^(?P<n>\d+)")
    cols["n_rooms"] = pc.cast(pc.struct_field(cols["n_rooms"], "n"), pa.int8())
    cols["is_open_floor"] = pc.equal(plan, "オープンフロア")

    for col_name, arr in cols.items():
        table = table.append_column(col_name, arr)

    return table


def add_age(table: pa.Table) -> pa.Table:
    """age = year - BuildingYear.

    age == -1 -> clip to 0 (sold before completion)
    age < -1 or BuildingYear null -> drop row
    Also add new_build = age <= 0. Keep the existing prewar flag.
    """
    # Append age column
    age = pc.subtract(table["year"], table["BuildingYear"])
    table = table.append_column("age", age)

    # Filter out rows where age is null or less than -1
    table = table.filter(pc.field("age") >= -1)

    # Replaces age column with age column that is clipped -1 -> 0
    clipped = pc.max_element_wise(table["age"], 0)
    idx = table.schema.get_field_index("age")
    table = table.set_column(idx, "age", clipped)

    # Add a new column to indicate new build
    new_build = pc.equal(table["age"], 0)
    table = table.append_column("new_build", new_build)

    return table

def add_log_price(table: pa.Table) -> pa.Table:
    """Add log_price, the target."""
    log_price = pc.ln(table["TradePrice"])
    table = table.append_column("log_price", log_price)

    return table   


# ----- Pipeline -----
def build_gold(table: pa.Table) -> pa.Table:
    """Silver -> gold steps, in the order they must run"""
    table = filter_types(table)
    table = drop_bad_remarks(table)
    table = drop_cheap(table)
    table = drop_big_area(table)
    table = split_areas(table)
    table = drop_missing_floor_area(table)
    table = parse_floor_plan(table)
    table = add_age(table)
    table = add_log_price(table)
    return table.select(GOLD_COLUMNS)


def silver_to_gold(silver_folder: Path, output_folder: Path) -> None:
    """Builds gold from silver batch by batch, keeping hive partitions"""
    silver = ds.dataset(silver_folder, format="parquet", partitioning=PARTITIONING)

    gold_schema = build_gold(silver.head(0)).schema
    batches = (build_gold(batch) for batch in silver.to_batches())

    ds.write_dataset(
        batches,
        output_folder,
        schema=gold_schema,
        format="parquet",
        partitioning=PARTITIONING,
        existing_data_behavior="delete_matching",
    )


if __name__ == "__main__":
    print("Silver -> Gold Conversion...")
    silver_to_gold(SILVER_FOLDER, OUTPUT_FOLDER)
    print("Done.")