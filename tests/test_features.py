import math

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from jp_housing_pred import features, transform


# ------------ filter_types() ------------
def test_filter_types():
    # Arrange
    table = pa.table({
        "Type": [
            "宅地(土地と建物)",
            "宅地(土地)",
            "中古マンション等",
            "中古マンション等",
            "農地",
            "林地"
        ],
        "year": [2019, 2023, 2019, 2020, 2021, 2025]
        })

    expected = pa.table({
        "Type": [
            "宅地(土地と建物)",
            "中古マンション等",
            "中古マンション等",
        ],
        "year": [2019, 2019, 2020]
        })

    # Act
    result = features.filter_types(table)

    # Assert
    assert result.schema ==  table.schema
    assert result.to_pydict() == expected.to_pydict()


# ------------ drop_bad_remarks() ------------
def test_drop_bad_remarks():
    # Arrange
    table = pa.table ({
        "Remarks": [
            None,
            "私道を含む取引",
            "調停・競売等",
            "私道を含む取引、関係者間取引",
            "その他事情有り、私道を含む取引",
            "古屋付き・取壊し前提"
        ],
        "year": [2019, 2019, 2020, 2021, 2022, 2023]
    })

    expected = pa.table({
        "Remarks": [
            None,
            "私道を含む取引",
        ],
        "year": [2019, 2019]
    })

    # Act
    result = features.drop_bad_remarks(table)

    # Assert
    assert result.schema == table.schema
    assert result.to_pydict() == expected.to_pydict()

# ------------ drop_cheap() ------------
def test_drop_cheap():
    # Arrange
    table = pa.table({
        "TradePrice": [1000000, 999999, 1100000, None],
        "year": [2009, 2010, 2011, 2012],
    })

    expected = pa.table({
        "TradePrice": [1000000, 1100000],
        "year": [2009, 2011],
    })

    # Act
    result = features.drop_cheap(table)

    # Assert
    assert result.schema == table.schema
    assert result.to_pydict() == expected.to_pydict()

# ------------ drop_big_area() ------------
def test_drop_big_area():
    # Arrange
    table = pa.table({
        "Type": ["中古マンション等", 
            "中古マンション等", 
            "宅地(土地と建物)", 
            "宅地(土地と建物)", 
            "宅地(土地と建物)",
        ],
        "Area": [9999, 60, 1000, 10000, 1002],
        "TotalFloorArea": [None, None, 103, 100, 9999],
    })

    expected = pa.table({
        "Type": ["中古マンション等", "宅地(土地と建物)", "宅地(土地と建物)"],
        "Area": [60, 1000, 10000],
        "TotalFloorArea": [None, 103, 100],
    })

    # Act
    result = features.drop_big_area(table)

    # Assert
    assert result.schema == table.schema
    assert result.to_pydict() == expected.to_pydict()

# ------------ drop_split_areas() ------------
def test_split_areas():
    # Arrange
    table = pa.table({
        "Type": ["中古マンション等", "宅地(土地と建物)"],
        "Area": [64,450],
        "TotalFloorArea": [None, 120],
    })

    # Act
    result = features.split_areas(table)

    # Assert
    assert result.column("floor_area").to_pylist() == [64, 120]
    assert result.column("land_area").to_pylist() == [None, 450]

# ------------ drop_big_area() ------------  
def test_drop_missing_floor_area():
    # Arrange
    table = pa.table({
        "Type": ["宅地(土地と建物)", "宅地(土地と建物)"],
        "floor_area": [None, 200],
        "land_area": [63, 500],
    })

    expected = pa.table({
        "Type": ["宅地(土地と建物)"],
        "floor_area": [200],
        "land_area": [500],
    })

    # Act
    result = features.drop_missing_floor_area(table)

    # Assert
    assert result.schema == table.schema
    assert result.to_pydict() == expected.to_pydict()

# ------------ parse_floor_plan() ------------
def test_parse_floor_plan():
    # Arrange
    floorPlans = ["3LDK", "1K", "1R", "1LDK+S",  
            "スタジオ", "オープンフロア", "メゾネット",
            None,
    ]
    years = ["2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016"]
    

    table = pa.table({
        "FloorPlan": floorPlans,
        "year": years
    })
    
    expected = pa.table({
        "FloorPlan": floorPlans,
        "year": years,
        "n_rooms": [3, 1, 1, 1, 1, None, None, None],
        "has_L": [True, False, False, True, False, None, None, None],
        "has_D": [True, False, False, True, False, None, None, None],
        "has_K": [True, True, False, True, False, None, None, None],
        "has_S": [False, False, False, True, False, None, None, None],
        "is_open_floor": [False, False, False, False, False, True, False, None]
   })

    # Act
    result = features.parse_floor_plan(table)

    # Assert
    assert result.to_pydict() == expected.to_pydict()

# ------------ add_age() ------------
def test_add_age():
    # Arrange
    table = pa.table({
        "year": [2020, 2020, 2020, 2020, 2020, 2020],
        "BuildingYear": [2000, 2020, 2021, 2022, None, 1945],
        "prewar": [False, False, False, False, False, True],
    })

    expected = pa.table({
        "year": [2020, 2020, 2020, 2020],
        "BuildingYear": [2000, 2020, 2021, 1945],
        "prewar": [False, False, False, True],
        "age": [20, 0, 0, 75],
        "new_build": [False, True, True, False],
    })

    # Act
    result = features.add_age(table)

    # Assert
    assert result.schema == expected.schema
    assert result.to_pydict() == expected.to_pydict()

# ------------ add_log_price() ------------
def test_add_log_price():
    # Arrange
    table = pa.table({
        "TradePrice":  [10000000, 25000000, None],
    })

    expected_log_price = [math.log(10000000), math.log(25000000), None]

    # Act
    result = features.add_log_price(table)

    # Assert
    assert result.column("TradePrice").to_pylist() == [10000000, 25000000, None]
    assert result.column("log_price").to_pylist() == pytest.approx(expected_log_price)

# ------------ build_gold() ------------
SILVER_SCHEMA = pa.schema([
    ("PriceCategory", pa.string()),
    ("Type", pa.string()),
    ("Region", pa.string()),
    ("MunicipalityCode", pa.string()),
    ("Prefecture", pa.string()),
    ("Municipality", pa.string()),
    ("DistrictName", pa.string()),
    ("TradePrice", pa.int64()),
    ("PricePerUnit", pa.float64()),
    ("FloorPlan", pa.string()),
    ("Area", pa.float64()),
    ("UnitPrice", pa.float64()),
    ("LandShape", pa.string()),
    ("Frontage", pa.float64()),
    ("TotalFloorArea", pa.float64()),
    ("BuildingYear", pa.int16()),
    ("Structure", pa.string()),
    ("Use", pa.string()),
    ("Purpose", pa.string()),
    ("Direction", pa.string()),
    ("Classification", pa.string()),
    ("Breadth", pa.float64()),
    ("CityPlanning", pa.string()),
    ("CoverageRatio", pa.float64()),
    ("FloorAreaRatio", pa.float64()),
    ("Period", pa.string()),
    ("Renovation", pa.string()),
    ("Remarks", pa.string()),
    ("DistrictCode", pa.string()),
    ("prewar", pa.bool_()),
    ("pref", pa.string()),
    ("year", pa.int16()),
    ("quarter", pa.int8()),
])

def make_silver_row(**overrides):
    row = {
        "Type": "中古マンション等",
        "TradePrice": 30000000,
        "FloorPlan": "3LDK",
        "Area": 70.0,
        "TotalFloorArea": None,
        "BuildingYear": 2000,
        "Remarks": None,
        "prewar": False,
        "pref": "13",
        "year": 2020,
        "quarter": 1,
    }
    row.update(overrides)
    return row

def test_build_gold():
    # Arrange
    rows = [
        make_silver_row(),                                                    # condo: kept
        make_silver_row(Type="宅地(土地と建物)", Area=200.0, TotalFloorArea=100.0),  # house: kept
        make_silver_row(Type="宅地(土地のみ)"),                                 # wrong type
        make_silver_row(Remarks="関係者間取引"),                                 # bad remark
        make_silver_row(TradePrice=999999),                                   # too cheap
        make_silver_row(Area=9999.0),                                         # area capped
        make_silver_row(Type="宅地(土地と建物)", Area=200.0),                    # house, no floor area
        make_silver_row(BuildingYear=None),                                   # no building year
    ]
    table = pa.Table.from_pylist(rows, schema=SILVER_SCHEMA)

    # Act
    result = features.build_gold(table)

    # Assert
    assert result.column_names == features.GOLD_COLUMNS
    assert result.column("Type").to_pylist() == ["中古マンション等", "宅地(土地と建物)"]
    assert result.column("floor_area").to_pylist() == [70.0, 100.0]
    assert result.column("land_area").to_pylist() == [None, 200.0]
    assert result.column("age").to_pylist() == [20, 20]
    assert result.column("log_price").to_pylist() == pytest.approx([math.log(30000000)] * 2)

# ------------ silver_to_gold() ------------
def read_gold(gold):
    return ds.dataset(gold, format="parquet", partitioning=transform.PARTITIONING).to_table()

def test_silver_to_gold(tmp_path):
    # Arrange
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"

    rows = [
        make_silver_row(),
        make_silver_row(pref="20"),
        make_silver_row(pref="22", TradePrice=999999)
    ]
    table = pa.Table.from_pylist(rows, schema=SILVER_SCHEMA)

    ds.write_dataset(table, silver, format="parquet", partitioning=transform.PARTITIONING)

    # Act
    features.silver_to_gold(silver, gold)
    result = read_gold(gold).sort_by("pref")

    # Assert
    assert result.num_rows == 2
    assert result.column("pref").to_pylist() == ["13", "20"]
    assert sorted(result.column_names) == sorted(features.GOLD_COLUMNS)