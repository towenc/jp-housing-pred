from itertools import combinations

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

from jp_housing_pred.split import SPLITS, load_split, load_years
from jp_housing_pred.transform import PARTITIONING


# ----------load_years() ------------
@pytest.mark.parametrize("start, end, expected_years", [
    (2023, 2025, [2023, 2024, 2025]),
    (2025, None, [2025, 2026]),
    (None, 2023, [2022, 2023]),
    (None, None, [2022, 2023, 2024, 2025, 2026]),
])
def test_load_years(gold_folder, start, end, expected_years):
    # Act
    result = load_years(gold_folder, start=start, end=end)

    # Assert
    assert sorted(result["year"].to_pylist()) == expected_years

# ----------load_split() ------------
@pytest.fixture
def gold_folder(tmp_path):
    table = pa.table({
            "log_price": [10.0, 12.0, 13.0, 14.0, 15.0],
            "pref": ["13", "13", "13", "13", "13"],
            "year": [2022,2023, 2024, 2025, 2026],
            "quarter": [1, 1, 1, 1, 1],
        })
    ds.write_dataset(table, tmp_path, format="parquet", partitioning=PARTITIONING)
    return tmp_path

@pytest.mark.parametrize("split, expected_years", [
    ("train", [2022,2023]),
    ("validation", [2024]),
    ("test", [2025, 2026])
])
def test_load_split(gold_folder, split,expected_years):
    # Act
    result = load_split(gold_folder, split)

    # Assert
    assert sorted(result["year"].to_pylist()) == expected_years

def test_splits_no_overlap(gold_folder):
    # Act
    splits = {}
    for name in SPLITS:
        splits[name] = load_split(gold_folder, name)

    # Assert
    # Check all pairs of splits for overlapping years
    for a, b in combinations(splits, 2):
        years_a = set(splits[a]["year"].to_pylist())
        years_b = set(splits[b]["year"].to_pylist())
        assert not years_a & years_b, f"{a} and {b} share years"

def test_all_years_covered(gold_folder):
    # Act
    splits = {}
    for name in SPLITS:
        splits[name] = load_split(gold_folder, name)
    all_data = load_years(gold_folder)

    # Assert
    # Check that total number of rows in splits matches total in all the data
    total = 0
    for t in splits.values():
        total += t.num_rows
    assert total == all_data.num_rows, "Total rows in splits do not match all data"

def test_load_split_unknown_split(tmp_path):
    # Act & Assert
    with pytest.raises(ValueError, match="Unknown split"):
        load_split(tmp_path, "unknown")