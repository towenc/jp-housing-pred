import pyarrow.compute as pc
import pyarrow.dataset as ds

from jp_housing_pred.transform import PARTITIONING

SPLITS = {
    "train": {"start": None, "end": 2023},
    "validation": {"start": 2024, "end": 2024},
    "test": {"start": 2025, "end": None}
}

def load_years(folder, start=None, end=None):
    """Load data for the specified year range given by start and end.

    If start is None, data from all years up to and including end will be loaded.
    If end is None, data from all years from start onwards will be loaded.
    If both are None, all data will be loaded.
    """
    dataset = ds.dataset(folder, format="parquet", partitioning=PARTITIONING)

    if start is not None and end is not None:
        condition = (pc.field("year") >= start) & (pc.field("year") <= end)
    elif start is not None:
        condition = (pc.field("year") >= start)
    elif end is not None:
        condition = (pc.field("year") <= end)
    else:
        condition = None

    return dataset.to_table(filter=condition)

def load_split(folder, split):
    """Load data for the specified split (train, validation, or test)."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")

    split_info = SPLITS[split]
    return load_years(folder, start=split_info["start"], end=split_info["end"])