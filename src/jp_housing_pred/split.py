import pyarrow.compute as pc
import pyarrow.dataset as ds
from jp_housing_pred.transform import PARTITIONING

dataset = ds.dataset("data/gold", format="parquet", partitioning=PARTITIONING)

condition = pc.field("year") == 2024
table = dataset.to_table(filter=condition)
print(table.num_rows)