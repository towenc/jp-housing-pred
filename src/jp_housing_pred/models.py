import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

class GlobalMean:
    def __init__(self):
        self.mean = None
    def fit(self, X, y):
        self.mean = pc.mean(y).as_py()
        return self
    def predict(self, X):
        return np.full(len(X), self.mean)

class GroupMean:
    def __init__(self, group):
        self.group = group
        self.group_mean = None
    def fit(self, X, y):
        t = pa.table({self.group: X[self.group], "y": y})
        self.group_mean = t.group_by(self.group).aggregate([("y", "mean")])
        self.global_mean = pc.mean(y).as_py()
        return self
    def predict(self, X):
        


