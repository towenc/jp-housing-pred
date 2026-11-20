from unicodedata import numeric

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


# Baselines
class GlobalMean:
    """Simply predicts the global mean of the target variable"""
    def __init__(self):
        self.mean_ = None
    def fit(self, X, y):
        self.mean_ = pc.mean(y).as_py()
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)

class GroupMean:
    """Predicts the mean of the target variable for each group, falling back to the global mean if the group is unseen."""
    def __init__(self, group):
        self.group = group
        self.group_mean_ = None
        self.global_mean_ = None
    def fit(self, X, y):
        t = pa.table({self.group: X[self.group], "y": y})
        self.group_mean_ = t.group_by(self.group).aggregate([("y", "mean")])
        self.global_mean_ = pc.mean(y).as_py()
        return self
    def predict(self, X):
        keys = self.group_mean_[self.group].to_pylist()
        means = self.group_mean_["y_mean"].to_pylist()
        group_means = dict(zip(keys, means))

        groups = X[self.group].to_pylist()
        preds = [group_means.get(g, self.global_mean_) for g in groups]
        return np.array(preds)

def ridge(alpha=1.0, numeric=(), categorical=()):
    """Ridge regression with preprocessing
    
    Args:
        alpha (float): Regularization strength for Ridge regression.
        numeric (tuple): Names of numeric columns to be standardized.
        categorical (tuple): Names of categorical columns to be one-hot encoded.
    """

    numeric_steps = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
    )
    categorical_steps = OneHotEncoder(handle_unknown='ignore')

    pre = ColumnTransformer([
        ("num", numeric_steps, list(numeric)),
        ("cat", categorical_steps, list(categorical))
    ])

    return Pipeline([
        ("pre", pre),
        ("ridge", Ridge(alpha=alpha))
    ])