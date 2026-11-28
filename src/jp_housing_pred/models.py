import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


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
    
# Models
# def ridge(alpha=1.0, numeric=(), categorical=()):
#     """Ridge regression with preprocessing
#     For numerical columns, median imputation is done for missing valuues.
#     Standard scaling is applied to numerical columns after imputation.

#     For categorical columns, one-hot encoding is applied after handling unknown categories.
#     Args:
#         alpha (float): Regularization strength for Ridge regression.
#         numeric (tuple): Names of numeric columns to be standardized.
#         categorical (tuple): Names of categorical columns to be one-hot encoded.
#     """

#     numeric_steps = make_pipeline(
#         SimpleImputer(strategy="median", add_indicator=True),
#         StandardScaler(),
#     )
#     categorical_steps = OneHotEncoder(handle_unknown='ignore')

#     pre = ColumnTransformer([
#         ("num", numeric_steps, list(numeric)),
#         ("cat", categorical_steps, list(categorical))
#     ])

#     return Pipeline([
#         ("pre", pre),
#         ("ridge", Ridge(alpha=alpha))
#     ])

class RidgeRegressor:
    """"Ridge regression with preprocessing
    For numerical columns, median imputation is done for missing valuues.
    Standard scaling is applied to numerical columns after imputation.

    For categorical columns, one-hot encoding is applied after handling unknown categories.
    """
    def __init__(self, numeric=(), categorical=(), alpha=1.0):
        numeric_steps = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
        )
        categorical_steps = OneHotEncoder(handle_unknown='ignore')

        pre = ColumnTransformer([
            ("num", numeric_steps, list(numeric)),
            ("cat", categorical_steps, list(categorical)),
        ])
        self.pipe = Pipeline([
            ("pre", pre),
            ("ridge", Ridge(alpha=alpha))
        ])

    def fit(self, X, y):
        self.pipe.fit(X,y)
        return self

    def predict(self, X):
        return self.pipe.predict(X)





class LGBM:
    """LightGBM with preprocessing
    
    For numerical columns, no preprocessing (passthrough)
    For categorical columns, ordinal encoding done to prevent sparse rows from OneHotEncoding
    """
    def __init__(self, numeric=(), categorical=(), **params):
        pre = ColumnTransformer([
            ("num", "passthrough", list(numeric)),
            ("cat", OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), list(categorical))
        ])
        self.pipe = Pipeline([
            ("pre", pre),
            ("lgbm", LGBMRegressor(**params)),
        ])
        self.cat_idx = list(range(len(numeric), len(numeric) + len(categorical)))


    def fit(self, X, y):
        self.pipe.fit(X.to_pandas(), y, lgbm__categorical_feature=self.cat_idx)
        return self

    def predict(self, X):
        return self.pipe.predict(X.to_pandas())