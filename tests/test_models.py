import numpy as np
import pyarrow as pa
import pytest

from jp_housing_pred.models import GlobalMean, GroupMean, RidgeRegressor, LGBM


@pytest.fixture
def train_data():
    X = pa.table({
        "pref": ["13", "13", "27"],
    })
    y = pa.array([10.0, 12.0, 14.0])
    return X, y

# ------------ GlobalMean ------------
def test_global_mean(train_data):
    # Arrange
    X, y = train_data

    # Act
    model = GlobalMean().fit(X, y)
    preds = model.predict(X)

    # Assert
    assert np.all(preds == np.array([12.0, 12.0, 12.0]))

# ------------ GroupMean ------------
def test_group_mean(train_data):
    # Arrange
    X, y = train_data

    # Act 
    model = GroupMean("pref").fit(X, y)
    preds = model.predict(X)

    # Assert
    assert np.all(preds == np.array([11.0, 11.0, 14.0]))

def test_group_mean_unseen_val(train_data):
    # Arrange
    X, y = train_data
    X_unseen = pa.table({
        "pref": ["99"]
    })

    # Act
    model = GroupMean("pref").fit(X, y)
    preds = model.predict(X_unseen)

    # Assert
    assert np.all(preds == np.array([12.0]))

def test_group_mean_missing_col(train_data):
    # Arrange
    X, y = train_data
    model = GroupMean("random_col")

    # Act / Assert
    with pytest.raises(KeyError):
        model.fit(X, y)

# ------------ RidgeRegressor ------------
@pytest.fixture
def small_data():
    X = pa.table({
        "area": [65.0, 80.0, None, 120.0, 45.0],
        "pref": ["13", "13", "27", "27", "15"]
    })
    y = pa.array([10.0, 12.0, 14.0, 16.0, 8.0])
    return X, y

def test_ridge_shape(small_data):
    # Arrange
    X, y = small_data

    # Act
    model = RidgeRegressor(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X)

    # Assert
    assert preds.shape == (X.num_rows,)

def test_ridge_unseen_cat(small_data):
    # Arrange
    X, y = small_data
    X_unseen = pa.table({
        "area": [70.0],
        "pref": ["99"]
    })

    # Act
    model = RidgeRegressor(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X_unseen)

    # Assert
    assert np.all(np.isfinite(preds))

def test_ridge_nulls(small_data):
    # Arrange
    X, y = small_data
    X_null = pa.table({
        "area": [70, None],
        "pref": ["13", "13"]
    })

    # Act
    model = RidgeRegressor(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X_null)

    # Assert
    assert np.all(np.isfinite(preds))

def test_ridge_numeric_inputs(small_data):
    # Arrange
    X, y = small_data
    X_test = pa.table({
        "area": [50.0, 110.0],
        "pref": ["13", "13"]
    })

    # Act
    model = RidgeRegressor(numeric=("area",), categorical=("pref",)).fit(X,y)
    preds = model.predict(X_test)

    # Assert
    assert preds[1] > preds[0]

def test_ridge_categorical_inputs(small_data):
    # Arrange
    X, y = small_data
    X_test = pa.table({
        "area": [70.0, 70.0],
        "pref": ["27", "13"]
    })

    # Act
    model = RidgeRegressor(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X_test)

    # Assert
    assert preds[0] > preds[1]

# ------------ LGBM ------------
def test_lgbm_shape(small_data):
    # Arrage
    X, y = small_data

    # Act
    model = LGBM(numeric=("area",), categorical=("pref",), verbose=-1).fit(X, y)
    preds = model.predict(X)

    #Assert
    assert preds.shape == (X .num_rows,)

def test_lgbm_unseen_cat(small_data):
    # Arrange
    X, y = small_data
    X_unseen = pa.table({
        "area": [70.0],
        "pref": ["99"]
    })

    # Act
    model = LGBM(numeric=("area",), categorical=("pref",), verbose=-1, min_child_samples=1, min_data_in_bin=1).fit(X, y)
    preds = model.predict(X_unseen)

    # Assert
    assert np.all(np.isfinite(preds))

def test_lgbm_nulls(small_data):
    # Arrange
    X, y = small_data
    X_null = pa.table({
        "area": [70, None],
        "pref": ["13", "13"]
    })

    # Act
    model = LGBM(numeric=("area",), categorical=("pref",), verbose=-1, min_child_samples=1, min_data_in_bin=1).fit(X, y)
    preds = model.predict(X_null)

    # Assert
    assert np.all(np.isfinite(preds))

def test_lgbm_cat_idx():
    # Arrange / Act
    model = LGBM(numeric=("a", "b", "c",), categorical=("d", "e",), verbose=-1)
    # Assert
    assert model.cat_idx == [3, 4]