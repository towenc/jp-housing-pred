import numpy as np
import pyarrow as pa
import pytest

from jp_housing_pred.models import GlobalMean, GroupMean, ridge


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

# ------------ ridge ------------
@pytest.fixture
def ridge_data():
    X = pa.table({
        "area": [65.0, 80.0, None, 120.0, 45.0],
        "pref": ["13", "13", "27", "27", "15"]
    })
    y = pa.array([10.0, 12.0, 14.0, 16.0, 8.0])
    return X, y

def test_ridge_shape(ridge_data):
    # Arrange
    X, y = ridge_data

    # Act
    model = ridge(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X)

    # Assert
    assert preds.shape == (X.num_rows,)

def test_ridge_unseen_cat(ridge_data):
    # Arrange
    X, y = ridge_data
    X_unseen = pa.table({
        "area": [70.0],
        "pref": ["99"]
    })

    # Act
    model = ridge(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X_unseen)

    # Assert
    assert np.all(np.isfinite(preds))

def test_ridge_nulls(ridge_data):
    # Arrange
    X, y = ridge_data
    X_null = pa.table({
        "area": [70, None],
        "pref": ["13", "13"]
    })

    # Act
    model = ridge(numeric=("area",), categorical=("pref",)).fit(X, y)
    preds = model.predict(X_null)

    # Assert
    assert np.all(np.isfinite(preds))