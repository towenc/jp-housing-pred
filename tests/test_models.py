import numpy as np
import pyarrow as pa
import pytest

from jp_housing_pred.models import GlobalMean, GroupMean


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