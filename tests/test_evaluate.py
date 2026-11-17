import numpy as np
import pyarrow as pa
import pytest

from jp_housing_pred.evaluate import evaluate, mdape, ppe


# ------------ MdAPE ------------
def test_mdape():
    # Arrange
    y_true = np.log([60000000, 50000000, 25000000])
    y_pred = np.log([65000000, 40000000, 26000000])

    # Act
    result = mdape(y_true, y_pred)

    # Assert
    assert result == pytest.approx(1/12)

def test_mdape_perfect():
    # Arrange
    y_true = np.log([60000000, 50000000, 25000000])
    y_pred = y_true

    # Act
    result = mdape(y_true, y_pred)

    # Assert
    assert result == pytest.approx(0)

# ------------ PPE ------------
def test_ppe():
    # Arrange
    y_true = np.log([60000000, 50000000, 25000000])
    y_pred = np.log([65000000, 40000000, 26000000])

    # Act
    result = ppe(y_true, y_pred)

    # Assert
    assert result == pytest.approx(2/3)

def test_ppe_perfect():
    # Arrange
    y_true = np.log([60000000, 50000000, 25000000])
    y_pred = y_true

    # Act
    result = ppe(y_true, y_pred)

    # Assert
    assert result == pytest.approx(1)

# ------------ evaluate ------------
class StubModel:
    def fit(self, X, y):
        self.train_rows  = len(X)  # store number of rows for training
        self.fit_columns = X.column_names # store columns of input
        
        return self
    def predict(self, X):
        self.predict_rows = len(X)  # store number of rows for prediction
        return np.zeros(len(X))     # always returns 0

def test_evaluate():
    # Arrange
    train = pa.table({
        "area": [50, 70, 90],
        "log_price": [np.log(60000000), np.log(50000000), np.log(25000000)]
    })

    val = pa.table({
        "area": [80, 100],
        "log_price": [np.log(60000000), np.log(30000000)]
    })
    model = StubModel()
  
    # Act
    result = evaluate(model, train, val)

    # Assert
    # check all metrics are present
    assert set(result.keys()) == {"rmse", "mae", "r2", "mdape", "ppe"}
    # check that the model did not use the target column during fitting
    assert "log_price" not in model.fit_columns, "evaluate should not pass the target column to the modelcl"
    # check that the model evaluated on val
    assert model.predict_rows == len(val)
    # check that the model saw the correct number of training rows
    assert model.train_rows == len(train)
    # check that model scored on y_val
    assert result["mae"] == pytest.approx(np.mean(val["log_price"].to_numpy())), "evaluate does not score on y_val"
    