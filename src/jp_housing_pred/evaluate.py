import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def MdAPE(y_true, y_pred):
    """Median absolute percentage error in yen"""
    true = np.exp(y_true)
    pred = np.exp(y_pred)

    return np.median(np.abs(true - pred) / true)

def PPE(y_true, y_pred, p = 0.1):
    """Percentage of Predictions within Error
    Share of predictions within the specified error margins
    """
    true = np.exp(y_true)
    pred = np.exp(y_pred)

    return np.mean(np.abs(true - pred) / true <= p)

TARGET = "log_price"

METRICS = {
    "rmse": root_mean_squared_error,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "mdape": MdAPE,
    "ppe": PPE
}

def evaluate(model, train, val):
    """Fit on train, predict on val and return evaluation metrics
    """
    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_val = val.drop(columns=[TARGET])
    y_val = val[TARGET]

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    return {name: fn(y_val, preds) for name, fn in METRICS.items()}