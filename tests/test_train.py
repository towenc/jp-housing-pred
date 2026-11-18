import pytest   
from jp_housing_pred.train import build_model
from jp_housing_pred.models import GlobalMean, GroupMean

# ------------ build_model() ------------
def test_build_model_with_params():
    # Arrange
    entry = {"name": "group_mean", "group": "pref"}

    # Act
    model = build_model(entry)

    # Assert
    assert isinstance(model, GroupMean)
    assert model.group == "pref"

def test_build_model_without_params():
    # Arrange
    entry = {"name": "global_mean"}

    # Act
    model = build_model(entry)

    # Assert
    assert isinstance(model, GlobalMean)

def test_build_model_invalid_name():
    # Arrange
    entry = {"name": "invalid_model"}

    # Act & Assert
    with pytest.raises(KeyError):
        build_model(entry)