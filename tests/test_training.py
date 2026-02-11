import os
import pytest

def test_model_file_exists():
    """
    Verify that training creates the model file
    """
    model_path = "models/ott_dropoff_model.pkl"
    assert os.path.exists(model_path), f"Model file not found at {model_path}"