import pytest
from pydantic import ValidationError

# Ensure project root is on sys.path so `src` package is importable when running pytest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.schemas.customer import PredictionRequest, PredictionResponse


def test_valid_request():
    data = {
        "Age": 25,
        "Tenure": 3,
        "Monthly_Charges": 90.0,
        "Contract": "Month-to-month",
    }
    req = PredictionRequest(**data)
    assert req.Age == 25
    assert req.Contract.value == "Month-to-month"


def test_invalid_age():
    data = {
        "Age": -5,
        "Tenure": 3,
        "Monthly_Charges": 90.0,
        "Contract": "Month-to-month",
    }
    with pytest.raises(ValidationError):
        PredictionRequest(**data)


def test_missing_fields():
    data = {
        "Age": 25,
        "Tenure": 3,
        "Contract": "Month-to-month",
    }
    with pytest.raises(ValidationError):
        PredictionRequest(**data)


def test_wrong_data_type():
    data = {
        "Age": 25,
        "Tenure": 3,
        "Monthly_Charges": "yes",
        "Contract": "Month-to-month",
    }
    with pytest.raises(ValidationError):
        PredictionRequest(**data)


def test_business_range():
    data = {
        "Age": 150,
        "Tenure": 3,
        "Monthly_Charges": 90.0,
        "Contract": "Month-to-month",
    }
    with pytest.raises(ValidationError):
        PredictionRequest(**data)


def test_prediction_response():
    res = PredictionResponse(prediction=1, confidence=0.9084)
    assert res.prediction in (0, 1)
    assert 0.0 <= res.confidence <= 1.0
