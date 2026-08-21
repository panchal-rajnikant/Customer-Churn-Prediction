from fastapi import APIRouter, Depends, HTTPException
from models.customer import Customer
from src.logger import log_event
# API Imports
from src.predictions.prediction import predict as predict_model, metrics
import json
from pathlib import Path
router = APIRouter()

@router.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API"
    }

@router.post("/predict")
def predict(customer: Customer):
    log_event("API Request for Prediction received")
    result = predict_model(customer)
    return result

@router.get("/health")
def health_check():
    base_dir = Path(__file__).resolve().parents[0]
    json_path = base_dir / "models" / "model_metadata.json"
    with open(json_path, "r") as f:
        metadata = json.load(f)

    return {
        "status": "healthy",
        "model_version": metadata["model_version"]
    }

@router.get("/metrics")
def get_metrics():

    total = metrics["total_predictions"]

    average_latency = (
        metrics["total_latency_ms"] / total
        if total > 0
        else 0
    )

    return {
        "total_predictions": total,
        "total_errors": metrics["total_errors"],
        "average_latency_ms": round(
            average_latency,
            2
        )
    }
