from fastapi import APIRouter, Depends, HTTPException
from models.customer import Customer
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Imports
from src.predictions.prediction import predict as predict_model

router = APIRouter()

@router.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API"
    }

@router.post("/predict")
def predict(customer: Customer):
    logger.info("API Request for Prediction received")
    result = predict_model(customer)
    return result

@router.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
