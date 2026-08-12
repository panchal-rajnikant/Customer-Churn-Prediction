from fastapi import APIRouter, Depends, HTTPException
from models.customer import Customer
import pandas as pd

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

    result = predict_model(customer)
    return {"prediction": int(result)}
