from fastapi import APIRouter, Depends, HTTPException
from models.customer import Customer
import joblib
import pandas as pd
router = APIRouter()

@router.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API"
    }

@router.post("/predict")
def predict(customer: Customer):

    model = joblib.load("models/best_model.pkl")

    df = pd.DataFrame([{
        "Age": customer.Age,
        "Tenure": customer.Tenure,
        "Monthly Charges": customer.Monthly_Charges,
        "Contract": customer.Contract
    }])

    X = pd.get_dummies(df, columns=["Contract"])

    # make the feature order match training exactly
    feature_names = model.feature_names_in_
    X = X.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(X)[0]

    return {"prediction": int(prediction)}
