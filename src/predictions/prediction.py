import joblib
import pandas as pd
from models.customer import Customer
import logging
from fastapi import HTTPException
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(customer: Customer):
    logger.info("Prediction request received")
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
    
    try:
        # prediction = model.predict(X)[0]
        # logger.info("Prediction completed successfully")
        # # return prediction

        prediction =model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        confidence = probabilities[int(prediction)]

        logger.info(
            "Prediction successful: %s, confidence: %.2f",
            prediction,
            confidence
        )
        return {
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4)
        }
    except Exception as e:
        logger.error("Prediction failed: %s", e)

    raise HTTPException(
        status_code=500,
        detail="Prediction failed"
    )
    
# def predict_proba():
