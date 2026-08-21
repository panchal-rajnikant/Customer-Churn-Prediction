import joblib
import pandas as pd
from models.customer import Customer
from fastapi import HTTPException
from src.logger import log_event
import time

metrics = {
    "total_predictions": 0,
    "total_errors": 0,
    "total_latency_ms": 0.0
}

def predict(customer: Customer):
    log_event("Prediction request received")
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
        start_time = time.perf_counter()
        prediction =model.predict(X)[0]

        probabilities = model.predict_proba(X)[0]

        confidence = probabilities[int(prediction)]
        latency = time.perf_counter() - start_time
        log_event(
            "Prediction successful: %s, confidence: %.2f",
            prediction,
            confidence,
            latency * 1000
        )

        metrics["total_predictions"] += 1
        metrics["total_latency_ms"] += latency
        return {
            "prediction": int(prediction),
            "confidence": round(float(confidence), 4),
            "latency_ms": round(latency * 1000, 2)
        }
    except Exception as e:
        log_event("Prediction failed: %s", e)
        metrics["total_errors"] += 1

    raise HTTPException(
        status_code=500,
        detail="Prediction failed"
    )
    
# def predict_proba():
