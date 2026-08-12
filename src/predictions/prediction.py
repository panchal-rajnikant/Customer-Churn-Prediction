import joblib
import pandas as pd
from models.customer import Customer

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
    return prediction
# def predict_proba():
