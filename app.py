from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import uvicorn
# from models import Customer
from routes import router

app = FastAPI(title="Customer Churn Prediction API")

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8082, reload=True)


