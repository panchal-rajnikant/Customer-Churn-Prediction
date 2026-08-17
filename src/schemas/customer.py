from __future__ import annotations
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ContractEnum(str, Enum):
    Month_to_month = "Month-to-month"
    One_year = "One year"
    Two_year = "Two year"


class PredictionRequest(BaseModel):
    Age: int = Field(..., ge=18, le=100, description="Customer age (18-100)")
    Tenure: int = Field(..., ge=0, le=100, description="Tenure in months")
    Monthly_Charges: float = Field(..., gt=0, description="Monthly charge as a positive number")
    Contract: ContractEnum


class PredictionResponse(BaseModel):
    prediction: int = Field(..., ge=0, le=1, description="Predicted label: 0 (no churn) or 1 (churn)")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional confidence between 0 and 1")
