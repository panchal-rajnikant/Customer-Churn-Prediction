from pydantic import BaseModel, Field
from typing import Literal

class Customer(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Tenure: int = Field(..., ge=0)
    Monthly_Charges: float = Field(..., gt=0)
    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]