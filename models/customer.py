from pydantic import BaseModel

class Customer(BaseModel):
    Age: int
    Tenure: int
    Monthly_Charges: float
    Contract: str