# Empty for now; add Pydantic models if needed later
from pydantic import BaseModel

class UserResponse(BaseModel):
    user_id: str
    message: str

class PredictionResponse(BaseModel):
    predicted_disease: str
    confidence: float