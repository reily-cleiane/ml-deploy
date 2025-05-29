from pydantic import BaseModel
from typing import Dict, Optional

class TextRequest(BaseModel):
    text: str

class PredictionResult(BaseModel):
    text: str
    prediction: str
    certainty: float
    all_probabilities: Optional[Dict[str, float]] = None
