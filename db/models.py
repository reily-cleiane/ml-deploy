from pydantic import BaseModel

class TextRequest(BaseModel):
    text: str

class PredictionResult(BaseModel):
    text: str
    prediction: str
    certainty: float
