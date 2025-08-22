from fastapi import FastAPI
from pydantic import BaseModel
from src.model_manager import predict_pipeline

class InputData(BaseModel):
    f0: float; f1: float; f2: float; f3: float
    f4: float; f5: float; f6: float; f7: float
    f8: float; f9: float; f10: float; f11: float
    treatment: int | None = None
    visit: int | None = None

app = FastAPI()

@app.post("/predict/{target}")
def predict(target: str, input: InputData):
    features = input.dict(exclude_none=True)
    pred = predict_pipeline(target, features)
    return {"target": target, "prediction": int(pred)}
