from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.model import predict_uplift

class InputData(BaseModel):
    f0: float; f1: float; f2: float; f3: float
    f4: float; f5: float; f6: float; f7: float
    f8: float; f9: float; f10: float; f11: float

app = FastAPI()

@app.post("/uplift")
def uplift(input: InputData):
    df = pd.DataFrame([input.dict()])
    uplift_score = predict_uplift(df)[0]
    return {"uplift": uplift_score}
