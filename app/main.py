from fastapi import FastAPI
import joblib
import pandas as pd
from src.schema import OTTEvent

app = FastAPI(title="OTT Drop-off Prediction API")

model = joblib.load("models/ott_dropoff_model1.pkl")

@app.post("/predict")
def predict(event: OTTEvent):
    data = event.dict()
    df = pd.DataFrame([data])

    pred = model.predict(df)
    prob = model.predict_proba(df)

    return {
        "drop_off": int(pred[0]),
        "probability": float(prob[0][1])
    }
