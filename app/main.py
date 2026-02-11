from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os
from src.schema import OTTEvent

app = FastAPI(title="OTT Drop-off Prediction API")

# Lazy-loaded model
model = None

def get_model():
    global model
    if model is None:
        model_path = "models/ott_dropoff_model.pkl"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=500, detail="Model file not found")
        model = joblib.load(model_path)
    return model

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(event: OTTEvent):
    model = get_model()

    data = event.dict()
    df = pd.DataFrame([data])

    pred = model.predict(df)
    prob = model.predict_proba(df)

    return {
        "drop_off": int(pred[0]),
        "probability": float(prob[0][1])
    }