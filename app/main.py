from fastapi import FastAPI, HTTPException
import joblib
import os
from src.schema import OTTEvent

app = FastAPI(title="OTT Drop-off Prediction API")

model = None  # lazy-loaded model


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
    # perform preprocessing & prediction
    return {"prediction": 1}
