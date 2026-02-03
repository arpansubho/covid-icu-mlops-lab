from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd
from src.monitoring import log_request, log_prediction

MODEL_NAME = "covid_mortality_model"

app = FastAPI(title="COVID Mortality Predictor")


class Patient(BaseModel):
    USMER: int
    MEDICAL_UNIT: int
    SEX: int
    PATIENT_TYPE: int
    INTUBED: int
    PNEUMONIA: int
    AGE: int
    PREGNANT: int
    DIABETES: int
    COPD: int
    ASTHMA: int
    INMSUPR: int
    HIPERTENSION: int
    OTHER_DISEASE: int
    CARDIOVASCULAR: int
    OBESITY: int
    RENAL_CHRONIC: int
    TOBACCO: int
    CLASIFFICATION_FINAL: int
    ICU: int


model = None

def get_model():
    global model
    if model is None:
        mlflow.set_tracking_uri("file:./mlruns")
        model_uri = f"models:/{MODEL_NAME}/Staging"
        model = mlflow.pyfunc.load_model(model_uri)
    return model


@app.post("/predict")
def predict(patient: Patient):
    mdl = get_model()

    payload = patient.dict()
    log_request(payload)

    data = pd.DataFrame([payload])

    # Handle models with or without predict_proba (for tests / dummy model)
    if hasattr(mdl, "predict_proba"):
        prob = mdl.predict_proba(data)[0, 1]
    else:
        prob = mdl.predict(data)[0]

    pred = int(prob >= 0.5)

    log_prediction(pred, float(prob))

    return {
        "mortality_probability": float(prob),
        "mortality_prediction": pred,
    }

