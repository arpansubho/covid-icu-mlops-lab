from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import pandas as pd

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
    data = pd.DataFrame([patient.dict()])
    prob = mdl.predict(data)[0]
    pred = int(prob >= 0.5)
    return {
        "mortality_probability": float(prob),
        "mortality_prediction": pred,
    }
