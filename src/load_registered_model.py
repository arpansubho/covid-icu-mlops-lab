import mlflow
import pandas as pd

MODEL_NAME = "covid_mortality_model"

def main():
    mlflow.set_tracking_uri("file:./mlruns")

    model_uri = f"models:/{MODEL_NAME}/Staging"
    model = mlflow.pyfunc.load_model(model_uri)

    print("Loaded model from registry:", model_uri)

    # Dummy example input (you'll replace with real schema later)
    sample = pd.DataFrame([{
        "USMER": 1,
        "MEDICAL_UNIT": 1,
        "SEX": 1,
        "PATIENT_TYPE": 1,
        "INTUBED": 2,
        "PNEUMONIA": 1,
        "AGE": 45,
        "PREGNANT": 97,
        "DIABETES": 2,
        "COPD": 2,
        "ASTHMA": 2,
        "INMSUPR": 2,
        "HIPERTENSION": 1,
        "OTHER_DISEASE": 2,
        "CARDIOVASCULAR": 2,
        "OBESITY": 2,
        "RENAL_CHRONIC": 2,
        "TOBACCO": 2,
        "CLASIFFICATION_FINAL": 3,
        "ICU": 2,
    }])

    preds = model.predict(sample)
    print("Prediction:", preds)

if __name__ == "__main__":
    main()
