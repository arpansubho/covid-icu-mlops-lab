from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_predict_schema():
    sample = {
        "USMER": 1,
        "MEDICAL_UNIT": 1,
        "SEX": 1,
        "PATIENT_TYPE": 1,
        "INTUBED": 2,
        "PNEUMONIA": 1,
        "AGE": 50,
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
    }

    response = client.post("/predict", json=sample)
    assert response.status_code == 200
    body = response.json()
    assert "mortality_probability" in body
    assert "mortality_prediction" in body
