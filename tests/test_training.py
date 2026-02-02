import os
import subprocess

def test_training_creates_model():
    subprocess.run(["python", "src/train.py"], check=True)
    assert os.path.exists("models/baseline_logreg.joblib")
