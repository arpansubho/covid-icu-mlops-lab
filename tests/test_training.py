import pandas as pd
import subprocess
import os

def test_training_creates_model(tmp_path):
    # Create dummy clean dataset
    df = pd.DataFrame({
        "AGE": [30, 70],
        "SEX": [1, 2],
        "ICU": [1, 2],
        "MORTALITY": [0, 1]
    })

    data_dir = tmp_path / "data/processed"
    data_dir.mkdir(parents=True)

    clean_path = data_dir / "clean.csv"
    df.to_csv(clean_path, index=False)

    # Run training with this dummy data
    env = os.environ.copy()
    env["DATA_PATH"] = str(clean_path)

    subprocess.run(
        ["python", "-m", "src.train"],
        check=True,
        env=env
    )
