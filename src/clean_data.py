import pandas as pd
import numpy as np
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def replace_unknowns(df: pd.DataFrame, unknown_codes: list) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        df[col] = df[col].replace(unknown_codes, np.nan)
    return df


def create_mortality_label(df: pd.DataFrame, date_col: str, label_col: str) -> pd.DataFrame:
    df = df.copy()
    df[label_col] = (df[date_col] != "9999-99-99").astype(int)
    df = df.drop(columns=[date_col])
    return df


def main():
    config = load_config("configs/config.yaml")

    dtypes = {
        "USMER": "int8",
        "MEDICAL_UNIT": "int8",
        "SEX": "int8",
        "PATIENT_TYPE": "int8",
        "INTUBED": "int8",
        "PNEUMONIA": "int8",
        "AGE": "int16",
        "PREGNANT": "int8",
        "DIABETES": "int8",
        "COPD": "int8",
        "ASTHMA": "int8",
        "INMSUPR": "int8",
        "HIPERTENSION": "int8",
        "OTHER_DISEASE": "int8",
        "CARDIOVASCULAR": "int8",
        "OBESITY": "int8",
        "RENAL_CHRONIC": "int8",
        "TOBACCO": "int8",
        "CLASIFFICATION_FINAL": "int8",
        "ICU": "int8",
    }

    df = pd.read_csv(config["paths"]["raw_data"], dtype=dtypes)

    df = replace_unknowns(df, config["data"]["unknown_codes"])
    df = create_mortality_label(
        df,
        config["target"]["date_column"],
        config["target"]["label_column"],
    )

    df.to_csv(config["paths"]["processed_data"], index=False)
    print("Clean dataset saved to", config["paths"]["processed_data"])


if __name__ == "__main__":
    main()
