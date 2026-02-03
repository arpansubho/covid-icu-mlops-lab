import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("monitoring")


def log_request(data: dict):
    logger.info(f"Received prediction request: {data}")


def log_prediction(prediction: int, prob: float):
    logger.info(f"Prediction={prediction}, probability={prob:.4f}")


def detect_drift(reference_df: pd.DataFrame, new_df: pd.DataFrame, threshold=0.2):
    """
    Simple drift stub using mean comparison.
    In real systems: use KS test, PSI, Evidently, etc.
    """
    drift_report = {}
    for col in reference_df.columns:
        ref_mean = reference_df[col].mean()
        new_mean = new_df[col].mean()
        diff = abs(ref_mean - new_mean)

        drift_report[col] = diff > threshold

    return drift_report
