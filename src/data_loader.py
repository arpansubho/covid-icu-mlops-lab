import pandas as pd
from typing import Dict


def load_data(path: str, dtypes: Dict[str, str], sample_frac: float = 1.0) -> pd.DataFrame:
    """
    Memory-safe CSV loader with optional sampling.
    """
    df = pd.read_csv(path, dtype=dtypes)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    return df
