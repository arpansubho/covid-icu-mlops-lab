import pandas as pd
import numpy as np
from src.clean_data import replace_unknowns


def test_unknown_codes_to_nan():
    df = pd.DataFrame({
        "A": [1, 97, 2, 99],
        "B": [98, 3, 4, 5],
    })
    cleaned = replace_unknowns(df, [97, 98, 99])
    assert np.isnan(cleaned.loc[1, "A"])
    assert np.isnan(cleaned.loc[0, "B"])
