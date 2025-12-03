import pandas as pd
from typing import Any

class Inference:
    def predict(self, model: Any, df: pd.DataFrame, feature_cols: list) -> pd.Series:
        return pd.Series(model.predict(df[feature_cols]))
