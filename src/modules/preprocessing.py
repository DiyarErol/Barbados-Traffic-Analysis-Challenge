import pandas as pd

class Preprocessing:
    """Basic preprocessing utilities."""
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates()
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
