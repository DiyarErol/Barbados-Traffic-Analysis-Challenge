import pandas as pd

class FeatureEngineering:
    """Minimal feature engineering placeholder."""
    def add_basic_time_features(self, df: pd.DataFrame, time_col: str = 'video_time') -> pd.DataFrame:
        df = df.copy()
        dt = pd.to_datetime(df[time_col])
        df['hour'] = dt.dt.hour
        df['day_of_week'] = dt.dt.dayofweek
        return df
