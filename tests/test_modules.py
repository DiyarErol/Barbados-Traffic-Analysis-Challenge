import pandas as pd
from pathlib import Path
from src.modules.data_loader import DataLoader
from src.modules.preprocessing import Preprocessing
from src.modules.feature_engineering import FeatureEngineering


def test_data_loader():
    dl = DataLoader(train_csv=Path('sample_data/Train.csv'))
    df = dl.load_train()
    assert not df.empty


def test_preprocessing():
    df = pd.DataFrame({'a': [1, None, 2]})
    pp = Preprocessing()
    out = pp.clean(df)
    assert out.isna().sum().sum() == 0


def test_feature_engineering():
    df = pd.DataFrame({'video_time': ['2024-01-01 08:00:00']})
    fe = FeatureEngineering()
    out = fe.add_basic_time_features(df)
    assert {'hour', 'day_of_week'}.issubset(out.columns)
