import pytest
from pathlib import Path
from src.modules.data_loader import DataLoader
from src.features.video_features import VideoFeatureExtractor
from src.models.predictor import ModelPredictor


def test_data_loader_missing_file():
    dl = DataLoader(train_csv=Path('does_not_exist.csv'))
    with pytest.raises(FileNotFoundError):
        dl.load_train()


def test_video_extractor_missing_file():
    vfe = VideoFeatureExtractor()
    with pytest.raises(FileNotFoundError):
        vfe.extract(Path('missing_video.mp4'))


def test_model_predictor_missing_model():
    mp = ModelPredictor()
    with pytest.raises(FileNotFoundError):
        mp.load_model(Path('nonexistent_model'))
