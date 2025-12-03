"""
Path Configuration
==================
Central management of all file paths.
"""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class PathConfig:
    """Configuration for all project paths."""
    
    # Root directory
    root_dir: Path = Path(__file__).parent.parent.parent
    
    # Data directories
    data_dir: Path = None
    video_dir: Path = None
    train_csv: Path = None
    test_csv: Path = None
    sample_submission: Path = None
    
    # Output directories
    output_dir: Path = None
    models_dir: Path = None
    features_dir: Path = None
    predictions_dir: Path = None
    logs_dir: Path = None
    plots_dir: Path = None
    
    # Reports
    reports_dir: Path = None
    
    def __post_init__(self):
        """Initialize all paths relative to root."""
        if self.data_dir is None:
            self.data_dir = self.root_dir / "data"
        
        if self.video_dir is None:
            self.video_dir = self.root_dir / "videos"
        
        if self.train_csv is None:
            self.train_csv = self.root_dir / "Train.csv"
        
        if self.test_csv is None:
            self.test_csv = self.root_dir / "TestInputSegments.csv"
        
        if self.sample_submission is None:
            self.sample_submission = self.root_dir / "SampleSubmission.csv"
        
        if self.output_dir is None:
            self.output_dir = self.root_dir / "output"
        
        if self.models_dir is None:
            self.models_dir = self.output_dir / "models"
        
        if self.features_dir is None:
            self.features_dir = self.output_dir / "features"
        
        if self.predictions_dir is None:
            self.predictions_dir = self.output_dir / "predictions"
        
        if self.logs_dir is None:
            self.logs_dir = self.output_dir / "logs"
        
        if self.plots_dir is None:
            self.plots_dir = self.output_dir / "plots"
        
        if self.reports_dir is None:
            self.reports_dir = self.root_dir / "reports"
    
    def create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.data_dir,
            self.video_dir,
            self.output_dir,
            self.models_dir,
            self.features_dir,
            self.predictions_dir,
            self.logs_dir,
            self.plots_dir,
            self.reports_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for a specific model."""
        return self.models_dir / f"{model_name}.pkl"
    
    def get_feature_path(self, feature_type: str) -> Path:
        """Get path for specific features."""
        return self.features_dir / f"{feature_type}_features.csv"
    
    def get_prediction_path(self, submission_name: str) -> Path:
        """Get path for predictions."""
        return self.predictions_dir / f"{submission_name}.csv"
