"""
Inference Pipeline
==================
End-to-end pipeline for production inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List
from tqdm import tqdm

from ..config import PathConfig, FeatureConfig
from ..features import VideoFeatureExtractor, TemporalFeatureExtractor
from ..models import ModelPredictor


class InferencePipeline:
    """End-to-end inference pipeline for production."""
    
    def __init__(self,
                 model_path: Path,
                 feature_config: Optional[FeatureConfig] = None,
                 path_config: Optional[PathConfig] = None):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model
            feature_config: Feature configuration
            path_config: Path configuration
        """
        self.feature_config = feature_config or FeatureConfig()
        self.path_config = path_config or PathConfig()
        
        # Initialize feature extractors
        self.video_extractor = VideoFeatureExtractor(self.feature_config.video)
        self.temporal_extractor = TemporalFeatureExtractor(self.feature_config.temporal)
        
        # Load model
        self.predictor = ModelPredictor(model_path)
        
        print(f"✓ Inference pipeline ready")
    
    def load_test_data(self) -> pd.DataFrame:
        """
        Load test data.
        
        Returns:
            Test DataFrame
        """
        if not self.path_config.test_csv.exists():
            raise FileNotFoundError(f"Test data not found: {self.path_config.test_csv}")
        
        df = pd.read_csv(self.path_config.test_csv)
        print(f"Loaded {len(df)} test samples")
        
        return df
    
    def extract_features(self, df: pd.DataFrame, extract_video: bool = True) -> pd.DataFrame:
        """
        Extract features from test data.
        
        Args:
            df: Test DataFrame
            extract_video: Whether to extract video features
            
        Returns:
            DataFrame with features
        """
        print(f"\nExtracting features...")
        
        features = df.copy()
        
        # Temporal features
        temporal_features = self.temporal_extractor.extract(df)
        features = pd.concat([features, temporal_features], axis=1)
        
        # Video features (if enabled)
        if extract_video and 'video_path' in df.columns:
            video_paths = [self.path_config.video_dir / path for path in df['video_path']]
            video_features = self.video_extractor.extract_batch(video_paths, show_progress=True)
            features = pd.concat([features, video_features.drop(columns=['video_path'])], axis=1)
        
        print(f"✓ Extracted {features.shape[1]} features")
        
        return features
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            features: Feature DataFrame
            
        Returns:
            Predictions array
        """
        # Get feature columns (exclude metadata)
        metadata_cols = ['video_id', 'video_time', 'video_path', 'datetime']
        feature_cols = [col for col in features.columns if col not in metadata_cols]
        
        X = features[feature_cols]
        
        print(f"\nMaking predictions...")
        predictions = self.predictor.predict(X)
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions
    
    def create_submission(self, df: pd.DataFrame, 
                         enter_predictions: np.ndarray,
                         exit_predictions: np.ndarray,
                         output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Create submission file.
        
        Args:
            df: Original test DataFrame
            enter_predictions: Predictions for enter rating
            exit_predictions: Predictions for exit rating
            output_path: Path to save submission (optional)
            
        Returns:
            Submission DataFrame
        """
        submission = []
        
        for i, row in df.iterrows():
            # Enter prediction
            submission.append({
                'video_id': row['video_id'],
                'prediction_label': 'congestion_enter_rating',
                'prediction': enter_predictions[i]
            })
            
            # Exit prediction
            submission.append({
                'video_id': row['video_id'],
                'prediction_label': 'congestion_exit_rating',
                'prediction': exit_predictions[i]
            })
        
        submission_df = pd.DataFrame(submission)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            submission_df.to_csv(output_path, index=False)
            print(f"✓ Submission saved to: {output_path}")
        
        return submission_df
    
    def run(self, extract_video: bool = True, 
            output_name: str = "submission") -> pd.DataFrame:
        """
        Run complete inference pipeline.
        
        Args:
            extract_video: Whether to extract video features
            output_name: Name for output submission file
            
        Returns:
            Submission DataFrame
        """
        print(f"\n{'#'*80}")
        print("# TRAFFIC ANALYSIS INFERENCE PIPELINE")
        print(f"{'#'*80}\n")
        
        # Load test data
        df = self.load_test_data()
        
        # Extract features
        features = self.extract_features(df, extract_video=extract_video)
        
        # Make predictions (for both enter and exit)
        # Note: In practice, you'd need separate models or handle this differently
        predictions = self.predict(features)
        
        # Create submission
        output_path = self.path_config.get_prediction_path(output_name)
        submission = self.create_submission(df, predictions, predictions, output_path)
        
        print(f"\n{'#'*80}")
        print("# INFERENCE COMPLETE")
        print(f"{'#'*80}\n")
        
        return submission
    
    def predict_single_video(self, video_path: Path, timestamp: str) -> dict:
        """
        Predict for a single video.
        
        Args:
            video_path: Path to video file
            timestamp: Video timestamp
            
        Returns:
            Dictionary with predictions
        """
        # Create temporary DataFrame
        df = pd.DataFrame([{
            'video_path': str(video_path),
            'video_time': timestamp
        }])
        
        # Extract features
        features = self.extract_features(df, extract_video=True)
        
        # Predict
        prediction = self.predict(features)[0]
        
        return {
            'video_path': str(video_path),
            'timestamp': timestamp,
            'prediction': prediction
        }
