"""
Training Pipeline
=================
End-to-end pipeline for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm

from ..config import ModelConfig, FeatureConfig, PathConfig
from ..features import VideoFeatureExtractor, TemporalFeatureExtractor, StatisticalFeatureExtractor
from ..models import ModelTrainer, ModelEvaluator


class TrainingPipeline:
    """End-to-end training pipeline."""
    
    def __init__(self,
                 model_config: Optional[ModelConfig] = None,
                 feature_config: Optional[FeatureConfig] = None,
                 path_config: Optional[PathConfig] = None):
        """
        Initialize training pipeline.
        
        Args:
            model_config: Model configuration
            feature_config: Feature configuration
            path_config: Path configuration
        """
        self.model_config = model_config or ModelConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.path_config = path_config or PathConfig()
        
        # Create output directories
        self.path_config.create_directories()
        
        # Initialize extractors
        self.video_extractor = VideoFeatureExtractor(self.feature_config.video)
        self.temporal_extractor = TemporalFeatureExtractor(self.feature_config.temporal)
        self.statistical_extractor = StatisticalFeatureExtractor(self.feature_config.statistical)
        
        # Initialize trainer and evaluator
        self.trainer = ModelTrainer(self.model_config)
        self.evaluator = ModelEvaluator()
        
        self.features = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
    
    def load_data(self) -> pd.DataFrame:
        """
        Load training data.
        
        Returns:
            Training DataFrame
        """
        print(f"\n{'='*80}")
        print("Loading Training Data")
        print(f"{'='*80}")
        
        if not self.path_config.train_csv.exists():
            raise FileNotFoundError(f"Training data not found: {self.path_config.train_csv}")
        
        df = pd.read_csv(self.path_config.train_csv)
        print(f"Loaded {len(df)} training samples")
        
        return df
    
    def extract_features(self, df: pd.DataFrame, extract_video: bool = True) -> pd.DataFrame:
        """
        Extract all features from training data.
        
        Args:
            df: Training DataFrame
            extract_video: Whether to extract video features
            
        Returns:
            DataFrame with all features
        """
        print(f"\n{'='*80}")
        print("Feature Extraction")
        print(f"{'='*80}")
        
        features = df.copy()
        
        # Extract temporal features
        print("\n1. Extracting temporal features...")
        temporal_features = self.temporal_extractor.extract(df)
        features = pd.concat([features, temporal_features], axis=1)
        print(f"   Added {len(temporal_features.columns)} temporal features")
        
        # Extract video features (if enabled)
        if extract_video and 'video_path' in df.columns:
            print("\n2. Extracting video features...")
            video_paths = [self.path_config.video_dir / path for path in df['video_path']]
            video_features = self.video_extractor.extract_batch(video_paths)
            features = pd.concat([features, video_features.drop(columns=['video_path'])], axis=1)
            print(f"   Added {len(video_features.columns)-1} video features")
        
        # Extract statistical features
        print("\n3. Extracting statistical features...")
        stat_cols = [col for col in features.columns if col not in df.columns]
        if stat_cols:
            stat_features = self.statistical_extractor.extract(features, stat_cols)
            features = pd.concat([features, stat_features], axis=1)
            print(f"   Added {len(stat_features.columns)} statistical features")
        
        print(f"\n✓ Total features: {features.shape[1]}")
        
        self.features = features
        return features
    
    def prepare_data(self, features: pd.DataFrame, 
                    target_cols: list = ['congestion_enter_rating', 'congestion_exit_rating'],
                    val_split: float = 0.2) -> None:
        """
        Prepare train/validation split.
        
        Args:
            features: Feature DataFrame
            target_cols: Target column names
            val_split: Validation split ratio
        """
        print(f"\n{'='*80}")
        print("Preparing Train/Validation Split")
        print(f"{'='*80}")
        
        # Separate features and targets
        feature_cols = [col for col in features.columns if col not in target_cols]
        X = features[feature_cols]
        
        # For now, use the first target column
        y = features[target_cols[0]]
        
        # Train/val split
        split_idx = int(len(X) * (1 - val_split))
        
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_val = X.iloc[split_idx:]
        self.y_val = y.iloc[split_idx:]
        
        print(f"Training samples: {len(self.X_train)}")
        print(f"Validation samples: {len(self.X_val)}")
        print(f"Features: {self.X_train.shape[1]}")
    
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Training metrics
        """
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        metrics = self.trainer.train(self.X_train, self.y_train, 
                                     self.X_val, self.y_val)
        
        return metrics
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model.
        
        Returns:
            Evaluation metrics
        """
        if self.X_val is None:
            raise ValueError("Validation data not available.")
        
        y_pred = self.trainer.model.predict(self.X_val)
        metrics = self.evaluator.evaluate(self.y_val, y_pred, "Validation")
        
        return metrics
    
    def save_model(self, model_name: str = "traffic_model"):
        """
        Save trained model.
        
        Args:
            model_name: Name for saved model
        """
        model_path = self.path_config.get_model_path(model_name)
        self.trainer.save_model(model_path)
    
    def run(self, extract_video: bool = True) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            extract_video: Whether to extract video features
            
        Returns:
            Pipeline results
        """
        print(f"\n{'#'*80}")
        print("# TRAFFIC ANALYSIS TRAINING PIPELINE")
        print(f"{'#'*80}\n")
        
        # Load data
        df = self.load_data()
        
        # Extract features
        features = self.extract_features(df, extract_video=extract_video)
        
        # Prepare data
        self.prepare_data(features)
        
        # Train model
        train_metrics = self.train()
        
        # Evaluate model
        eval_metrics = self.evaluate()
        
        # Save model
        self.save_model()
        
        # Generate report
        report_path = self.path_config.reports_dir / "training_report.md"
        self.evaluator.generate_report(self.y_val, 
                                       self.trainer.model.predict(self.X_val),
                                       "Traffic Model",
                                       report_path)
        
        print(f"\n{'#'*80}")
        print("# PIPELINE COMPLETE")
        print(f"{'#'*80}\n")
        
        return {
            'train_metrics': train_metrics,
            'eval_metrics': eval_metrics
        }
