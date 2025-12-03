"""
Base Feature Extractor
======================
Abstract base class for all feature extractors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd
import numpy as np
from pathlib import Path


class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""
    
    def __init__(self, config: Any = None):
        """
        Initialize feature extractor.
        
        Args:
            config: Configuration object for this extractor
        """
        self.config = config
        self.feature_names: List[str] = []
    
    @abstractmethod
    def extract(self, data: Any) -> pd.DataFrame:
        """
        Extract features from input data.
        
        Args:
            data: Input data (varies by extractor type)
            
        Returns:
            DataFrame with extracted features
        """
        pass
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names produced by this extractor."""
        return self.feature_names
    
    def validate_input(self, data: Any) -> bool:
        """
        Validate input data format.
        
        Args:
            data: Input data to validate
            
        Returns:
            True if valid, raises ValueError otherwise
        """
        return True
    
    def save_features(self, features: pd.DataFrame, path: Path):
        """Save extracted features to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(path, index=False)
    
    def load_features(self, path: Path) -> pd.DataFrame:
        """Load previously extracted features."""
        return pd.read_csv(path)
