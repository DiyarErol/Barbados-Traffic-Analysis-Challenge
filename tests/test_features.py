"""
Unit Tests for Feature Extractors
=================================
Example tests demonstrating testing approach.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.features import (
    VideoFeatureExtractor,
    TemporalFeatureExtractor,
    StatisticalFeatureExtractor
)
from src.config import (
    VideoFeatureConfig,
    TemporalFeatureConfig,
    StatisticalFeatureConfig
)


class TestVideoFeatureExtractor:
    """Tests for VideoFeatureExtractor."""
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = VideoFeatureExtractor()
        assert extractor is not None
        assert len(extractor.feature_names) > 0
        assert 'vehicle_count' in extractor.feature_names
    
    def test_initialization_with_config(self):
        """Test initialization with custom config."""
        config = VideoFeatureConfig(target_fps=10, resize_dims=(800, 600))
        extractor = VideoFeatureExtractor(config)
        
        assert extractor.config.target_fps == 10
        assert extractor.config.resize_dims == (800, 600)
    
    def test_feature_names(self):
        """Test that all expected features are present."""
        extractor = VideoFeatureExtractor()
        expected_features = [
            'vehicle_count',
            'density_score',
            'movement_score',
            'avg_contour_area',
            'motion_intensity',
            'frame_difference',
            'foreground_ratio',
            'active_regions'
        ]
        
        for feature in expected_features:
            assert feature in extractor.feature_names


class TestTemporalFeatureExtractor:
    """Tests for TemporalFeatureExtractor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample temporal data."""
        return pd.DataFrame({
            'video_time': [
                '2024-01-15 08:30:00',
                '2024-01-15 09:00:00',
                '2024-01-15 17:30:00'
            ]
        })
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = TemporalFeatureExtractor()
        assert extractor is not None
        assert len(extractor.feature_names) > 0
    
    def test_extract_features(self, sample_data):
        """Test feature extraction from timestamps."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_data)
        
        assert len(features) == len(sample_data)
        assert features.shape[1] > 0
        
        # Check that features exist
        assert 'is_rush_hour' in features.columns
        assert 'time_of_day' in features.columns
    
    def test_cyclical_encoding(self, sample_data):
        """Test cyclical encoding of time features."""
        config = TemporalFeatureConfig(use_cyclical_encoding=True)
        extractor = TemporalFeatureExtractor(config)
        features = extractor.extract(sample_data)
        
        assert 'hour_sin' in features.columns
        assert 'hour_cos' in features.columns
        
        # Check that values are in valid range
        assert features['hour_sin'].between(-1, 1).all()
        assert features['hour_cos'].between(-1, 1).all()
    
    def test_rush_hour_detection(self, sample_data):
        """Test rush hour detection."""
        extractor = TemporalFeatureExtractor()
        features = extractor.extract(sample_data)
        
        # First row: 08:30 - morning rush
        assert features.loc[0, 'is_morning_rush'] == 1
        
        # Second row: 09:00 - still morning rush
        assert features.loc[1, 'is_morning_rush'] == 1
        
        # Third row: 17:30 - evening rush
        assert features.loc[2, 'is_evening_rush'] == 1


class TestStatisticalFeatureExtractor:
    """Tests for StatisticalFeatureExtractor."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data with numeric features."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100) * 10
        })
    
    def test_initialization(self):
        """Test extractor initialization."""
        extractor = StatisticalFeatureExtractor()
        assert extractor is not None
    
    def test_extract_basic_stats(self, sample_data):
        """Test basic statistical feature extraction."""
        extractor = StatisticalFeatureExtractor()
        features = extractor.extract(sample_data, ['feature1'])
        
        # Check that statistics were computed
        assert 'feature1_mean' in features.columns
        assert 'feature1_std' in features.columns
        assert 'feature1_min' in features.columns
        assert 'feature1_max' in features.columns
    
    def test_rolling_stats(self, sample_data):
        """Test rolling window statistics."""
        config = StatisticalFeatureConfig(rolling_windows=[5, 10])
        extractor = StatisticalFeatureExtractor(config)
        
        rolling_features = extractor.extract_rolling_stats(sample_data, 'feature1')
        
        assert 'feature1_roll_mean_5' in rolling_features.columns
        assert 'feature1_roll_mean_10' in rolling_features.columns
        
        # Check that first few values are NaN (due to rolling window)
        assert rolling_features['feature1_roll_mean_5'].iloc[0:4].isnull().any()
    
    def test_trend_features(self, sample_data):
        """Test trend feature extraction."""
        config = StatisticalFeatureConfig(compute_trends=True, trend_windows=[5])
        extractor = StatisticalFeatureExtractor(config)
        
        trend_features = extractor.extract_trend_features(sample_data, 'feature1')
        
        assert 'feature1_diff_5' in trend_features.columns
        assert 'feature1_pct_change_5' in trend_features.columns
        assert 'feature1_trend_5' in trend_features.columns


class TestFeatureIntegration:
    """Integration tests for feature extraction pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create comprehensive sample data."""
        return pd.DataFrame({
            'video_time': [
                '2024-01-15 08:30:00',
                '2024-01-15 09:00:00',
                '2024-01-15 17:30:00'
            ],
            'video_id': ['SEG001', 'SEG002', 'SEG003']
        })
    
    def test_combined_feature_extraction(self, sample_data):
        """Test combining features from multiple extractors."""
        # Extract temporal features
        temporal_ext = TemporalFeatureExtractor()
        temporal_features = temporal_ext.extract(sample_data)
        
        # Combine with original data
        combined = pd.concat([sample_data, temporal_features], axis=1)
        
        # Check that we have both original and extracted features
        assert 'video_id' in combined.columns
        assert 'is_rush_hour' in combined.columns
        assert len(combined) == len(sample_data)


# Example of parameterized test
@pytest.mark.parametrize("model_type,expected_params", [
    ("gradient_boosting", ["n_estimators", "learning_rate"]),
    ("random_forest", ["n_estimators", "max_depth"]),
])
def test_model_config(model_type, expected_params):
    """Test model configuration for different types."""
    from src.config import ModelConfig
    
    config = ModelConfig(model_type=model_type)
    sklearn_params = config.to_sklearn_params()
    
    for param in expected_params:
        assert param in sklearn_params


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
