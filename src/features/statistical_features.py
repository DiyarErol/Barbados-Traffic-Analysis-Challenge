"""
Statistical Feature Extraction
==============================
Extract statistical features and aggregations.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple

from .base import BaseFeatureExtractor
from ..config.feature_config import StatisticalFeatureConfig


class StatisticalFeatureExtractor(BaseFeatureExtractor):
    """Extract statistical features from data."""
    
    def __init__(self, config: StatisticalFeatureConfig = None):
        """
        Initialize statistical feature extractor.
        
        Args:
            config: Statistical feature configuration
        """
        super().__init__(config or StatisticalFeatureConfig())
    
    def extract(self, data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Extract statistical features from specified columns.
        
        Args:
            data: Input DataFrame
            feature_cols: Columns to compute statistics for
            
        Returns:
            DataFrame with statistical features
        """
        features = pd.DataFrame(index=data.index)
        
        for col in feature_cols:
            if col not in data.columns:
                continue
            
            # Basic statistics
            if self.config.compute_mean:
                features[f'{col}_mean'] = data[col].mean()
            
            if self.config.compute_std:
                features[f'{col}_std'] = data[col].std()
            
            if self.config.compute_min:
                features[f'{col}_min'] = data[col].min()
            
            if self.config.compute_max:
                features[f'{col}_max'] = data[col].max()
            
            if self.config.compute_median:
                features[f'{col}_median'] = data[col].median()
            
            # Percentiles
            if self.config.compute_percentiles:
                for p in self.config.percentiles:
                    features[f'{col}_p{p}'] = data[col].quantile(p/100)
        
        return features
    
    def extract_rolling_stats(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Extract rolling window statistics.
        
        Args:
            df: Input DataFrame (must be sorted by time)
            col: Column to compute rolling stats for
            
        Returns:
            DataFrame with rolling features
        """
        features = pd.DataFrame(index=df.index)
        
        for window in self.config.agg_windows:
            if self.config.compute_mean:
                features[f'{col}_roll_mean_{window}'] = df[col].rolling(window).mean()
            
            if self.config.compute_std:
                features[f'{col}_roll_std_{window}'] = df[col].rolling(window).std()
            
            if self.config.compute_min:
                features[f'{col}_roll_min_{window}'] = df[col].rolling(window).min()
            
            if self.config.compute_max:
                features[f'{col}_roll_max_{window}'] = df[col].rolling(window).max()
            
            if self.config.compute_median:
                features[f'{col}_roll_median_{window}'] = df[col].rolling(window).median()
        
        return features
    
    def extract_trend_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """
        Extract trend analysis features.
        
        Args:
            df: Input DataFrame (must be sorted by time)
            col: Column to analyze trends for
            
        Returns:
            DataFrame with trend features
        """
        if not self.config.compute_trends:
            return pd.DataFrame(index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        for window in self.config.trend_windows:
            # Simple difference
            features[f'{col}_diff_{window}'] = df[col].diff(window)
            
            # Percentage change
            features[f'{col}_pct_change_{window}'] = df[col].pct_change(window)
            
            # Linear trend (slope)
            features[f'{col}_trend_{window}'] = self._calculate_rolling_slope(df[col], window)
        
        return features
    
    def _calculate_rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling linear trend slope."""
        def slope(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
        
        return series.rolling(window).apply(slope, raw=False)
    
    def extract_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract interaction features between columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        if not self.config.compute_interactions:
            return pd.DataFrame(index=df.index)
        
        features = pd.DataFrame(index=df.index)
        
        # Default interactions if not specified
        if not self.config.interaction_pairs:
            return features
        
        for col1, col2 in self.config.interaction_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Multiplication
                features[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Division (with safety)
                features[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                
                # Addition
                features[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
                
                # Difference
                features[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
        
        return features
    
    def extract_aggregations(self, df: pd.DataFrame, group_col: str, agg_cols: List[str]) -> pd.DataFrame:
        """
        Extract group-wise aggregations.
        
        Args:
            df: Input DataFrame
            group_col: Column to group by
            agg_cols: Columns to aggregate
            
        Returns:
            DataFrame with aggregated features
        """
        agg_dict = {}
        
        for col in agg_cols:
            if col not in df.columns:
                continue
            
            agg_funcs = []
            if self.config.compute_mean:
                agg_funcs.append('mean')
            if self.config.compute_std:
                agg_funcs.append('std')
            if self.config.compute_min:
                agg_funcs.append('min')
            if self.config.compute_max:
                agg_funcs.append('max')
            if self.config.compute_median:
                agg_funcs.append('median')
            
            agg_dict[col] = agg_funcs
        
        grouped = df.groupby(group_col).agg(agg_dict)
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        
        return df.merge(grouped, left_on=group_col, right_index=True, how='left')
    
    def extract_variance_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Extract variance-based features.
        
        Args:
            df: Input DataFrame
            cols: Columns to compute variance features for
            
        Returns:
            DataFrame with variance features
        """
        features = pd.DataFrame(index=df.index)
        
        for col in cols:
            if col not in df.columns:
                continue
            
            # Coefficient of variation
            mean_val = df[col].mean()
            std_val = df[col].std()
            features[f'{col}_cv'] = std_val / (mean_val + 1e-8)
            
            # Range
            features[f'{col}_range'] = df[col].max() - df[col].min()
            
            # Interquartile range
            features[f'{col}_iqr'] = df[col].quantile(0.75) - df[col].quantile(0.25)
        
        return features
