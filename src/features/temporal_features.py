"""
Temporal Feature Extraction
===========================
Extract time-based features from timestamps.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict

from .base import BaseFeatureExtractor
from ..config.feature_config import TemporalFeatureConfig


class TemporalFeatureExtractor(BaseFeatureExtractor):
    """Extract temporal features from timestamps."""
    
    def __init__(self, config: TemporalFeatureConfig = None):
        """
        Initialize temporal feature extractor.
        
        Args:
            config: Temporal feature configuration
        """
        super().__init__(config or TemporalFeatureConfig())
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names based on config."""
        self.feature_names = []
        
        if self.config.use_cyclical_encoding:
            if self.config.encode_hour:
                self.feature_names.extend(['hour_sin', 'hour_cos'])
            if self.config.encode_minute:
                self.feature_names.extend(['minute_sin', 'minute_cos'])
            if self.config.encode_day_of_week:
                self.feature_names.extend(['day_sin', 'day_cos'])
            if self.config.encode_month:
                self.feature_names.extend(['month_sin', 'month_cos'])
        else:
            if self.config.encode_hour:
                self.feature_names.append('hour')
            if self.config.encode_minute:
                self.feature_names.append('minute')
            if self.config.encode_day_of_week:
                self.feature_names.append('day_of_week')
            if self.config.encode_month:
                self.feature_names.append('month')
        
        # Rush hour features
        self.feature_names.extend([
            'is_rush_hour',
            'is_morning_rush',
            'is_evening_rush'
        ])
        
        # Time of day
        self.feature_names.extend([
            'time_of_day',
            'is_night',
            'is_morning',
            'is_midday',
            'is_afternoon',
            'is_evening'
        ])
        
        # Additional time features
        self.feature_names.extend([
            'is_weekend',
            'is_weekday',
            'minutes_since_midnight',
            'minutes_until_midnight'
        ])
    
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from DataFrame with timestamp column.
        
        Args:
            data: DataFrame with 'video_time' or 'timestamp' column
            
        Returns:
            DataFrame with temporal features
        """
        df = data.copy()
        
        # Parse timestamp
        time_col = 'video_time' if 'video_time' in df.columns else 'timestamp'
        if time_col not in df.columns:
            raise ValueError("DataFrame must have 'video_time' or 'timestamp' column")
        
        df['datetime'] = pd.to_datetime(df[time_col])
        
        features = pd.DataFrame(index=df.index)
        
        # Extract cyclical time features
        if self.config.use_cyclical_encoding:
            features = pd.concat([features, self._extract_cyclical_features(df)], axis=1)
        else:
            features = pd.concat([features, self._extract_basic_time_features(df)], axis=1)
        
        # Rush hour features
        features = pd.concat([features, self._extract_rush_hour_features(df)], axis=1)
        
        # Time of day features
        features = pd.concat([features, self._extract_time_of_day_features(df)], axis=1)
        
        # Additional features
        features = pd.concat([features, self._extract_additional_features(df)], axis=1)
        
        return features
    
    def _extract_cyclical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclical encoding of time features."""
        features = pd.DataFrame(index=df.index)
        
        if self.config.encode_hour:
            hour = df['datetime'].dt.hour
            features['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        
        if self.config.encode_minute:
            minute = df['datetime'].dt.minute
            features['minute_sin'] = np.sin(2 * np.pi * minute / 60)
            features['minute_cos'] = np.cos(2 * np.pi * minute / 60)
        
        if self.config.encode_day_of_week:
            day = df['datetime'].dt.dayofweek
            features['day_sin'] = np.sin(2 * np.pi * day / 7)
            features['day_cos'] = np.cos(2 * np.pi * day / 7)
        
        if self.config.encode_month:
            month = df['datetime'].dt.month
            features['month_sin'] = np.sin(2 * np.pi * month / 12)
            features['month_cos'] = np.cos(2 * np.pi * month / 12)
        
        return features
    
    def _extract_basic_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract basic time features without cyclical encoding."""
        features = pd.DataFrame(index=df.index)
        
        if self.config.encode_hour:
            features['hour'] = df['datetime'].dt.hour
        
        if self.config.encode_minute:
            features['minute'] = df['datetime'].dt.minute
        
        if self.config.encode_day_of_week:
            features['day_of_week'] = df['datetime'].dt.dayofweek
        
        if self.config.encode_month:
            features['month'] = df['datetime'].dt.month
        
        return features
    
    def _extract_rush_hour_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract rush hour indicators."""
        features = pd.DataFrame(index=df.index)
        
        hour = df['datetime'].dt.hour
        
        morning_rush = (hour >= self.config.morning_rush_start) & (hour < self.config.morning_rush_end)
        evening_rush = (hour >= self.config.evening_rush_start) & (hour < self.config.evening_rush_end)
        
        features['is_rush_hour'] = (morning_rush | evening_rush).astype(int)
        features['is_morning_rush'] = morning_rush.astype(int)
        features['is_evening_rush'] = evening_rush.astype(int)
        
        return features
    
    def _extract_time_of_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time of day features."""
        features = pd.DataFrame(index=df.index)
        
        hour = df['datetime'].dt.hour
        
        # Create time of day categories
        time_of_day = pd.cut(hour, bins=self.config.time_bins, labels=self.config.time_labels, include_lowest=True)
        features['time_of_day'] = pd.Categorical(time_of_day).codes
        
        # Binary indicators
        features['is_night'] = (time_of_day == 'Night').astype(int)
        features['is_morning'] = (time_of_day == 'Morning').astype(int)
        features['is_midday'] = (time_of_day == 'Midday').astype(int)
        features['is_afternoon'] = (time_of_day == 'Afternoon').astype(int)
        features['is_evening'] = (time_of_day == 'Evening').astype(int)
        
        return features
    
    def _extract_additional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract additional temporal features."""
        features = pd.DataFrame(index=df.index)
        
        # Weekend indicator
        features['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        features['is_weekday'] = (df['datetime'].dt.dayofweek < 5).astype(int)
        
        # Minutes since/until midnight
        minutes_since = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute
        features['minutes_since_midnight'] = minutes_since
        features['minutes_until_midnight'] = 1440 - minutes_since
        
        return features
    
    def extract_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Extract lag features for a target column.
        
        Args:
            df: DataFrame with temporal data
            target_col: Column to create lags for
            
        Returns:
            DataFrame with lag features
        """
        lag_features = pd.DataFrame(index=df.index)
        
        for lag in self.config.lag_periods:
            lag_features[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        return lag_features
    
    def extract_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Extract rolling window statistics.
        
        Args:
            df: DataFrame with temporal data
            target_col: Column to compute rolling stats for
            
        Returns:
            DataFrame with rolling features
        """
        rolling_features = pd.DataFrame(index=df.index)
        
        for window in self.config.rolling_windows:
            for stat in self.config.rolling_stats:
                col_name = f'{target_col}_rolling_{window}_{stat}'
                
                if stat == 'mean':
                    rolling_features[col_name] = df[target_col].rolling(window).mean()
                elif stat == 'std':
                    rolling_features[col_name] = df[target_col].rolling(window).std()
                elif stat == 'min':
                    rolling_features[col_name] = df[target_col].rolling(window).min()
                elif stat == 'max':
                    rolling_features[col_name] = df[target_col].rolling(window).max()
        
        return rolling_features
