"""
Feature Extraction Configuration
================================
Configuration for all feature extraction parameters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VideoFeatureConfig:
    """Configuration for video-based feature extraction."""
    
    # Video processing
    target_fps: int = 5  # Process every Nth frame
    resize_dims: Tuple[int, int] = (640, 480)
    
    # Background subtraction
    use_background_subtraction: bool = True
    bg_history: int = 500
    bg_var_threshold: int = 16
    bg_detect_shadows: bool = True
    
    # Vehicle detection
    min_contour_area: int = 500
    max_contour_area: int = 50000
    
    # Density calculation
    grid_size: Tuple[int, int] = (8, 8)
    density_threshold: float = 0.3
    
    # Motion detection
    motion_threshold: int = 25
    motion_blur_kernel: int = 5
    
    # Deep learning detection (optional)
    use_yolo: bool = False
    yolo_model: str = "yolov8n"
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45


@dataclass
class TemporalFeatureConfig:
    """Configuration for temporal feature extraction."""
    
    # Time encoding
    use_cyclical_encoding: bool = True
    encode_hour: bool = True
    encode_minute: bool = True
    encode_day_of_week: bool = True
    encode_month: bool = True
    
    # Rush hour detection
    morning_rush_start: int = 7
    morning_rush_end: int = 9
    evening_rush_start: int = 16
    evening_rush_end: int = 18
    
    # Time of day bins
    time_bins: List[int] = field(default_factory=lambda: [0, 6, 10, 14, 18, 24])
    time_labels: List[str] = field(default_factory=lambda: ['Night', 'Morning', 'Midday', 'Afternoon', 'Evening'])
    
    # Lag features
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    
    # Rolling statistics
    rolling_windows: List[int] = field(default_factory=lambda: [3, 5, 10, 15])
    rolling_stats: List[str] = field(default_factory=lambda: ['mean', 'std', 'min', 'max'])


@dataclass
class StatisticalFeatureConfig:
    """Configuration for statistical feature extraction."""
    
    # Aggregation windows
    agg_windows: List[int] = field(default_factory=lambda: [5, 10, 15, 30])
    
    # Statistics to compute
    compute_mean: bool = True
    compute_std: bool = True
    compute_min: bool = True
    compute_max: bool = True
    compute_median: bool = True
    compute_percentiles: bool = True
    percentiles: List[int] = field(default_factory=lambda: [25, 50, 75])
    
    # Trend analysis
    compute_trends: bool = True
    trend_windows: List[int] = field(default_factory=lambda: [5, 10, 15])
    
    # Interaction features
    compute_interactions: bool = True
    interaction_pairs: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class FeatureConfig:
    """Master feature configuration."""
    
    video: VideoFeatureConfig = field(default_factory=VideoFeatureConfig)
    temporal: TemporalFeatureConfig = field(default_factory=TemporalFeatureConfig)
    statistical: StatisticalFeatureConfig = field(default_factory=StatisticalFeatureConfig)
    
    # Feature selection
    use_feature_selection: bool = True
    feature_selection_method: str = "importance"  # Options: importance, correlation, recursive
    max_features: int = 100
    
    # Feature importance threshold
    importance_threshold: float = 0.001
    
    # Correlation threshold
    correlation_threshold: float = 0.95
