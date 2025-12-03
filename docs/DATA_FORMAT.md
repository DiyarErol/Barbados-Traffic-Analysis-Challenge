# Data Format Specification

## Overview

This document specifies all data formats used in the Barbados Traffic Analysis system, including input data, intermediate features, model outputs, and submission formats.

## Table of Contents

1. [Input Data](#input-data)
2. [Feature Formats](#feature-formats)
3. [Model Outputs](#model-outputs)
4. [Submission Format](#submission-format)
5. [Configuration Files](#configuration-files)

---

## Input Data

### Training Data (`Train.csv`)

CSV file containing labeled traffic data.

**Columns:**
```
video_id                     : str     - Unique identifier for video segment
video_time                   : str     - Timestamp (format: YYYY-MM-DD HH:MM:SS)
signaling                    : bool    - Traffic signal status (True/False)
congestion_enter_rating      : str     - Entry congestion label
congestion_exit_rating       : str     - Exit congestion label
```

**Congestion Labels:**
- `"free flowing"` - No congestion
- `"light delay"` - Minor delays
- `"moderate delay"` - Moderate congestion
- `"heavy delay"` - Severe congestion

**Example:**
```csv
video_id,video_time,signaling,congestion_enter_rating,congestion_exit_rating
SEG001,2024-01-15 08:30:00,True,light delay,free flowing
SEG002,2024-01-15 08:45:00,False,moderate delay,light delay
```

**Requirements:**
- No missing values in required columns
- Timestamps in ascending order (recommended)
- Valid congestion labels only

### Test Input Data (`TestInputSegments.csv`)

CSV file containing unlabeled test data.

**Columns:**
```
video_id       : str - Unique identifier for video segment
video_time     : str - Timestamp (format: YYYY-MM-DD HH:MM:SS)
signaling      : bool - Traffic signal status
```

**Example:**
```csv
video_id,video_time,signaling
TEST001,2024-02-01 09:00:00,True
TEST002,2024-02-01 09:15:00,False
```

### Video Files

**Location:** `videos/normanniles1/`

**Format:** MP4 video files

**Naming:** Corresponds to `video_id` in CSV files

**Specifications:**
- Duration: ~60 seconds
- Resolution: Variable (will be resized)
- FPS: Variable
- Codec: H.264 (preferred)

**Example Structure:**
```
videos/
└── normanniles1/
    ├── SEG001.mp4
    ├── SEG002.mp4
    └── ...
```

---

## Feature Formats

### Video Features

Extracted from video files using computer vision.

**Format:** CSV or NumPy array

**Features:**
```
vehicle_count        : float - Average detected vehicles per frame
density_score        : float - Traffic density (0-1)
movement_score       : float - Frame-to-frame motion (0-1)
avg_contour_area     : float - Average vehicle contour size
motion_intensity     : float - Overall motion magnitude (0-1)
frame_difference     : float - Average frame difference (0-1)
foreground_ratio     : float - Foreground pixel ratio (0-1)
active_regions       : float - Number of active traffic regions
```

**Storage:**
```csv
video_id,vehicle_count,density_score,movement_score,...
SEG001,15.3,0.67,0.45,...
```

### Temporal Features

Extracted from timestamps.

**Cyclical Encoding:**
```
hour_sin            : float - sin(2π * hour / 24)
hour_cos            : float - cos(2π * hour / 24)
minute_sin          : float - sin(2π * minute / 60)
minute_cos          : float - cos(2π * minute / 60)
day_sin             : float - sin(2π * day / 7)
day_cos             : float - cos(2π * day / 7)
```

**Categorical Features:**
```
is_rush_hour        : int   - Rush hour indicator (0/1)
is_morning_rush     : int   - Morning rush (0/1)
is_evening_rush     : int   - Evening rush (0/1)
time_of_day         : int   - Encoded time period (0-4)
is_weekend          : int   - Weekend indicator (0/1)
```

**Numeric Features:**
```
minutes_since_midnight    : int   - Minutes elapsed since 00:00
minutes_until_midnight    : int   - Minutes remaining until 00:00
```

### Statistical Features

Derived from rolling windows and aggregations.

**Rolling Statistics:**
```
{feature}_roll_mean_{window}   : float - Rolling mean
{feature}_roll_std_{window}    : float - Rolling standard deviation
{feature}_roll_min_{window}    : float - Rolling minimum
{feature}_roll_max_{window}    : float - Rolling maximum
```

**Trend Features:**
```
{feature}_diff_{window}        : float - Difference from n periods ago
{feature}_pct_change_{window}  : float - Percentage change
{feature}_trend_{window}       : float - Linear trend slope
```

**Example:**
```
vehicle_count_roll_mean_5 : 14.8
vehicle_count_trend_10    : 0.23
```

---

## Model Outputs

### Predictions

**Format:** NumPy array or pandas Series

**Values:** One of the congestion labels:
- `"free flowing"`
- `"light delay"`
- `"moderate delay"`
- `"heavy delay"`

**Shape:** `(n_samples,)` where n_samples is number of test samples

### Prediction Probabilities

**Format:** NumPy array

**Shape:** `(n_samples, n_classes)` where n_classes = 4

**Columns:** Probabilities for each class in order:
```
[free_flowing_prob, light_delay_prob, moderate_delay_prob, heavy_delay_prob]
```

**Example:**
```python
array([[0.1, 0.6, 0.25, 0.05],   # Sample 1: Likely "light delay"
       [0.7, 0.2, 0.08, 0.02]])  # Sample 2: Likely "free flowing"
```

### Model Metadata

**Format:** JSON

**File:** `{model_name}.json`

**Structure:**
```json
{
  "model_type": "gradient_boosting",
  "feature_names": ["feature1", "feature2", ...],
  "training_history": {
    "train_score": 0.85,
    "val_score": 0.82,
    "n_samples": 1000,
    "n_features": 50
  },
  "config": {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 7
  }
}
```

---

## Submission Format

### Required Format (`SampleSubmission.csv`)

CSV file with predictions for test data.

**Columns:**
```
video_id          : str - Test video identifier
prediction_label  : str - Type of prediction (enter/exit)
prediction        : str - Congestion label
```

**Prediction Label Values:**
- `"congestion_enter_rating"` - Entry prediction
- `"congestion_exit_rating"` - Exit prediction

**Requirements:**
- Exactly 2 rows per test video (enter + exit)
- Must include all test video IDs
- Valid congestion labels only

**Example:**
```csv
video_id,prediction_label,prediction
TEST001,congestion_enter_rating,light delay
TEST001,congestion_exit_rating,free flowing
TEST002,congestion_enter_rating,moderate delay
TEST002,congestion_exit_rating,light delay
```

**Validation:**
```python
# Verify submission format
def validate_submission(submission_df, test_df):
    # Check columns
    required_cols = ['video_id', 'prediction_label', 'prediction']
    assert all(col in submission_df.columns for col in required_cols)
    
    # Check number of rows
    assert len(submission_df) == len(test_df) * 2
    
    # Check valid labels
    valid_labels = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    assert submission_df['prediction'].isin(valid_labels).all()
    
    # Check prediction labels
    valid_pred_labels = ['congestion_enter_rating', 'congestion_exit_rating']
    assert submission_df['prediction_label'].isin(valid_pred_labels).all()
```

---

## Configuration Files

### Model Configuration

**Format:** Python dataclass or JSON

**Example (JSON):**
```json
{
  "model_type": "gradient_boosting",
  "n_estimators": 200,
  "learning_rate": 0.1,
  "max_depth": 7,
  "min_samples_split": 50,
  "min_samples_leaf": 20,
  "random_state": 42
}
```

### Feature Configuration

**Format:** Python dataclass or JSON

**Example (JSON):**
```json
{
  "video": {
    "target_fps": 5,
    "resize_dims": [640, 480],
    "use_background_subtraction": true
  },
  "temporal": {
    "use_cyclical_encoding": true,
    "encode_hour": true,
    "encode_minute": true
  },
  "statistical": {
    "compute_mean": true,
    "compute_std": true,
    "rolling_windows": [3, 5, 10]
  }
}
```

---

## Data Validation

### Input Validation

```python
def validate_training_data(df):
    """Validate training data format."""
    required_columns = [
        'video_id', 'video_time', 'signaling',
        'congestion_enter_rating', 'congestion_exit_rating'
    ]
    
    # Check columns exist
    for col in required_columns:
        assert col in df.columns, f"Missing column: {col}"
    
    # Check no missing values
    assert not df[required_columns].isnull().any().any()
    
    # Validate congestion labels
    valid_labels = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    assert df['congestion_enter_rating'].isin(valid_labels).all()
    assert df['congestion_exit_rating'].isin(valid_labels).all()
```

### Feature Validation

```python
def validate_features(features_df):
    """Validate extracted features."""
    # Check for NaN values
    nan_cols = features_df.columns[features_df.isnull().any()].tolist()
    if nan_cols:
        print(f"Warning: NaN values in columns: {nan_cols}")
    
    # Check for infinite values
    inf_cols = features_df.columns[np.isinf(features_df).any()].tolist()
    if inf_cols:
        print(f"Warning: Infinite values in columns: {inf_cols}")
```

---

## Notes

- All timestamps should be in ISO format or `YYYY-MM-DD HH:MM:SS`
- Feature values should be numeric (float or int)
- Missing video files are handled gracefully (zero features)
- All CSV files should use UTF-8 encoding
- Boolean values: `True`/`False` or `1`/`0`
