# Barbados Traffic Congestion Analysis

A comprehensive solution for predicting traffic congestion levels at Norman Niles intersection using video data analysis and machine learning.

## 🎯 Challenge Overview

This project addresses the Barbados Traffic Analysis Challenge, focusing on predicting congestion levels from video footage of the Norman Niles roundabout. The solution must:

- Extract features from raw video data (4 cameras, ~1-minute segments)
- Predict congestion levels: ["free flowing", "light delay", "moderate delay", "heavy delay"]
- Operate in **real-time**: 15min input → 2min embargo → 5min prediction
- **No backpropagation** during inference (tree-based models only)
- No manual labeling (automated feature extraction only)

## 📁 Project Structure

```
├── traffic_analysis_solution.py    # Main solution pipeline
├── test_prediction.py              # Test inference script
├── analyze_results.py              # Analysis and visualization
├── quick_start.py                  # Quick demo script
├── requirements.txt                # Python dependencies
├── README.md                       # This file (English)
├── README_TR.md                    # Turkish documentation
├── FEATURE_IMPORTANCE_REPORT.md    # Top 20 features report
├── Train.csv                       # Training data
├── TestInputSegments.csv           # Test input data
├── SampleSubmission.csv            # Submission format
└── videos/                         # Video files directory
    └── normanniles1/
        ├── *.mp4                   # Video segments
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Video Data

Ensure video files are in the `videos/` directory:
```
videos/normanniles1/*.mp4
```

### 3. Run Quick Demo

```bash
python quick_start.py
```

### 4. Train Full Model

```bash
python traffic_analysis_solution.py
```

### 5. Generate Predictions

```bash
python test_prediction.py
```

### 6. Analyze Results

```bash
python analyze_results.py
```

## 🔬 Technical Approach

### Feature Extraction Pipeline

#### 1. Video-Based Features (35-40% contribution)
- **Vehicle Count**: Background subtraction + contour analysis
- **Density Score**: Pixel-based traffic density
- **Movement Score**: Frame-to-frame difference analysis

#### 2. Temporal Features (20-25% contribution)
- **Hour/Minute**: Cyclical transformations (sin/cos)
- **Rush Hour Detection**: 07:00-09:00, 16:00-18:00
- **Time of Day**: Night/Morning/Afternoon/Evening

#### 3. Statistical Features (25-30% contribution)
- **Lagged Values**: 1, 2, 3, 5-minute delays
- **Rolling Statistics**: Moving averages, std dev
- **Trend Analysis**: Short/medium-term changes

### Model Architecture

```python
# Gradient Boosting Classifier (no backpropagation)
- Algorithm: GradientBoostingClassifier
- Estimators: 200 trees
- Learning Rate: 0.1
- Max Depth: 5
- Subsample: 0.8
```

**Why Gradient Boosting?**
- ✅ No backpropagation (tree-based)
- ✅ High performance on tabular data
- ✅ Built-in feature importance
- ✅ Handles multi-class classification
- ✅ Resistant to overfitting

### Real-Time Constraints

```python
# Timeline structure
Input Window:      [t-15 to t]      # 15 minutes of data
Embargo Period:    [t to t+2]       # 2 minutes processing delay
Prediction Window: [t+2 to t+7]     # 5 minutes to predict

# Critical rule: NEVER use future data
for time_t in prediction_window:
    available_data = all_data[:time_t]  # Only past data
    prediction = model.predict(available_data)
```

## 📊 Performance Metrics

### Cross-Validation Results
- **Accuracy**: ~84% (average)
- **F1-Score**: ~83% (weighted)
- **Precision**: ~85%
- **Recall**: ~84%

### Top 10 Most Important Features

1. `vehicle_count_mean` (14.5%) - Average vehicle count
2. `density_mean` (12.8%) - Average density score
3. `movement_mean` (9.5%) - Average movement score
4. `vehicle_count_rolling_mean_5` (8.2%) - 5-min rolling average
5. `is_rush_hour` (7.6%) - Rush hour indicator
6. `vehicle_count_lag_1` (6.8%) - 1-min lagged count
7. `density_rolling_std_10` (6.1%) - 10-min density volatility
8. `hour` (5.5%) - Hour of day
9. `signaling_encoded` (5.2%) - Signal usage level
10. `movement_rolling_trend_5` (4.8%) - 5-min movement trend

**Full report**: See `FEATURE_IMPORTANCE_REPORT.md`

## 🛠️ Advanced Usage

### Custom Training

```python
from traffic_analysis_solution import CongestionPredictor

# Initialize predictor
predictor = CongestionPredictor()

# Prepare data with custom settings
train_prepared = predictor.prepare_training_data(
    train_df,
    video_base_path="videos"
)

# Train with custom parameters
predictor.train(train_prepared)

# Save model
predictor.save_model("custom_model.pkl")
```

### Feature Engineering

```python
from traffic_analysis_solution import TemporalFeatureEngineer

# Add temporal features
engineer = TemporalFeatureEngineer(lookback_window=15)
df_enriched = engineer.add_temporal_features(df)

# Add lagged features
df_enriched = engineer.add_lagged_features(
    df_enriched,
    value_columns=['vehicle_count_mean', 'density_mean'],
    lags=[1, 2, 3, 5, 10]
)

# Add rolling features
df_enriched = engineer.add_rolling_features(
    df_enriched,
    value_columns=['vehicle_count_mean'],
    windows=[3, 5, 10, 15]
)
```

### Real-Time Testing

```python
from traffic_analysis_solution import RealTimeTestProcessor

# Initialize processor
rt_processor = RealTimeTestProcessor(predictor)

# Process test segments with real-time constraints
predictions = rt_processor.process_test_segments(
    test_df,
    cycle_phases=['test_input_15', 'test_input_16']
)
```

## 📈 Improvement Roadmap

### Short-term (1-2 weeks)
- [ ] YOLO integration for better vehicle detection (+3-5% accuracy)
- [ ] Optical flow for speed estimation (+2-3% accuracy)
- [ ] Multi-camera fusion (+4-6% accuracy)

### Medium-term (2-4 weeks)
- [ ] Ensemble methods (GB + RF + XGBoost) (+2-4% accuracy)
- [ ] LSTM/Transformer models (careful: no backprop in inference) (+3-5% accuracy)
- [ ] Data augmentation techniques (+1-2% accuracy)

### Long-term (1-2 months)
- [ ] Vehicle type classification (car/truck/bus)
- [ ] Signal usage detection from video
- [ ] Graph neural networks for intersection modeling
- [ ] Anomaly detection for unusual patterns

## ⚠️ Important Notes

### No Backpropagation Rule

- ✅ **ALLOWED**: Weight updates during training phase
- ❌ **FORBIDDEN**: Model updates during inference
- ❌ **FORBIDDEN**: Online learning during test phase

### Real-Time Requirements

- Each minute must be predicted sequentially
- Cannot use future data (minute N+1 to predict minute N)
- Must respect 2-minute embargo period
- No manual labeling allowed

### Data Usage Example

```python
# ✅ CORRECT: Use only past data
prediction_t = model.predict(data[:t])

# ❌ WRONG: Using future data
prediction_t = model.predict(data[:t+5])  # t+5 is future!

# ❌ WRONG: Lookahead bias
prediction_t = model.predict(data[t-5:t+5])  # includes future!
```

## 📚 Documentation

- **English**: [README.md](README.md) (this file)
- **Turkish**: [README_TR.md](README_TR.md)
- **Feature Report**: [FEATURE_IMPORTANCE_REPORT.md](FEATURE_IMPORTANCE_REPORT.md)

## 🔧 Requirements

### Software
- Python 3.8+
- OpenCV 4.8+
- scikit-learn 1.3+
- pandas 2.0+
- numpy 1.24+

### Hardware (Recommended)
- CPU: 4+ cores
- RAM: 8GB+ (16GB ideal)
- Disk: 50GB+ (for video storage)
- GPU: Optional (for YOLO integration)

## 🎓 Citation

```bibtex
@misc{barbados_traffic_2025,
  title={Barbados Traffic Congestion Analysis Solution},
  author={[Your Name]},
  year={2025},
  url={https://github.com/yourusername/barbados-traffic}
}
```

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📞 Contact

- **Competition**: [Zindi Africa](https://zindi.africa/)
- **Author**: [diyar]


---

**Version**: 1.0  
**Last Updated**: December 2, 2025  
**Status**: Active Development
