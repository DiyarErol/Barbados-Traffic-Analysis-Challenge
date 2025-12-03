# Usage Guide

Complete guide to using the modular Barbados Traffic Analysis system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Feature Extraction](#feature-extraction)
4. [Model Training](#model-training)
5. [Inference](#inference)
6. [Benchmarking](#benchmarking)
7. [Advanced Usage](#advanced-usage)

---

## Quick Start

### Installation

```powershell
# Clone repository
git clone <repository-url>
cd Barbados-Traffic-Analysis-Challenge-main

# Install dependencies
pip install -r requirements.txt
```

### Basic Training Pipeline

```python
from src.pipelines import TrainingPipeline
from src.config import ModelConfig, FeatureConfig

# Initialize pipeline with default configs
pipeline = TrainingPipeline()

# Run complete training
results = pipeline.run(extract_video=True)

print(f"Training F1: {results['train_metrics']['train_score']:.4f}")
print(f"Validation F1: {results['eval_metrics']['f1_macro']:.4f}")
```

### Basic Inference

```python
from src.pipelines import InferencePipeline
from pathlib import Path

# Load trained model
model_path = Path("output/models/traffic_model.pkl")
pipeline = InferencePipeline(model_path)

# Generate predictions
submission = pipeline.run(extract_video=True)
```

---

## Configuration

### Model Configuration

```python
from src.config import ModelConfig

# Create custom configuration
config = ModelConfig(
    model_type="gradient_boosting",
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    cv_folds=5
)

# Use in pipeline
from src.pipelines import TrainingPipeline
pipeline = TrainingPipeline(model_config=config)
```

### Feature Configuration

```python
from src.config import FeatureConfig, VideoFeatureConfig

# Customize video features
video_config = VideoFeatureConfig(
    target_fps=10,  # Process more frames
    resize_dims=(800, 600),  # Higher resolution
    use_background_subtraction=True
)

feature_config = FeatureConfig(video=video_config)

# Use in pipeline
pipeline = TrainingPipeline(feature_config=feature_config)
```

### Path Configuration

```python
from src.config import PathConfig
from pathlib import Path

# Custom paths
paths = PathConfig(
    root_dir=Path("c:/my_project"),
    video_dir=Path("d:/videos"),
    output_dir=Path("c:/output")
)

# Create directories
paths.create_directories()

# Use in pipeline
pipeline = TrainingPipeline(path_config=paths)
```

---

## Feature Extraction

### Video Features Only

```python
from src.features import VideoFeatureExtractor
from pathlib import Path

# Initialize extractor
extractor = VideoFeatureExtractor()

# Extract from single video
video_path = Path("videos/SEG001.mp4")
features = extractor.extract(video_path)

print(features)
# Output: DataFrame with 8 video features

# Batch extraction
video_paths = [Path(f"videos/SEG{i:03d}.mp4") for i in range(1, 11)]
batch_features = extractor.extract_batch(video_paths)
```

### Temporal Features

```python
from src.features import TemporalFeatureExtractor
import pandas as pd

# Initialize extractor
extractor = TemporalFeatureExtractor()

# Create sample data
df = pd.DataFrame({
    'video_time': ['2024-01-15 08:30:00', '2024-01-15 09:00:00']
})

# Extract features
temporal_features = extractor.extract(df)

print(temporal_features.columns.tolist())
# ['hour_sin', 'hour_cos', 'is_rush_hour', 'time_of_day', ...]
```

### Statistical Features

```python
from src.features import StatisticalFeatureExtractor

# Initialize extractor
extractor = StatisticalFeatureExtractor()

# Compute statistics for specific columns
feature_cols = ['vehicle_count', 'density_score']
stats = extractor.extract(df, feature_cols)

# Rolling statistics
rolling_stats = extractor.extract_rolling_stats(df, 'vehicle_count')

# Trend features
trends = extractor.extract_trend_features(df, 'density_score')
```

### Combined Feature Pipeline

```python
# Extract all features
from src.features import (
    VideoFeatureExtractor,
    TemporalFeatureExtractor,
    StatisticalFeatureExtractor
)

# Initialize extractors
video_ext = VideoFeatureExtractor()
temporal_ext = TemporalFeatureExtractor()
stat_ext = StatisticalFeatureExtractor()

# Load data
df = pd.read_csv('Train.csv')

# Extract temporal features
features = temporal_ext.extract(df)

# Extract video features
video_features = video_ext.extract_batch(video_paths)
features = pd.concat([features, video_features], axis=1)

# Extract statistical features
stat_features = stat_ext.extract(features, feature_cols)
features = pd.concat([features, stat_features], axis=1)
```

---

## Model Training

### Train with Different Algorithms

```python
from src.models import ModelTrainer
from src.config import ModelConfig

# Gradient Boosting
config = ModelConfig(model_type="gradient_boosting")
trainer = ModelTrainer(config)
trainer.train(X_train, y_train)

# Random Forest
config = ModelConfig(model_type="random_forest", n_estimators=500)
trainer = ModelTrainer(config)
trainer.train(X_train, y_train)

# XGBoost (if installed)
config = ModelConfig(model_type="xgboost")
trainer = ModelTrainer(config)
trainer.train(X_train, y_train)
```

### Cross-Validation

```python
# Perform k-fold cross-validation
cv_results = trainer.cross_validate(X, y)

print(f"CV Scores: {cv_results['cv_scores']}")
print(f"Mean: {cv_results['mean_score']:.4f}")
print(f"Std: {cv_results['std_score']:.4f}")
```

### Feature Importance

```python
# Get feature importance
importance = trainer.get_feature_importance(top_n=20)

print(importance)
# Shows top 20 features with importance scores
```

### Save/Load Models

```python
from pathlib import Path

# Save model
model_path = Path("models/my_model.pkl")
trainer.save_model(model_path)

# Load model
trainer.load_model(model_path)
```

---

## Inference

### Basic Prediction

```python
from src.models import ModelPredictor

# Load model
predictor = ModelPredictor("models/traffic_model.pkl")

# Make predictions
predictions = predictor.predict(X_test)

# Get probabilities
probabilities = predictor.predict_proba(X_test)
```

### Ensemble Prediction

```python
from src.models import EnsemblePredictor

# Load multiple models
model_paths = [
    Path("models/model1.pkl"),
    Path("models/model2.pkl"),
    Path("models/model3.pkl")
]

# Create ensemble with weights
weights = [0.5, 0.3, 0.2]
ensemble = EnsemblePredictor(model_paths, weights)

# Predict with voting
predictions = ensemble.predict(X_test, method='voting')

# Predict with weighted probabilities
predictions = ensemble.predict(X_test, method='weighted_proba')
```

### Hybrid Model

```python
from src.models import HybridModel
from src.config import HybridModelConfig

# Configure hybrid model
config = HybridModelConfig(
    use_deep_features=True,
    fusion_method='late',  # or 'early', 'stacking'
    tree_weight=0.7,
    deep_weight=0.3
)

# Initialize and train
hybrid = HybridModel(config)
hybrid.train(X_tree, y, X_deep)

# Predict
predictions = hybrid.predict(X_tree_test, X_deep_test)
```

---

## Benchmarking

### Performance Benchmarking

```python
from benchmarks.performance_benchmark import PerformanceBenchmark

# Initialize benchmark
benchmark = PerformanceBenchmark()

# Benchmark feature extraction
from src.features import VideoFeatureExtractor

extractor = VideoFeatureExtractor()
video_paths = [...]  # List of video paths

result = benchmark.benchmark_video_processing(extractor, video_paths)
print(result)

# Benchmark training
from src.models import ModelTrainer

trainer = ModelTrainer()
result = benchmark.benchmark_model_training(trainer, X_train, y_train)

# Benchmark inference
from src.models import ModelPredictor

predictor = ModelPredictor("models/model.pkl")
result = benchmark.benchmark_inference(predictor, X_test)

# Compare and save results
benchmark.compare_results()
benchmark.save_results()
benchmark.generate_report()
```

---

## Advanced Usage

### Custom Feature Extractor

```python
from src.features.base import BaseFeatureExtractor
import pandas as pd

class CustomFeatureExtractor(BaseFeatureExtractor):
    """Extract custom features."""
    
    def __init__(self, config=None):
        super().__init__(config)
        self.feature_names = ['custom_feat_1', 'custom_feat_2']
    
    def extract(self, data):
        """Extract features."""
        features = pd.DataFrame()
        
        # Your custom feature extraction logic
        features['custom_feat_1'] = data['some_col'] * 2
        features['custom_feat_2'] = data['another_col'].rolling(5).mean()
        
        return features

# Use in pipeline
extractor = CustomFeatureExtractor()
features = extractor.extract(df)
```

### Model Evaluation

```python
from src.models import ModelEvaluator

evaluator = ModelEvaluator()

# Evaluate predictions
metrics = evaluator.evaluate(y_true, y_pred, "Test Set")

# Plot confusion matrix
evaluator.plot_confusion_matrix(y_true, y_pred, 
                               save_path=Path("plots/confusion_matrix.png"))

# Analyze errors
errors = evaluator.analyze_errors(y_true, y_pred, X_test)

# Generate report
report = evaluator.generate_report(y_true, y_pred, 
                                   model_name="Traffic Model",
                                   save_path=Path("reports/evaluation.md"))
```

### Complete Custom Pipeline

```python
from src.pipelines import TrainingPipeline
from src.config import ModelConfig, FeatureConfig, PathConfig

# Configure everything
model_config = ModelConfig(
    model_type="gradient_boosting",
    n_estimators=500,
    learning_rate=0.05
)

feature_config = FeatureConfig()
path_config = PathConfig()

# Create pipeline
pipeline = TrainingPipeline(
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config
)

# Load and process data
df = pipeline.load_data()
features = pipeline.extract_features(df, extract_video=True)
pipeline.prepare_data(features)

# Train and evaluate
train_metrics = pipeline.train()
eval_metrics = pipeline.evaluate()

# Save model
pipeline.save_model("my_custom_model")
```

---

## Tips & Best Practices

### Memory Optimization

```python
# Process videos in batches
batch_size = 10
for i in range(0, len(video_paths), batch_size):
    batch = video_paths[i:i+batch_size]
    features = extractor.extract_batch(batch)
    # Save features immediately
    features.to_csv(f"features_batch_{i}.csv")
```

### GPU Acceleration

```python
# Enable GPU for deep features (if available)
from src.config import HybridModelConfig

config = HybridModelConfig(
    use_deep_features=True,
    use_gpu=True,
    gpu_id=0
)
```

### Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

### Reproducibility

```python
# Set random seeds in configuration
config = ModelConfig(random_state=42)

# Also set numpy/sklearn seeds
import numpy as np
np.random.seed(42)
```

---

## Troubleshooting

### Common Issues

**Issue: Video not found**
```python
# Check video path
from pathlib import Path
video_path = Path("videos/SEG001.mp4")
print(f"Exists: {video_path.exists()}")
```

**Issue: Feature mismatch**
```python
# Ensure feature order matches
expected_features = predictor.feature_names
X_test = X_test[expected_features]
```

**Issue: Memory error**
```python
# Reduce batch size or image resolution
config = VideoFeatureConfig(
    resize_dims=(320, 240),  # Smaller resolution
    target_fps=3  # Fewer frames
)
```

---

## Next Steps

- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- See [DATA_FORMAT.md](DATA_FORMAT.md) for data specifications
- Review example scripts in `examples/` directory
