# Migration Guide: From Legacy to Modular Architecture

This guide helps you transition from the old monolithic scripts to the new modular architecture.

## Quick Comparison

### Old Way (Legacy)
```python
# Run monolithic script
python traffic_analysis_solution.py

# Hard to customize
# Difficult to test
# No performance monitoring
# Limited model options
```

### New Way (Modular)
```python
# Use clean pipeline
from src.pipelines import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run()

# Easy to customize
# Simple to test
# Built-in benchmarking
# Multiple model support
```

---

## Step-by-Step Migration

### 1. Update Imports

**Old:**
```python
# Everything in one file
import cv2
import pandas as pd
# ... scattered imports
```

**New:**
```python
# Organized imports
from src.pipelines import TrainingPipeline, InferencePipeline
from src.config import ModelConfig, FeatureConfig
from src.features import VideoFeatureExtractor
from src.models import ModelTrainer
```

### 2. Configuration Management

**Old:**
```python
# Hardcoded values
n_estimators = 200
learning_rate = 0.1
max_depth = 7
```

**New:**
```python
# Centralized configuration
from src.config import ModelConfig

config = ModelConfig(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=7
)
```

### 3. Feature Extraction

**Old:**
```python
# Inline feature extraction
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    # ... lots of code ...
    return features
```

**New:**
```python
# Modular extraction
from src.features import VideoFeatureExtractor

extractor = VideoFeatureExtractor()
features = extractor.extract(video_path)
```

### 4. Model Training

**Old:**
```python
# Manual training
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1
)
model.fit(X_train, y_train)
```

**New:**
```python
# Pipeline-based training
from src.models import ModelTrainer
from src.config import ModelConfig

config = ModelConfig(model_type="gradient_boosting")
trainer = ModelTrainer(config)
trainer.train(X_train, y_train)
trainer.save_model("models/traffic_model.pkl")
```

### 5. Complete Workflow

**Old:**
```python
# Multiple scripts to run manually
python preprocess_data.py
python train_model.py
python make_predictions.py
```

**New:**
```python
# Single pipeline
from src.pipelines import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run()  # Does everything
```

---

## Common Migration Patterns

### Pattern 1: Feature Extraction

**Old Pattern:**
```python
def process_video(video_path):
    features = {}
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Extract features...
    
    return features
```

**New Pattern:**
```python
from src.features import VideoFeatureExtractor

extractor = VideoFeatureExtractor()
features = extractor.extract(video_path)
# Or batch process
features = extractor.extract_batch(video_paths)
```

### Pattern 2: Model Selection

**Old Pattern:**
```python
# Switch between models manually
if model_type == "gb":
    model = GradientBoostingClassifier(...)
elif model_type == "rf":
    model = RandomForestClassifier(...)
```

**New Pattern:**
```python
from src.config import ModelConfig

# Just change config
config = ModelConfig(model_type="gradient_boosting")
# Or
config = ModelConfig(model_type="random_forest")
```

### Pattern 3: Evaluation

**Old Pattern:**
```python
from sklearn.metrics import f1_score

pred = model.predict(X_test)
score = f1_score(y_test, pred, average='macro')
print(f"Score: {score}")
```

**New Pattern:**
```python
from src.models import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_test, pred)
# Gets F1, confusion matrix, detailed report automatically
```

---

## Code Conversion Examples

### Example 1: Simple Training Script

**Old Code:**
```python
# old_train.py
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

# Load data
df = pd.read_csv('Train.csv')

# Manual feature extraction
# ... 100+ lines of code ...

# Train
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Save
import joblib
joblib.dump(model, 'model.pkl')
```

**New Code:**
```python
# new_train.py
from src.pipelines import TrainingPipeline

pipeline = TrainingPipeline()
results = pipeline.run()
# Done! Model saved automatically
```

### Example 2: Custom Feature Extraction

**Old Code:**
```python
# custom_features.py
def extract_my_features(df):
    features = []
    for _, row in df.iterrows():
        # Custom logic
        features.append(custom_value)
    return features
```

**New Code:**
```python
# custom_features.py
from src.features.base import BaseFeatureExtractor
import pandas as pd

class MyFeatureExtractor(BaseFeatureExtractor):
    def extract(self, data):
        features = pd.DataFrame()
        # Custom logic
        return features

# Use it
extractor = MyFeatureExtractor()
features = extractor.extract(df)
```

### Example 3: Model Evaluation

**Old Code:**
```python
# evaluate.py
from sklearn.metrics import f1_score, confusion_matrix

pred = model.predict(X_test)
f1 = f1_score(y_test, pred, average='macro')
cm = confusion_matrix(y_test, pred)

print(f"F1: {f1}")
print("Confusion Matrix:")
print(cm)
```

**New Code:**
```python
# evaluate.py
from src.models import ModelEvaluator

evaluator = ModelEvaluator()
metrics = evaluator.evaluate(y_test, pred, "Test Set")
evaluator.plot_confusion_matrix(y_test, pred)
evaluator.generate_report(y_test, pred, save_path="report.md")
```

---

## File Mapping

### Where Did Things Move?

| Old File | New Location | Notes |
|----------|--------------|-------|
| `traffic_analysis_solution.py` | `src/pipelines/training_pipeline.py` | Now modular |
| Feature extraction code | `src/features/` | Split into modules |
| Model training code | `src/models/trainer.py` | Separated |
| Inference code | `src/pipelines/inference_pipeline.py` | Clean interface |
| Configuration | `src/config/` | Centralized |
| Utility functions | `src/utils/` | Organized |

---

## Configuration Migration

### Old Configuration
```python
# Scattered throughout code
N_ESTIMATORS = 200
LEARNING_RATE = 0.1
VIDEO_FPS = 5
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480
```

### New Configuration
```python
from src.config import ModelConfig, FeatureConfig

model_config = ModelConfig(
    n_estimators=200,
    learning_rate=0.1
)

feature_config = FeatureConfig()
feature_config.video.target_fps = 5
feature_config.video.resize_dims = (640, 480)
```

---

## Testing Migration

### Old Testing
```python
# Manual testing
if __name__ == "__main__":
    # Test code here
    print("Testing...")
```

### New Testing
```python
# tests/test_features.py
import pytest

def test_video_extractor():
    from src.features import VideoFeatureExtractor
    
    extractor = VideoFeatureExtractor()
    assert extractor is not None
    # More tests...

# Run with: pytest tests/
```

---

## Performance Monitoring

### Old Approach
```python
import time

start = time.time()
# Do something
duration = time.time() - start
print(f"Took {duration}s")
```

### New Approach
```python
from benchmarks.performance_benchmark import PerformanceBenchmark

benchmark = PerformanceBenchmark()
result = benchmark.benchmark(
    my_function,
    args,
    name="My Operation"
)
# Gets CPU, memory, throughput automatically
```

---

## Common Issues and Solutions

### Issue 1: Can't Find Old Script

**Problem:** "Where is `traffic_analysis_solution.py`?"

**Solution:** It's been split into modules. Use the pipeline instead:
```python
from src.pipelines import TrainingPipeline
pipeline = TrainingPipeline()
pipeline.run()
```

### Issue 2: Imports Not Working

**Problem:** `ImportError: No module named 'src'`

**Solution:** Make sure you're in the project root directory:
```powershell
cd Barbados-Traffic-Analysis-Challenge-main
python examples/complete_example.py
```

### Issue 3: Custom Features

**Problem:** "How do I add my own features?"

**Solution:** Create a custom extractor:
```python
from src.features.base import BaseFeatureExtractor

class MyExtractor(BaseFeatureExtractor):
    def extract(self, data):
        # Your logic
        return features_df
```

### Issue 4: Different Model

**Problem:** "How do I use XGBoost instead?"

**Solution:** Change the configuration:
```python
config = ModelConfig(model_type="xgboost")
```

---

## Gradual Migration Strategy

### Phase 1: Keep Both (Recommended)
1. Keep old scripts for reference
2. Test new system in parallel
3. Compare results

### Phase 2: Switch Gradually
1. Start with inference pipeline
2. Move to training pipeline
3. Add custom features if needed

### Phase 3: Full Migration
1. Remove old scripts
2. Use only new architecture
3. Add tests
4. Add monitoring

---

## Benefits of Migration

| Benefit | Impact |
|---------|--------|
| **Modularity** | Easy to modify individual components |
| **Testing** | Can test each part independently |
| **Performance** | Built-in monitoring and optimization |
| **Collaboration** | Clear structure for teams |
| **Maintenance** | Easier to fix bugs and add features |
| **Documentation** | Comprehensive guides available |

---

## Getting Help

- **Documentation**: See `docs/` directory
- **Examples**: Check `examples/complete_example.py`
- **Architecture**: Read `docs/ARCHITECTURE.md`
- **Usage Guide**: See `docs/USAGE_GUIDE.md`

---

## Next Steps

1. ✅ Read this migration guide
2. ✅ Run `examples/complete_example.py`
3. ✅ Try simple modifications
4. ✅ Read architecture documentation
5. ✅ Gradually migrate your code

---

*The new architecture is designed to be intuitive. Most tasks are simpler than before!*
