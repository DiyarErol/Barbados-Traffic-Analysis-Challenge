# Project Structure Visualization

## Directory Tree

```
Barbados-Traffic-Analysis-Challenge-main/
│
├── 📁 src/                                    # NEW: Modular source code
│   ├── 📄 __init__.py
│   │
│   ├── 📁 config/                            # Configuration management
│   │   ├── 📄 __init__.py
│   │   ├── 📄 model_config.py               # Model hyperparameters
│   │   ├── 📄 feature_config.py             # Feature settings
│   │   └── 📄 paths.py                      # Path management
│   │
│   ├── 📁 features/                          # Feature extraction modules
│   │   ├── 📄 __init__.py
│   │   ├── 📄 base.py                       # Abstract base class
│   │   ├── 📄 video_features.py             # Video processing (CV)
│   │   ├── 📄 temporal_features.py          # Time-based features
│   │   └── 📄 statistical_features.py       # Statistical aggregations
│   │
│   ├── 📁 models/                            # Model components
│   │   ├── 📄 __init__.py
│   │   ├── 📄 trainer.py                    # Training logic
│   │   ├── 📄 evaluator.py                  # Metrics & evaluation
│   │   ├── 📄 predictor.py                  # Inference
│   │   └── 📄 hybrid_model.py               # Hybrid architecture
│   │
│   └── 📁 pipelines/                         # End-to-end workflows
│       ├── 📄 __init__.py
│       ├── 📄 training_pipeline.py          # Training workflow
│       └── 📄 inference_pipeline.py         # Inference workflow
│
├── 📁 benchmarks/                            # NEW: Performance monitoring
│   └── 📄 performance_benchmark.py          # CPU/Memory/GPU tracking
│
├── 📁 docs/                                  # NEW: Documentation
│   ├── 📄 ARCHITECTURE.md                   # System design
│   ├── 📄 CONTRIBUTING.md                   # Contribution guide
│   ├── 📄 DATA_FORMAT.md                    # Data specifications
│   ├── 📄 USAGE_GUIDE.md                    # Usage tutorials
│   └── 📄 PROJECT_SUMMARY.md                # Improvement overview
│
├── 📁 examples/                              # NEW: Example scripts
│   └── 📄 complete_example.py               # Full workflow demo
│
├── 📁 tests/                                 # NEW: Unit tests
│   └── 📄 test_features.py                  # Feature tests
│
├── 📁 output/                                # Generated outputs
│   ├── 📁 models/                           # Trained models
│   ├── 📁 features/                         # Extracted features
│   ├── 📁 predictions/                      # Predictions
│   ├── 📁 logs/                             # Log files
│   └── 📁 plots/                            # Visualizations
│
├── 📁 videos/                                # Video data
│   └── 📁 normanniles1/                     # Video files
│
├── 📁 scripts/                               # Legacy scripts (kept)
│
├── 📄 Train.csv                              # Training data
├── 📄 TestInputSegments.csv                  # Test data
├── 📄 SampleSubmission.csv                   # Submission template
│
├── 📄 requirements.txt                       # Production dependencies
├── 📄 requirements-dev.txt                   # NEW: Dev dependencies
│
├── 📄 README.md                              # Original README
├── 📄 README_NEW.md                          # NEW: Updated README
├── 📄 ENHANCEMENT_COMPLETE.md                # NEW: Enhancement summary
│
└── 📄 .gitignore
```

## Module Dependencies

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface                          │
│  ┌────────────────────┐         ┌────────────────────┐      │
│  │  Training Pipeline │         │ Inference Pipeline │      │
│  └────────────────────┘         └────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                    │                        │
                    ▼                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Feature Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Video Features│  │Temporal Feat │  │Statistical   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Model Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Trainer    │  │  Evaluator   │  │  Predictor   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│  ┌──────────────────────────────────────────────────┐      │
│  │             Hybrid Model (Optional)               │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Model Config  │  │Feature Config│  │  Path Config │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Component Interaction Flow

### Training Flow
```
1. Load Data (CSV)
   ↓
2. Extract Features
   ├── Video Features (OpenCV)
   ├── Temporal Features (Time encoding)
   └── Statistical Features (Aggregations)
   ↓
3. Train Model
   ├── Gradient Boosting
   ├── Random Forest
   └── Hybrid (Optional)
   ↓
4. Evaluate
   ├── F1 Score
   ├── Confusion Matrix
   └── Feature Importance
   ↓
5. Save Model
```

### Inference Flow
```
1. Load Test Data (CSV)
   ↓
2. Extract Features (Same as training)
   ↓
3. Load Trained Model
   ↓
4. Predict
   ├── Single Model
   └── Ensemble (Optional)
   ↓
5. Generate Submission (CSV)
```

### Benchmarking Flow
```
1. Initialize Benchmark
   ↓
2. Run Function with Monitoring
   ├── CPU Usage
   ├── Memory Usage
   ├── GPU Usage (if available)
   └── Execution Time
   ↓
3. Calculate Metrics
   ├── Throughput
   ├── Peak Memory
   └── Resource Efficiency
   ↓
4. Generate Report
```

## Key Classes and Their Roles

```
BaseFeatureExtractor           → Abstract interface for extractors
  ├── VideoFeatureExtractor    → CV-based feature extraction
  ├── TemporalFeatureExtractor → Time-based features
  └── StatisticalFeatureExtractor → Stats and aggregations

ModelTrainer                   → Train ML models
ModelEvaluator                 → Evaluate performance
ModelPredictor                 → Production inference
EnsemblePredictor             → Multi-model ensemble
HybridModel                    → Tree + deep learning fusion

TrainingPipeline              → End-to-end training
InferencePipeline             → End-to-end inference

PerformanceBenchmark          → Monitor resource usage

ModelConfig                    → Model hyperparameters
FeatureConfig                  → Feature extraction settings
PathConfig                     → File path management
```

## Data Flow Diagram

```
┌──────────────┐
│  Raw Video   │
│    Files     │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Video Processing    │
│  - Frame extraction  │
│  - Vehicle detection │
│  - Motion analysis   │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐       ┌──────────────────────┐
│   CSV Data with      │◄──────│   Temporal Features  │
│   Video Features     │       │   - Hour/minute      │
└──────┬───────────────┘       │   - Rush hour        │
       │                       └──────────────────────┘
       │
       ▼
┌──────────────────────┐
│  Feature Matrix      │
│  (All features)      │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Model Training     │
│   - Cross-validation │
│   - Hyperparameters  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Trained Model      │
│   (.pkl file)        │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│   Predictions        │
│   (Submission CSV)   │
└──────────────────────┘
```

## File Size Summary

| Category | Files | Total Lines (Est.) |
|----------|-------|-------------------|
| Configuration | 4 | ~400 |
| Features | 5 | ~800 |
| Models | 5 | ~900 |
| Pipelines | 3 | ~500 |
| Benchmarks | 1 | ~400 |
| Tests | 1 | ~200 |
| Documentation | 5 | ~2,500 |
| Examples | 1 | ~100 |
| **Total** | **25+** | **~5,800+** |

## Quick Navigation

- **Getting Started**: `examples/complete_example.py`
- **Configuration**: `src/config/`
- **Feature Extraction**: `src/features/`
- **Model Training**: `src/models/trainer.py`
- **Inference**: `src/pipelines/inference_pipeline.py`
- **Benchmarking**: `benchmarks/performance_benchmark.py`
- **Documentation**: `docs/`
- **Tests**: `tests/`

---

*This structure provides a clear, professional organization that scales well for teams and production deployment.*
