# Architecture Guide

## Overview

This document describes the modular architecture of the Barbados Traffic Analysis system, designed for scalability, maintainability, and extensibility.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Input Layer                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Video Files │  │  CSV Data    │  │  Timestamps  │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Feature Extraction Layer                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Video      │  │  Temporal    │  │ Statistical  │      │
│  │  Features    │  │  Features    │  │  Features    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Model Training Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Tree Models  │  │ Deep Learning│  │   Hybrid     │      │
│  │   (GB/RF)    │  │   Features   │  │  Ensembles   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Inference/Prediction Layer                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Predictor   │  │  Ensemble    │  │  API/Service │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

### 1. Configuration (`src/config/`)

Central configuration management for all system components.

**Files:**
- `model_config.py`: Model hyperparameters and training settings
- `feature_config.py`: Feature extraction parameters
- `paths.py`: File path management

**Key Classes:**
- `ModelConfig`: Tree-based model configuration
- `HybridModelConfig`: Hybrid model settings
- `FeatureConfig`: Master feature configuration
- `PathConfig`: Path management

### 2. Feature Extraction (`src/features/`)

Independent, modular feature extractors following the Strategy pattern.

**Files:**
- `base.py`: Abstract base class for all extractors
- `video_features.py`: Video-based feature extraction
- `temporal_features.py`: Time-based features
- `statistical_features.py`: Statistical aggregations

**Key Classes:**
- `BaseFeatureExtractor`: Abstract interface
- `VideoFeatureExtractor`: CV-based vehicle detection
- `TemporalFeatureExtractor`: Cyclical time encoding
- `StatisticalFeatureExtractor`: Rolling stats, trends

### 3. Models (`src/models/`)

Model training, evaluation, and inference components.

**Files:**
- `trainer.py`: Model training with cross-validation
- `evaluator.py`: Comprehensive evaluation metrics
- `predictor.py`: Production inference
- `hybrid_model.py`: Hybrid tree + deep learning architecture

**Key Classes:**
- `ModelTrainer`: Train with multiple algorithms (GB, RF, XGBoost, LightGBM)
- `ModelEvaluator`: F1 scores, confusion matrices, reports
- `ModelPredictor`: Production inference
- `EnsemblePredictor`: Multiple model ensemble
- `HybridModel`: Tree + deep learning fusion

### 4. Pipelines (`src/pipelines/`)

End-to-end workflows orchestrating all components.

**Files:**
- `training_pipeline.py`: Complete training workflow
- `inference_pipeline.py`: Production inference workflow

**Key Classes:**
- `TrainingPipeline`: Data → Features → Training → Evaluation
- `InferencePipeline`: Data → Features → Prediction → Submission

### 5. Benchmarking (`benchmarks/`)

Performance monitoring and optimization tools.

**Files:**
- `performance_benchmark.py`: CPU, memory, GPU monitoring

**Key Classes:**
- `PerformanceBenchmark`: Monitor resource usage
- `BenchmarkResult`: Store benchmark metrics

## Design Patterns

### 1. Strategy Pattern (Feature Extraction)
Each feature extractor implements the same interface (`BaseFeatureExtractor`), allowing easy swapping and extension.

### 2. Pipeline Pattern
End-to-end workflows combine multiple stages in a clear, sequential manner.

### 3. Configuration Object Pattern
All settings centralized in configuration objects, supporting serialization and easy modification.

### 4. Factory Pattern (Model Creation)
Models created through factory methods based on configuration.

## Data Flow

### Training Flow
```
CSV Data → Load → Temporal Features → Video Features → 
Statistical Features → Feature Selection → Train Model → 
Evaluate → Save Model
```

### Inference Flow
```
Test Data → Load → Extract Features → Load Model → 
Predict → Format Submission → Save
```

## Extension Points

### Adding New Features
1. Create new extractor class inheriting from `BaseFeatureExtractor`
2. Implement `extract()` method
3. Add configuration to `FeatureConfig`
4. Register in pipeline

### Adding New Models
1. Add model type to `ModelConfig`
2. Implement creation in `ModelTrainer._create_model()`
3. Update ensemble configurations

### Hybrid Model Strategies
Three fusion methods supported:
- **Early Fusion**: Concatenate features before training
- **Late Fusion**: Weighted combination of predictions
- **Stacking**: Use base predictions as meta-features

## Performance Optimization

### Memory Management
- Lazy loading of video data
- Batch processing with generators
- Feature caching to disk

### CPU Optimization
- Parallel video processing
- Efficient OpenCV operations
- NumPy vectorization

### GPU Support
- Optional YOLO detection for videos
- Deep learning feature extraction
- Mixed precision training

## Testing Strategy

### Unit Tests
- Test each feature extractor independently
- Test model training with synthetic data
- Test pipeline components

### Integration Tests
- Test full training pipeline
- Test full inference pipeline
- Test ensemble predictions

### Performance Tests
- Benchmark feature extraction speed
- Monitor memory usage
- Measure inference latency

## Best Practices

1. **Configuration First**: Always use configuration objects
2. **Modularity**: Keep components independent
3. **Error Handling**: Graceful degradation (e.g., skip failed videos)
4. **Logging**: Comprehensive logging at each stage
5. **Reproducibility**: Set random seeds in configurations
6. **Documentation**: Document all public methods

## Migration from Legacy Code

To migrate from the old monolithic scripts:

1. **Extract Configuration**: Move hardcoded values to config files
2. **Separate Concerns**: Break down large functions into modules
3. **Use Pipelines**: Replace scripts with pipeline classes
4. **Add Tests**: Create unit tests for each component
5. **Benchmark**: Compare performance before/after

## Future Enhancements

1. **Real-time Processing**: Streaming video analysis
2. **Distributed Training**: Multi-GPU support
3. **AutoML Integration**: Automated hyperparameter tuning
4. **Model Versioning**: Track model iterations
5. **A/B Testing**: Compare model variants in production
