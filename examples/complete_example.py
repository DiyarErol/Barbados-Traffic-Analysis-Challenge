# Example: Complete Training and Inference

"""
This script demonstrates the complete workflow from data loading
to model training and inference using the modular architecture.
"""

from pathlib import Path
from src.config import ModelConfig, FeatureConfig, PathConfig
from src.pipelines import TrainingPipeline, InferencePipeline
from benchmarks.performance_benchmark import PerformanceBenchmark

def main():
    print("="*80)
    print("BARBADOS TRAFFIC ANALYSIS - COMPLETE EXAMPLE")
    print("="*80)
    
    # ========================================
    # 1. CONFIGURATION
    # ========================================
    print("\n1. Setting up configuration...")
    
    # Model configuration
    model_config = ModelConfig(
        model_type="gradient_boosting",
        n_estimators=200,
        learning_rate=0.1,
        max_depth=7,
        cv_folds=5
    )
    
    # Feature configuration
    feature_config = FeatureConfig()
    
    # Path configuration
    path_config = PathConfig()
    path_config.create_directories()
    
    print("✓ Configuration complete")
    
    # ========================================
    # 2. TRAINING
    # ========================================
    print("\n2. Training model...")
    
    # Initialize training pipeline
    train_pipeline = TrainingPipeline(
        model_config=model_config,
        feature_config=feature_config,
        path_config=path_config
    )
    
    # Run training (set extract_video=False for faster testing)
    results = train_pipeline.run(extract_video=False)
    
    print(f"✓ Training complete")
    print(f"  - Train F1: {results['train_metrics']['train_score']:.4f}")
    print(f"  - Val F1: {results['eval_metrics']['f1_macro']:.4f}")
    
    # ========================================
    # 3. INFERENCE
    # ========================================
    print("\n3. Running inference...")
    
    # Initialize inference pipeline
    model_path = path_config.get_model_path("traffic_model")
    inference_pipeline = InferencePipeline(
        model_path=model_path,
        feature_config=feature_config,
        path_config=path_config
    )
    
    # Generate predictions
    submission = inference_pipeline.run(
        extract_video=False,
        output_name="submission"
    )
    
    print(f"✓ Inference complete")
    print(f"  - Generated {len(submission)} predictions")
    
    # ========================================
    # 4. BENCHMARKING (Optional)
    # ========================================
    print("\n4. Running performance benchmarks...")
    
    benchmark = PerformanceBenchmark(output_dir=path_config.output_dir / "benchmarks")
    
    # Benchmark training
    from src.models import ModelTrainer
    trainer = ModelTrainer(model_config)
    
    if train_pipeline.X_train is not None:
        benchmark.benchmark_model_training(
            trainer, 
            train_pipeline.X_train.head(100),  # Small sample
            train_pipeline.y_train.head(100)
        )
    
    # Save benchmark results
    benchmark.save_results()
    benchmark.generate_report()
    
    print("✓ Benchmarking complete")
    
    # ========================================
    # 5. SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Model saved to: {path_config.models_dir}")
    print(f"Predictions saved to: {path_config.predictions_dir}")
    print(f"Reports saved to: {path_config.reports_dir}")
    print(f"Benchmarks saved to: {path_config.output_dir / 'benchmarks'}")
    print("="*80)


if __name__ == "__main__":
    main()
