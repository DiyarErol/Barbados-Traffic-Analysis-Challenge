"""
Prediction script for the test dataset.
Runs under real-time constraints.
"""

import pandas as pd
from traffic_analysis_solution import (
    CongestionPredictor,
    RealTimeTestProcessor,
    create_submission_file
)


def main():
    print("=" * 80)
    print("Test Prediction Starting")
    print("=" * 80)
    
    # Paths
    test_csv = "TestInputSegments.csv"
    sample_sub = "SampleSubmission.csv"
    model_path = "congestion_model.pkl"
    video_base_path = "videos"
    
    # Load model
    print("\n1. Loading model...")
    predictor = CongestionPredictor()
    predictor.load_model(model_path)
    
    # Load test data
    print("\n2. Loading test data...")
    test_df = pd.read_csv(test_csv)
    print(f"   Test samples: {len(test_df)}")
    
    # Prepare test data (feature extraction)
    print("\n3. Extracting features for test data...")
    test_prepared = predictor.prepare_training_data(
        test_df,
        video_base_path=video_base_path
    )
    
    # Real-time test processor
    print("\n4. Running real-time predictions...")
    print("   (15 min input + 2 min embargo + 5 min prediction)")
    
    rt_processor = RealTimeTestProcessor(predictor)
    
    # Collect all test cycle phases
    cycle_phases = test_df['cycle_phase'].unique()
    print(f"   Test phases: {list(cycle_phases)}")
    
    # Predictions
    predictions = rt_processor.process_test_segments(
        test_prepared,
        cycle_phases
    )
    
    print(f"\n   Total predictions: {len(predictions)}")
    
    # Create submission file
    print("\n5. Creating submission file...")
    create_submission_file(
        predictions,
        sample_sub,
        output_path="submission.csv"
    )
    
    print("\n" + "=" * 80)
    print("Test prediction completed!")
    print("File: submission.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
