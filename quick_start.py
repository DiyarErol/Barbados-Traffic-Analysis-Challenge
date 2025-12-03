"""
Quick start script - Example usage
NOTE: Real video files are required
"""

import os
import pandas as pd
from traffic_analysis_solution import CongestionPredictor
import warnings
warnings.filterwarnings('ignore')


def quick_start_demo():
    """
    Quick demo - check data presence and show basic workflow
    """
    print("=" * 80)
    print("BARBADOS TRAFFIC ANALYSIS - QUICK START")
    print("=" * 80)
    
    # Check data files
    print("\n1. Checking data files...")
    
    required_files = {
        'Train.csv': 'Training data',
        'TestInputSegments.csv': 'Test input data',
        'SampleSubmission.csv': 'Sample submission format'
    }
    
    all_exists = True
    for file, desc in required_files.items():
        if os.path.exists(file):
            print(f"   ✓ {file} - {desc}")
        else:
            print(f"   ✗ {file} - {desc} NOT FOUND!")
            all_exists = False
    
    # Check video folder
    video_dir = "videos"
    if os.path.exists(video_dir):
        video_count = sum([len(files) for _, _, files in os.walk(video_dir)])
        print(f"   ✓ {video_dir}/ folder - {video_count} files")
    else:
        print(f"   ✗ {video_dir}/ folder NOT FOUND!")
        print(f"      Note: Place video files under {video_dir}/")
        all_exists = False
    
    if not all_exists:
        print("\n⚠️  Missing files! Please prepare all required files.")
        print("\nRequired structure:")
        print("  ├── Train.csv")
        print("  ├── TestInputSegments.csv")
        print("  ├── SampleSubmission.csv")
        print("  └── videos/")
        print("      └── normanniles1/")
        print("          ├── normanniles1_2025-10-20-06-00-45.mp4")
        print("          └── ...")
        return
    
    # Inspect data
    print("\n2. Data statistics...")
    train_df = pd.read_csv('Train.csv')
    test_df = pd.read_csv('TestInputSegments.csv')
    
    print(f"   Training samples: {len(train_df)}")
    print(f"   Test samples: {len(test_df)}")
    
    # Class distribution
    print("\n   Training - Enter class distribution:")
    for label, count in train_df['congestion_enter_rating'].value_counts().items():
        pct = 100 * count / len(train_df)
        print(f"     {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("\n   Training - Exit class distribution:")
    for label, count in train_df['congestion_exit_rating'].value_counts().items():
        pct = 100 * count / len(train_df)
        print(f"     {label:20s}: {count:5d} ({pct:5.1f}%)")
    
    # Sample training
    print("\n3. Demo model training (first 500 samples, all classes included)...")
    print("   NOTE: Use full data for complete training!")
    
    choice = input("\n   Start demo training? (y/n): ").lower()
    
    if choice == 'y':
        print("\n   Starting...")
        
        # Balanced subset with samples from each class
        # Ensures the model sees all classes
        train_subset = []
        for label in train_df['congestion_enter_rating'].unique():
            label_samples = train_df[train_df['congestion_enter_rating'] == label].head(125)
            train_subset.append(label_samples)
        train_subset = pd.concat(train_subset).head(500)
        
        predictor = CongestionPredictor()
        
        print("   Extracting video features...")
        train_prepared = predictor.prepare_training_data(
            train_subset,
            video_base_path="videos"
        )
        
        print("   Training model...")
        predictor.train(train_prepared)
        
        print("   Saving model...")
        predictor.save_model("demo_model.pkl")
        
        print("\n   ✓ Demo training completed!")
        print("     Model: demo_model.pkl")
        
        # Feature importance
        top_feats = predictor.get_top_features(n=10)
        print("\n   Top 10 features:")
        print(top_feats[['Feature_Enter', 'Importance_Enter']].to_string(index=False))
    
    print("\n4. Next steps:")
    print("   a) Full training:")
    print("      python traffic_analysis_solution.py")
    print("   ")
    print("   b) Test prediction:")
    print("      python test_prediction.py")
    print("   ")
    print("   c) Detailed information:")
    print("      README.md")
    
    print("\n" + "=" * 80)
    print("QUICK START COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    quick_start_demo()
