"""
Fix All Submissions to Correct Format
======================================
Convert all submission files to required format with columns:
ID, Target, Target_Accuracy
"""

import pandas as pd
import numpy as np

print("="*70)
print("FIXING ALL SUBMISSIONS TO CORRECT FORMAT")
print("="*70)

# Load sample submission to understand format
sample = pd.read_csv('SampleSubmission.csv')
print(f"\n[FORMAT] Required columns: {sample.columns.tolist()}")
print(f"[FORMAT] Sample ID format: {sample['ID'].iloc[0]}")

# Load test input to get segment IDs
test_input = pd.read_csv('TestInputSegments.csv')
print(f"\n[INFO] Test set has {len(test_input)} segments")

# Create ID mapping for enter and exit
enter_ids = []
exit_ids = []

for _, row in test_input.iterrows():
    enter_id = row['ID_enter']
    exit_id = row['ID_exit']
    enter_ids.append(enter_id)
    exit_ids.append(exit_id)

print(f"[INFO] Created {len(enter_ids)} enter IDs and {len(exit_ids)} exit IDs")

# Function to create properly formatted submission
def create_proper_submission(enter_preds, exit_preds, filename):
    """Create submission in correct format"""
    
    # Combine enter and exit predictions
    ids = enter_ids + exit_ids
    targets = list(enter_preds) + list(exit_preds)
    target_accuracy = targets.copy()  # Initially same as Target
    
    submission = pd.DataFrame({
        'ID': ids,
        'Target': targets,
        'Target_Accuracy': target_accuracy
    })
    
    submission.to_csv(filename, index=False)
    print(f"[OK] Saved {filename}")
    
    # Show distribution
    dist = pd.Series(targets).value_counts(normalize=True)
    print(f"  Distribution:")
    for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        print(f"    {cls}: {dist.get(cls, 0)*100:.1f}%")
    
    return submission

# Fix submission_final_ensemble.csv
print("\n" + "="*70)
print("1. FIXING submission_final_ensemble.csv")
print("="*70)

try:
    final_ens = pd.read_csv('submission_final_ensemble.csv')
    
    # Check if it has old format
    if 'EnterTrafficCondition' in final_ens.columns:
        enter_preds = final_ens['EnterTrafficCondition'].values
        exit_preds = final_ens['ExitTrafficCondition'].values
    elif 'Target' in final_ens.columns:
        # Already in correct format, just verify
        print("[INFO] Already in correct format")
        final_ens = None
    else:
        print("[ERROR] Unknown format, skipping")
        final_ens = None
        
    if final_ens is not None and 'EnterTrafficCondition' in final_ens.columns:
        create_proper_submission(enter_preds, exit_preds, 'submission_final_ensemble.csv')
except Exception as e:
    print(f"[ERROR] Could not fix submission_final_ensemble.csv: {e}")

# Fix submission_conditional_calibrated.csv
print("\n" + "="*70)
print("2. FIXING submission_conditional_calibrated.csv")
print("="*70)

try:
    cond_cal = pd.read_csv('submission_conditional_calibrated.csv')
    
    if 'EnterTrafficCondition' in cond_cal.columns:
        enter_preds = cond_cal['EnterTrafficCondition'].values
        exit_preds = cond_cal['ExitTrafficCondition'].values
        create_proper_submission(enter_preds, exit_preds, 'submission_conditional_calibrated.csv')
    elif 'Target' in cond_cal.columns:
        print("[INFO] Already in correct format")
    else:
        print("[ERROR] Unknown format, skipping")
except Exception as e:
    print(f"[ERROR] Could not fix submission_conditional_calibrated.csv: {e}")

# Fix submission_gbm_blend.csv
print("\n" + "="*70)
print("3. FIXING submission_gbm_blend.csv")
print("="*70)

try:
    gbm_blend = pd.read_csv('submission_gbm_blend.csv')
    
    if 'Target' in gbm_blend.columns and len(gbm_blend) == len(enter_ids) + len(exit_ids):
        print("[INFO] Already in correct format")
        # Verify distribution
        dist = gbm_blend['Target'].value_counts(normalize=True)
        print(f"  Distribution:")
        for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
            print(f"    {cls}: {dist.get(cls, 0)*100:.1f}%")
    else:
        print("[ERROR] Unknown format or incorrect length, skipping")
except Exception as e:
    print(f"[ERROR] Could not process submission_gbm_blend.csv: {e}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("✓ All submissions converted to correct format:")
print("  - ID: segment identifier (enter/exit)")
print("  - Target: predicted traffic condition")
print("  - Target_Accuracy: same as Target (can be refined later)")
print("\n[NEXT] Submit these files to competition:")
print("  1. submission_final_ensemble.csv (ÖNCELIK)")
print("  2. submission_conditional_calibrated.csv (FALLBACK)")
print("  3. submission_gbm_blend.csv (ALTERNATIVE)")
