"""
Dev 14: Threshold Optimization
===============================
Optimize decision thresholds on OOF predictions to maximize macro F1
while maintaining distribution constraints close to Train distribution.

Strategy:
- Load OOF predictions from dev12_2_gbm_validation.py
- Grid search optimal thresholds for each class
- Apply soft distribution constraints
- Generate optimized submission
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, accuracy_score, classification_report
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

print("Loading OOF predictions from GBM validation...")
with open('oof_predictions_gbm.pkl', 'rb') as f:
    oof_data = pickle.load(f)

train = pd.read_csv('Train.csv')
test_input = pd.read_csv('TestInputSegments.csv')

# Target distribution from Train (correct column names)
target_dist_enter = train['congestion_enter_rating'].value_counts(normalize=True).to_dict()
target_dist_exit = train['congestion_exit_rating'].value_counts(normalize=True).to_dict()

print("\nTarget distributions:")
print("Enter:", {k: f"{v*100:.1f}%" for k, v in target_dist_enter.items()})
print("Exit:", {k: f"{v*100:.1f}%" for k, v in target_dist_exit.items()})

# Extract OOF predictions
oof_enter_probs = oof_data['enter_blend_probs']  # (n_samples, 4)
oof_exit_probs = oof_data['exit_blend_probs']
y_enter_true = oof_data['y_enter']
y_exit_true = oof_data['y_exit']

class_names = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}

def apply_thresholds(probs, thresholds):
    """
    Apply class-specific thresholds.
    thresholds: array of shape (4,) for each class
    For each sample, predict class with highest adjusted score: prob - threshold
    """
    adjusted_scores = probs - thresholds
    return np.argmax(adjusted_scores, axis=1)

def distribution_penalty(pred_dist, target_dist, weight=0.1):
    """Penalize deviation from target distribution"""
    penalty = 0
    for cls in class_names:
        target = target_dist.get(cls, 0)
        pred = pred_dist.get(cls, 0)
        penalty += abs(pred - target)
    return weight * penalty

def objective_enter(thresholds):
    """Objective: maximize F1 while staying close to target distribution"""
    preds = apply_thresholds(oof_enter_probs, thresholds)
    f1 = f1_score(y_enter_true, preds, average='macro')
    
    # Distribution penalty
    pred_dist = pd.Series([class_names[p] for p in preds]).value_counts(normalize=True).to_dict()
    dist_pen = distribution_penalty(pred_dist, target_dist_enter, weight=0.15)
    
    # We minimize negative F1 + penalty
    return -(f1 - dist_pen)

def objective_exit(thresholds):
    """Objective: maximize F1 while staying close to target distribution"""
    preds = apply_thresholds(oof_exit_probs, thresholds)
    f1 = f1_score(y_exit_true, preds, average='macro')
    
    # Distribution penalty
    pred_dist = pd.Series([class_names[p] for p in preds]).value_counts(normalize=True).to_dict()
    dist_pen = distribution_penalty(pred_dist, target_dist_exit, weight=0.15)
    
    return -(f1 - dist_pen)

print("\n" + "="*70)
print("PHASE 1: Optimizing Enter thresholds...")
print("="*70)

# Baseline (no threshold adjustment)
baseline_enter_preds = np.argmax(oof_enter_probs, axis=1)
baseline_enter_f1 = f1_score(y_enter_true, baseline_enter_preds, average='macro')
print(f"\nBaseline Enter F1: {baseline_enter_f1:.4f}")
print("Baseline Enter distribution:")
baseline_enter_dist = pd.Series([class_names[p] for p in baseline_enter_preds]).value_counts(normalize=True)
for cls in class_names:
    print(f"  {cls}: {baseline_enter_dist.get(cls, 0)*100:.1f}%")

# Optimize Enter thresholds
bounds_enter = [(-0.3, 0.3) for _ in range(4)]  # threshold range for each class
result_enter = differential_evolution(
    objective_enter,
    bounds_enter,
    seed=42,
    maxiter=100,
    popsize=15,
    tol=0.001,
    workers=1
)

optimal_thresholds_enter = result_enter.x
print(f"\nOptimal Enter thresholds: {optimal_thresholds_enter}")

# Evaluate optimized Enter
optimized_enter_preds = apply_thresholds(oof_enter_probs, optimal_thresholds_enter)
optimized_enter_f1 = f1_score(y_enter_true, optimized_enter_preds, average='macro')
print(f"Optimized Enter F1: {optimized_enter_f1:.4f} (Δ={optimized_enter_f1-baseline_enter_f1:+.4f})")
print("Optimized Enter distribution:")
optimized_enter_dist = pd.Series([class_names[p] for p in optimized_enter_preds]).value_counts(normalize=True)
for cls in class_names:
    print(f"  {cls}: {optimized_enter_dist.get(cls, 0)*100:.1f}%")

print("\n" + "="*70)
print("PHASE 2: Optimizing Exit thresholds...")
print("="*70)

# Baseline Exit
baseline_exit_preds = np.argmax(oof_exit_probs, axis=1)
baseline_exit_f1 = f1_score(y_exit_true, baseline_exit_preds, average='macro')
print(f"\nBaseline Exit F1: {baseline_exit_f1:.4f}")
print("Baseline Exit distribution:")
baseline_exit_dist = pd.Series([class_names[p] for p in baseline_exit_preds]).value_counts(normalize=True)
for cls in class_names:
    print(f"  {cls}: {baseline_exit_dist.get(cls, 0)*100:.1f}%")

# Optimize Exit thresholds
bounds_exit = [(-0.3, 0.3) for _ in range(4)]
result_exit = differential_evolution(
    objective_exit,
    bounds_exit,
    seed=42,
    maxiter=100,
    popsize=15,
    tol=0.001,
    workers=1
)

optimal_thresholds_exit = result_exit.x
print(f"\nOptimal Exit thresholds: {optimal_thresholds_exit}")

# Evaluate optimized Exit
optimized_exit_preds = apply_thresholds(oof_exit_probs, optimal_thresholds_exit)
optimized_exit_f1 = f1_score(y_exit_true, optimized_exit_preds, average='macro')
print(f"Optimized Exit F1: {optimized_exit_f1:.4f} (Δ={optimized_exit_f1-baseline_exit_f1:+.4f})")
print("Optimized Exit distribution:")
optimized_exit_dist = pd.Series([class_names[p] for p in optimized_exit_preds]).value_counts(normalize=True)
for cls in class_names:
    print(f"  {cls}: {optimized_exit_dist.get(cls, 0)*100:.1f}%")

print("\n" + "="*70)
print("PHASE 3: Combined OOF Performance")
print("="*70)

# Combined metrics
baseline_combined_f1 = (baseline_enter_f1 + baseline_exit_f1) / 2
optimized_combined_f1 = (optimized_enter_f1 + optimized_exit_f1) / 2
print(f"Baseline Combined F1: {baseline_combined_f1:.4f}")
print(f"Optimized Combined F1: {optimized_combined_f1:.4f} (Δ={optimized_combined_f1-baseline_combined_f1:+.4f})")

# Generate submission with optimized thresholds
print("\n" + "="*70)
print("PHASE 4: Generating Submission with Optimized Thresholds")
print("="*70)

# Load full GBM models
print("Loading GBM models...")
try:
    with open('gbm_enter_xgb.pkl', 'rb') as f:
        enter_xgb = pickle.load(f)
    with open('gbm_enter_lgb.pkl', 'rb') as f:
        enter_lgb = pickle.load(f)
    with open('gbm_exit_xgb.pkl', 'rb') as f:
        exit_xgb = pickle.load(f)
    with open('gbm_exit_lgb.pkl', 'rb') as f:
        exit_lgb = pickle.load(f)
except Exception as e:
    print(f"Error loading models: {e}")
    print("Models may be corrupted. Please re-run dev12_3_full_gbm_submission.py to regenerate.")
    import sys
    sys.exit(1)

with open('gbm_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    feature_cols = artifacts['feature_cols']
    location_encoder = artifacts['location_encoder']

# Prepare test features
test_input['Location_encoded'] = location_encoder.transform(test_input['Location'])
test_input['DateTime'] = pd.to_datetime(test_input['DateTime'])
test_input['hour'] = test_input['DateTime'].dt.hour
test_input['minute'] = test_input['DateTime'].dt.minute
test_input['day_of_week'] = test_input['DateTime'].dt.dayofweek
test_input['is_weekend'] = (test_input['day_of_week'] >= 5).astype(int)

# Cyclical encodings
test_input['hour_sin'] = np.sin(2 * np.pi * test_input['hour'] / 24)
test_input['hour_cos'] = np.cos(2 * np.pi * test_input['hour'] / 24)
test_input['minute_sin'] = np.sin(2 * np.pi * test_input['minute'] / 60)
test_input['minute_cos'] = np.cos(2 * np.pi * test_input['minute'] / 60)
test_input['day_sin'] = np.sin(2 * np.pi * test_input['day_of_week'] / 7)
test_input['day_cos'] = np.cos(2 * np.pi * test_input['day_of_week'] / 7)

# Polynomial features
test_input['hour_squared'] = test_input['hour'] ** 2
test_input['location_squared'] = test_input['Location_encoded'] ** 2

# Interactions
test_input['is_rush_hour'] = ((test_input['hour'] >= 7) & (test_input['hour'] <= 9) | 
                                (test_input['hour'] >= 16) & (test_input['hour'] <= 18)).astype(int)
test_input['rush_location'] = test_input['is_rush_hour'] * test_input['Location_encoded']
test_input['hour_location'] = test_input['hour'] * test_input['Location_encoded']
test_input['weekend_location'] = test_input['is_weekend'] * test_input['Location_encoded']

X_test = test_input[feature_cols]

# Predict with both models
print("Predicting with XGBoost and LightGBM...")
enter_xgb_probs = enter_xgb.predict_proba(X_test)
enter_lgb_probs = enter_lgb.predict_proba(X_test)
exit_xgb_probs = exit_xgb.predict_proba(X_test)
exit_lgb_probs = exit_lgb.predict_proba(X_test)

# Blend probabilities
enter_probs = (enter_xgb_probs + enter_lgb_probs) / 2
exit_probs = (exit_xgb_probs + exit_lgb_probs) / 2

# Apply optimized thresholds
enter_preds_idx = apply_thresholds(enter_probs, optimal_thresholds_enter)
exit_preds_idx = apply_thresholds(exit_probs, optimal_thresholds_exit)

# Convert to class names
enter_preds = [class_names[i] for i in enter_preds_idx]
exit_preds = [class_names[i] for i in exit_preds_idx]

# Create submission
submission = pd.DataFrame({
    'SegmentID': test_input['SegmentID'],
    'EnterTrafficCondition': enter_preds,
    'ExitTrafficCondition': exit_preds
})

submission.to_csv('submission_threshold_optimized.csv', index=False)
print("[OK] Saved submission_threshold_optimized.csv")

# Distribution stats
enter_dist = pd.Series(enter_preds).value_counts(normalize=True)
exit_dist = pd.Series(exit_preds).value_counts(normalize=True)
overall_dist = pd.concat([pd.Series(enter_preds), pd.Series(exit_preds)]).value_counts(normalize=True)

print("\n[STATS] Optimized submission distribution:")
print("Enter:")
for cls in class_names:
    print(f"  {cls}: {enter_dist.get(cls, 0)*100:.1f}%")
print("Exit:")
for cls in class_names:
    print(f"  {cls}: {exit_dist.get(cls, 0)*100:.1f}%")
print("Overall:")
for cls in class_names:
    print(f"  {cls}: {overall_dist.get(cls, 0)*100:.1f}%")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"OOF Performance Improvement:")
print(f"  Enter: {baseline_enter_f1:.4f} → {optimized_enter_f1:.4f} (Δ={optimized_enter_f1-baseline_enter_f1:+.4f})")
print(f"  Exit: {baseline_exit_f1:.4f} → {optimized_exit_f1:.4f} (Δ={optimized_exit_f1-baseline_exit_f1:+.4f})")
print(f"  Combined: {baseline_combined_f1:.4f} → {optimized_combined_f1:.4f} (Δ={optimized_combined_f1-baseline_combined_f1:+.4f})")
print(f"\nExpected leaderboard improvement: +{(optimized_combined_f1-baseline_combined_f1)*100:.2f}% macro F1")
print("\n[RECOMMENDATION] Submit submission_threshold_optimized.csv")
