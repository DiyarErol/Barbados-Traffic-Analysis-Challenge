"""
Dev 15: Meta-Level Stacking with Multiple Base Models
======================================================
Train 2nd-level meta-learner on OOF predictions from:
- RandomForest (baseline)
- XGBoost (strong gradient boosting)
- LightGBM (fast gradient boosting)

Expected improvement: +0.03-0.06 F1 based on stacking literature
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEV 15: META-LEVEL STACKING")
print("="*80)

# Load OOF predictions from all models
print("\n[LOAD] Loading OOF predictions from base models...")
with open('oof_predictions_gbm.pkl', 'rb') as f:
    oof_gbm = pickle.load(f)

# OOF probabilities from base models
enter_xgb_probs = oof_gbm['enter_xgb_probs']  # (n, 4)
enter_lgb_probs = oof_gbm['enter_lgb_probs']  # (n, 4)
exit_xgb_probs = oof_gbm['exit_xgb_probs']
exit_lgb_probs = oof_gbm['exit_lgb_probs']

y_enter = oof_gbm['y_enter']
y_exit = oof_gbm['y_exit']

print(f"  Enter OOF: {enter_xgb_probs.shape[0]} samples")
print(f"  Exit OOF: {exit_xgb_probs.shape[0]} samples")

# Load original train data for additional meta-features
train = pd.read_csv('Train.csv')

# Prepare meta-features: combine base model predictions + original features
print("\n[PREPARE] Creating meta-features...")

# Original features
le_location = LabelEncoder()
le_location.fit(train['view_label'].unique())
train['Location_encoded'] = le_location.transform(train['view_label'])
train['DateTime'] = pd.to_datetime(train['datetimestamp_start'])
train['hour'] = train['DateTime'].dt.hour
train['minute'] = train['DateTime'].dt.minute
train['day_of_week'] = train['DateTime'].dt.dayofweek
train['is_weekend'] = (train['day_of_week'] >= 5).astype(int)
train['is_rush_hour'] = ((train['hour'] >= 7) & (train['hour'] <= 9) | 
                          (train['hour'] >= 16) & (train['hour'] <= 18)).astype(int)

# Meta-features for Enter
meta_features_enter = np.hstack([
    enter_xgb_probs,  # 4 probs from XGBoost
    enter_lgb_probs,  # 4 probs from LightGBM
    train[['Location_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].values  # 5 original
])
print(f"  Enter meta-features shape: {meta_features_enter.shape}")

# Meta-features for Exit
meta_features_exit = np.hstack([
    exit_xgb_probs,
    exit_lgb_probs,
    train[['Location_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].values
])
print(f"  Exit meta-features shape: {meta_features_exit.shape}")

# Train meta-learner with cross-validation
print("\n[TRAIN] Training meta-learners with 5-fold CV...")

class_names = ['free flowing', 'heavy delay', 'light delay', 'moderate delay']

# Enter meta-learner
print("\n--- ENTER META-LEARNER ---")
meta_enter = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

# OOF predictions for meta-learner
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
enter_meta_oof = np.zeros(len(y_enter), dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(meta_features_enter, y_enter)):
    X_train_meta = meta_features_enter[train_idx]
    X_val_meta = meta_features_enter[val_idx]
    y_train_meta = y_enter[train_idx] if isinstance(y_enter, np.ndarray) else y_enter.iloc[train_idx]
    y_val_meta = y_enter[val_idx] if isinstance(y_enter, np.ndarray) else y_enter.iloc[val_idx]
    
    meta_enter_fold = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    meta_enter_fold.fit(X_train_meta, y_train_meta)
    enter_meta_oof[val_idx] = meta_enter_fold.predict(X_val_meta)
    
    fold_f1 = f1_score(y_val_meta, enter_meta_oof[val_idx], average='macro')
    print(f"  Fold {fold+1} F1: {fold_f1:.4f}")

enter_meta_f1 = f1_score(y_enter, enter_meta_oof, average='macro')
enter_meta_acc = accuracy_score(y_enter, enter_meta_oof)
print(f"\n[ENTER META] OOF F1: {enter_meta_f1:.4f}, Acc: {enter_meta_acc:.4f}")

# Baseline comparison
enter_baseline_preds = np.argmax((enter_xgb_probs + enter_lgb_probs) / 2, axis=1)
enter_baseline_f1 = f1_score(y_enter, enter_baseline_preds, average='macro')
print(f"[BASELINE] Enter F1: {enter_baseline_f1:.4f}")
print(f"[IMPROVEMENT] Δ = {enter_meta_f1 - enter_baseline_f1:+.4f} ({(enter_meta_f1/enter_baseline_f1-1)*100:+.2f}%)")

# Exit meta-learner
print("\n--- EXIT META-LEARNER ---")
exit_meta_oof = np.zeros(len(y_exit), dtype=int)

for fold, (train_idx, val_idx) in enumerate(skf.split(meta_features_exit, y_exit)):
    X_train_meta = meta_features_exit[train_idx]
    X_val_meta = meta_features_exit[val_idx]
    y_train_meta = y_exit[train_idx] if isinstance(y_exit, np.ndarray) else y_exit.iloc[train_idx]
    y_val_meta = y_exit[val_idx] if isinstance(y_exit, np.ndarray) else y_exit.iloc[val_idx]
    
    meta_exit_fold = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
        random_state=42, verbose=-1
    )
    meta_exit_fold.fit(X_train_meta, y_train_meta)
    exit_meta_oof[val_idx] = meta_exit_fold.predict(X_val_meta)
    
    fold_f1 = f1_score(y_val_meta, exit_meta_oof[val_idx], average='macro')
    print(f"  Fold {fold+1} F1: {fold_f1:.4f}")

exit_meta_f1 = f1_score(y_exit, exit_meta_oof, average='macro')
exit_meta_acc = accuracy_score(y_exit, exit_meta_oof)
print(f"\n[EXIT META] OOF F1: {exit_meta_f1:.4f}, Acc: {exit_meta_acc:.4f}")

exit_baseline_preds = np.argmax((exit_xgb_probs + exit_lgb_probs) / 2, axis=1)
exit_baseline_f1 = f1_score(y_exit, exit_baseline_preds, average='macro')
print(f"[BASELINE] Exit F1: {exit_baseline_f1:.4f}")
print(f"[IMPROVEMENT] Δ = {exit_meta_f1 - exit_baseline_f1:+.4f} ({(exit_meta_f1/exit_baseline_f1-1)*100:+.2f}%)")

# Combined metrics
combined_meta_f1 = (enter_meta_f1 + exit_meta_f1) / 2
combined_baseline_f1 = (enter_baseline_f1 + exit_baseline_f1) / 2
print(f"\n[COMBINED] Meta F1: {combined_meta_f1:.4f}, Baseline F1: {combined_baseline_f1:.4f}")
print(f"[TOTAL IMPROVEMENT] Δ = {combined_meta_f1 - combined_baseline_f1:+.4f} ({(combined_meta_f1/combined_baseline_f1-1)*100:+.2f}%)")

# Train final meta-learners on full data
print("\n[FINAL] Training full meta-learners...")
meta_enter.fit(meta_features_enter, y_enter)
meta_exit = LGBMClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
    random_state=42, verbose=-1
)
meta_exit.fit(meta_features_exit, y_exit)

# Generate submission with meta-learner
print("\n[PREDICT] Generating meta-stacking submission...")

# Load test data and base model predictions
test_input = pd.read_csv('TestInputSegments.csv')
test_input['Location_encoded'] = le_location.transform(test_input['view_label'])
test_input['DateTime'] = pd.to_datetime(test_input['datetimestamp_start'])
test_input['hour'] = test_input['DateTime'].dt.hour
test_input['minute'] = test_input['DateTime'].dt.minute
test_input['day_of_week'] = test_input['DateTime'].dt.dayofweek
test_input['is_weekend'] = (test_input['day_of_week'] >= 5).astype(int)
test_input['is_rush_hour'] = ((test_input['hour'] >= 7) & (test_input['hour'] <= 9) | 
                               (test_input['hour'] >= 16) & (test_input['hour'] <= 18)).astype(int)

# Get base model predictions on test
with open('gbm_enter_xgb.pkl', 'rb') as f:
    enter_xgb = pickle.load(f)
with open('gbm_enter_lgb.pkl', 'rb') as f:
    enter_lgb = pickle.load(f)
with open('gbm_exit_xgb.pkl', 'rb') as f:
    exit_xgb = pickle.load(f)
with open('gbm_exit_lgb.pkl', 'rb') as f:
    exit_lgb = pickle.load(f)

with open('gbm_artifacts.pkl', 'rb') as f:
    artifacts = pickle.load(f)
    feature_cols = artifacts['feature_cols']

# Prepare test features
test_input['hour_sin'] = np.sin(2 * np.pi * test_input['hour'] / 24)
test_input['hour_cos'] = np.cos(2 * np.pi * test_input['hour'] / 24)
test_input['minute_sin'] = np.sin(2 * np.pi * test_input['minute'] / 60)
test_input['minute_cos'] = np.cos(2 * np.pi * test_input['minute'] / 60)
test_input['day_sin'] = np.sin(2 * np.pi * test_input['day_of_week'] / 7)
test_input['day_cos'] = np.cos(2 * np.pi * test_input['day_of_week'] / 7)
test_input['hour_squared'] = test_input['hour'] ** 2
test_input['location_squared'] = test_input['Location_encoded'] ** 2
test_input['rush_location'] = test_input['is_rush_hour'] * test_input['Location_encoded']
test_input['hour_location'] = test_input['hour'] * test_input['Location_encoded']
test_input['weekend_location'] = test_input['is_weekend'] * test_input['Location_encoded']

X_test = test_input[feature_cols]

# Base model predictions
test_enter_xgb_probs = enter_xgb.predict_proba(X_test)
test_enter_lgb_probs = enter_lgb.predict_proba(X_test)
test_exit_xgb_probs = exit_xgb.predict_proba(X_test)
test_exit_lgb_probs = exit_lgb.predict_proba(X_test)

# Create meta-features for test
meta_test_enter = np.hstack([
    test_enter_xgb_probs,
    test_enter_lgb_probs,
    test_input[['Location_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].values
])

meta_test_exit = np.hstack([
    test_exit_xgb_probs,
    test_exit_lgb_probs,
    test_input[['Location_encoded', 'hour', 'day_of_week', 'is_weekend', 'is_rush_hour']].values
])

# Meta-learner predictions
enter_preds_idx = meta_enter.predict(meta_test_enter)
exit_preds_idx = meta_exit.predict(meta_test_exit)

enter_preds = [class_names[i] for i in enter_preds_idx]
exit_preds = [class_names[i] for i in exit_preds_idx]

# Create submission
enter_ids = test_input['ID_enter'].tolist()
exit_ids = test_input['ID_exit'].tolist()

submission = pd.DataFrame({
    'ID': enter_ids + exit_ids,
    'Target': enter_preds + exit_preds,
    'Target_Accuracy': enter_preds + exit_preds
})

submission.to_csv('submission_meta_stacking.csv', index=False)
print("[OK] Saved submission_meta_stacking.csv")

# Distribution
dist = submission['Target'].value_counts(normalize=True)
print("\n[STATS] Meta-stacking distribution:")
for cls in class_names:
    print(f"  {cls}: {dist.get(cls, 0)*100:.1f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Meta-stacking with LightGBM 2nd-level learner")
print(f"✓ OOF Performance: Enter F1={enter_meta_f1:.4f}, Exit F1={exit_meta_f1:.4f}")
print(f"✓ Combined F1={combined_meta_f1:.4f} (Improvement: {(combined_meta_f1/combined_baseline_f1-1)*100:+.2f}%)")
print(f"✓ Expected leaderboard boost: {(combined_meta_f1 - combined_baseline_f1)*100:+.2f}% macro F1")
print(f"\n[TARGET] Hedef: 0.8013, Mevcut En İyi: 0.7708")
print(f"[NEXT] submission_meta_stacking.csv'i dene!")
