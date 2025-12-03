"""
Dev 15.1: Advanced Feature Engineering + Hyperparameter Tuning
===============================================================
Create richer features and fine-tune XGBoost/LightGBM
to maximize F1 score targeting 0.8013
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DEV 15.1: ADVANCED FEATURES + HYPERPARAMETER TUNING")
print("="*80)

# Load data
train = pd.read_csv('Train.csv')
test_input = pd.read_csv('TestInputSegments.csv')

print(f"\n[DATA] Train: {len(train)} samples, Test: {len(test_input)} samples")

# Advanced Feature Engineering
def create_advanced_features(df, is_train=True):
    """Create comprehensive feature set"""
    
    # Location encoding
    le_location = LabelEncoder()
    if is_train:
        df['Location_encoded'] = le_location.fit_transform(df['view_label'])
    else:
        df['Location_encoded'] = le_location.transform(df['view_label'])
    
    # Time features
    df['DateTime'] = pd.to_datetime(df['datetimestamp_start'])
    df['hour'] = df['DateTime'].dt.hour
    df['minute'] = df['DateTime'].dt.minute
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['day_of_month'] = df['DateTime'].dt.day
    df['week_of_year'] = df['DateTime'].dt.isocalendar().week
    
    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Categorical time bins
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 18)).astype(int)
    df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_midday'] = ((df['hour'] >= 11) & (df['hour'] <= 14)).astype(int)
    
    # Polynomial features
    df['hour_squared'] = df['hour'] ** 2
    df['hour_cubed'] = df['hour'] ** 3
    df['location_squared'] = df['Location_encoded'] ** 2
    df['minute_squared'] = df['minute'] ** 2
    
    # Interactions
    df['rush_location'] = df['is_rush_hour'] * df['Location_encoded']
    df['hour_location'] = df['hour'] * df['Location_encoded']
    df['weekend_location'] = df['is_weekend'] * df['Location_encoded']
    df['morning_rush_location'] = df['is_morning_rush'] * df['Location_encoded']
    df['evening_rush_location'] = df['is_evening_rush'] * df['Location_encoded']
    df['hour_weekend'] = df['hour'] * df['is_weekend']
    df['location_day'] = df['Location_encoded'] * df['day_of_week']
    
    # Traffic flow patterns (location-specific time buckets)
    df['location_hour_bucket'] = df['Location_encoded'].astype(str) + '_' + (df['hour'] // 3).astype(str)
    le_bucket = LabelEncoder()
    df['location_hour_bucket'] = le_bucket.fit_transform(df['location_hour_bucket'])
    
    return df, le_location, le_bucket

print("\n[ENGINEER] Creating advanced features...")
train_fe, le_location, le_bucket = create_advanced_features(train.copy(), is_train=True)

# Use the same encoders for test
test_fe = test_input.copy()
test_fe['Location_encoded'] = le_location.transform(test_fe['view_label'])
test_fe['DateTime'] = pd.to_datetime(test_fe['datetimestamp_start'])
test_fe['hour'] = test_fe['DateTime'].dt.hour
test_fe['minute'] = test_fe['DateTime'].dt.minute
test_fe['day_of_week'] = test_fe['DateTime'].dt.dayofweek
test_fe['day_of_month'] = test_fe['DateTime'].dt.day
test_fe['week_of_year'] = test_fe['DateTime'].dt.isocalendar().week

test_fe['hour_sin'] = np.sin(2 * np.pi * test_fe['hour'] / 24)
test_fe['hour_cos'] = np.cos(2 * np.pi * test_fe['hour'] / 24)
test_fe['minute_sin'] = np.sin(2 * np.pi * test_fe['minute'] / 60)
test_fe['minute_cos'] = np.cos(2 * np.pi * test_fe['minute'] / 60)
test_fe['day_sin'] = np.sin(2 * np.pi * test_fe['day_of_week'] / 7)
test_fe['day_cos'] = np.cos(2 * np.pi * test_fe['day_of_week'] / 7)

test_fe['is_weekend'] = (test_fe['day_of_week'] >= 5).astype(int)
test_fe['is_morning_rush'] = ((test_fe['hour'] >= 7) & (test_fe['hour'] <= 9)).astype(int)
test_fe['is_evening_rush'] = ((test_fe['hour'] >= 16) & (test_fe['hour'] <= 18)).astype(int)
test_fe['is_rush_hour'] = (test_fe['is_morning_rush'] | test_fe['is_evening_rush']).astype(int)
test_fe['is_night'] = ((test_fe['hour'] >= 22) | (test_fe['hour'] <= 5)).astype(int)
test_fe['is_midday'] = ((test_fe['hour'] >= 11) & (test_fe['hour'] <= 14)).astype(int)

test_fe['hour_squared'] = test_fe['hour'] ** 2
test_fe['hour_cubed'] = test_fe['hour'] ** 3
test_fe['location_squared'] = test_fe['Location_encoded'] ** 2
test_fe['minute_squared'] = test_fe['minute'] ** 2

test_fe['rush_location'] = test_fe['is_rush_hour'] * test_fe['Location_encoded']
test_fe['hour_location'] = test_fe['hour'] * test_fe['Location_encoded']
test_fe['weekend_location'] = test_fe['is_weekend'] * test_fe['Location_encoded']
test_fe['morning_rush_location'] = test_fe['is_morning_rush'] * test_fe['Location_encoded']
test_fe['evening_rush_location'] = test_fe['is_evening_rush'] * test_fe['Location_encoded']
test_fe['hour_weekend'] = test_fe['hour'] * test_fe['is_weekend']
test_fe['location_day'] = test_fe['Location_encoded'] * test_fe['day_of_week']

test_fe['location_hour_bucket'] = test_fe['Location_encoded'].astype(str) + '_' + (test_fe['hour'] // 3).astype(str)
test_fe['location_hour_bucket'] = le_bucket.transform(test_fe['location_hour_bucket'])

# Feature list
feature_cols = [
    'Location_encoded', 'hour', 'minute', 'day_of_week', 'day_of_month', 'week_of_year',
    'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
    'is_weekend', 'is_morning_rush', 'is_evening_rush', 'is_rush_hour', 'is_night', 'is_midday',
    'hour_squared', 'hour_cubed', 'location_squared', 'minute_squared',
    'rush_location', 'hour_location', 'weekend_location',
    'morning_rush_location', 'evening_rush_location', 'hour_weekend', 'location_day',
    'location_hour_bucket'
]

print(f"[FEATURES] Total features: {len(feature_cols)}")

X_train = train_fe[feature_cols]

# Label encode targets
le_target = LabelEncoder()
y_enter = le_target.fit_transform(train_fe['congestion_enter_rating'])
y_exit = le_target.transform(train_fe['congestion_exit_rating'])
class_names = le_target.classes_

X_test = test_fe[feature_cols]

# Hyperparameter-tuned models
print("\n[TUNE] Training optimized XGBoost + LightGBM models...")

# XGBoost with fine-tuned parameters
xgb_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 2.0,
    'min_child_weight': 3,
    'gamma': 0.1,
    'random_state': 42,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss'
}

# LightGBM with fine-tuned parameters
lgb_params = {
    'n_estimators': 500,
    'max_depth': 10,
    'learning_rate': 0.02,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'reg_alpha': 0.3,
    'reg_lambda': 2.0,
    'min_child_samples': 20,
    'random_state': 42,
    'verbose': -1
}

# 5-Fold CV for robust evaluation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- ENTER MODELS ---")
enter_xgb_oof = np.zeros(len(y_enter))
enter_lgb_oof = np.zeros(len(y_enter))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_enter)):
    print(f"\nFold {fold+1}/5:")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_enter[train_idx], y_enter[val_idx]
    
    # XGBoost
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr)
    enter_xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_f1 = f1_score(y_val, enter_xgb_oof[val_idx], average='macro')
    print(f"  XGB F1: {xgb_f1:.4f}")
    
    # LightGBM
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr)
    enter_lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_f1 = f1_score(y_val, enter_lgb_oof[val_idx], average='macro')
    print(f"  LGB F1: {lgb_f1:.4f}")

enter_xgb_f1 = f1_score(y_enter, enter_xgb_oof, average='macro')
enter_lgb_f1 = f1_score(y_enter, enter_lgb_oof, average='macro')
print(f"\n[ENTER] XGB OOF F1: {enter_xgb_f1:.4f}, LGB OOF F1: {enter_lgb_f1:.4f}")

print("\n--- EXIT MODELS ---")
exit_xgb_oof = np.zeros(len(y_exit))
exit_lgb_oof = np.zeros(len(y_exit))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_exit)):
    print(f"\nFold {fold+1}/5:")
    
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_exit[train_idx], y_exit[val_idx]
    
    # XGBoost
    xgb_model = XGBClassifier(**xgb_params)
    xgb_model.fit(X_tr, y_tr)
    exit_xgb_oof[val_idx] = xgb_model.predict(X_val)
    xgb_f1 = f1_score(y_val, exit_xgb_oof[val_idx], average='macro')
    print(f"  XGB F1: {xgb_f1:.4f}")
    
    # LightGBM
    lgb_model = LGBMClassifier(**lgb_params)
    lgb_model.fit(X_tr, y_tr)
    exit_lgb_oof[val_idx] = lgb_model.predict(X_val)
    lgb_f1 = f1_score(y_val, exit_lgb_oof[val_idx], average='macro')
    print(f"  LGB F1: {lgb_f1:.4f}")

exit_xgb_f1 = f1_score(y_exit, exit_xgb_oof, average='macro')
exit_lgb_f1 = f1_score(y_exit, exit_lgb_oof, average='macro')
print(f"\n[EXIT] XGB OOF F1: {exit_xgb_f1:.4f}, LGB OOF F1: {exit_lgb_f1:.4f}")

# Combined
combined_xgb_f1 = (enter_xgb_f1 + exit_xgb_f1) / 2
combined_lgb_f1 = (enter_lgb_f1 + exit_lgb_f1) / 2
print(f"\n[COMBINED] XGB F1: {combined_xgb_f1:.4f}, LGB F1: {combined_lgb_f1:.4f}")

# Train final models on full data
print("\n[FINAL] Training full models...")
final_enter_xgb = XGBClassifier(**xgb_params)
final_enter_xgb.fit(X_train, y_enter)

final_enter_lgb = LGBMClassifier(**lgb_params)
final_enter_lgb.fit(X_train, y_enter)

final_exit_xgb = XGBClassifier(**xgb_params)
final_exit_xgb.fit(X_train, y_exit)

final_exit_lgb = LGBMClassifier(**lgb_params)
final_exit_lgb.fit(X_train, y_exit)

# Predict and blend
print("\n[PREDICT] Generating submission...")
enter_xgb_pred = final_enter_xgb.predict(X_test)
enter_lgb_pred = final_enter_lgb.predict(X_test)
exit_xgb_pred = final_exit_xgb.predict(X_test)
exit_lgb_pred = final_exit_lgb.predict(X_test)

# Weighted blend (XGB slightly better)
enter_preds_idx = []
exit_preds_idx = []

for i in range(len(X_test)):
    # Weighted vote
    if enter_xgb_pred[i] == enter_lgb_pred[i]:
        enter_preds_idx.append(enter_xgb_pred[i])
    else:
        # XGB gets 60% weight
        enter_preds_idx.append(enter_xgb_pred[i] if np.random.random() < 0.6 else enter_lgb_pred[i])
    
    if exit_xgb_pred[i] == exit_lgb_pred[i]:
        exit_preds_idx.append(exit_xgb_pred[i])
    else:
        exit_preds_idx.append(exit_xgb_pred[i] if np.random.random() < 0.6 else exit_lgb_pred[i])

# Convert back to class names
enter_preds = le_target.inverse_transform(enter_preds_idx)
exit_preds = le_target.inverse_transform(exit_preds_idx)

# Create submission
enter_ids = test_input['ID_enter'].tolist()
exit_ids = test_input['ID_exit'].tolist()

submission = pd.DataFrame({
    'ID': enter_ids + exit_ids,
    'Target': list(enter_preds) + list(exit_preds),
    'Target_Accuracy': list(enter_preds) + list(exit_preds)
})

submission.to_csv('submission_advanced_features.csv', index=False)
print("[OK] Saved submission_advanced_features.csv")

# Distribution
dist = submission['Target'].value_counts(normalize=True)
print("\n[STATS] Distribution:")
for cls in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
    print(f"  {cls}: {dist.get(cls, 0)*100:.1f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Advanced feature engineering: {len(feature_cols)} features")
print(f"✓ Hyperparameter-tuned XGBoost + LightGBM")
print(f"✓ 5-fold CV OOF: Enter={max(enter_xgb_f1, enter_lgb_f1):.4f}, Exit={max(exit_xgb_f1, exit_lgb_f1):.4f}")
print(f"✓ Combined best OOF F1: {max(combined_xgb_f1, combined_lgb_f1):.4f}")
print(f"\n[TARGET] Hedef: 0.8013, Mevcut En İyi: 0.7708, Gap: +{(0.8013-0.7708)*100:.2f}%")
print(f"[RECOMMENDATION] submission_advanced_features.csv'i dene!")
