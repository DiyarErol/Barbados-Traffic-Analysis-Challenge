"""
Development 12.2: XGBoost + LightGBM OOF Validation
Train stronger gradient boosting models with proper validation.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb

print('[DEV 12.2] XGBOOST + LIGHTGBM OOF VALIDATION')
print('='*80)

LABELS = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
label_to_idx = {l:i for i,l in enumerate(LABELS)}

# Load data
train_df = pd.read_csv('Train.csv')
train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
train_df['hour'] = train_df['datetime'].dt.hour
train_df['minute'] = train_df['datetime'].dt.minute
train_df['day_of_week'] = train_df['datetime'].dt.dayofweek
train_df['day_of_month'] = train_df['datetime'].dt.day
train_df['month'] = train_df['datetime'].dt.month

unique_locations = sorted(train_df['view_label'].unique())
location_encoder = {loc: idx for idx, loc in enumerate(unique_locations)}
train_df['location_encoded'] = train_df['view_label'].map(location_encoder)

def prepare_enhanced_features(df):
    df = df.copy()
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if h in [7,8,9,16,17,18] else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    df['is_morning'] = df['hour'].apply(lambda h: 1 if h < 12 else 0)
    df['is_evening'] = df['hour'].apply(lambda h: 1 if 17 <= h < 21 else 0)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['rush_x_location'] = df['is_rush_hour'] * df['location_encoded']
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    df['hour_x_location'] = df['hour'] * df['location_encoded']
    df['weekend_x_location'] = df['is_weekend'] * df['location_encoded']
    df['hour_squared'] = df['hour'] ** 2
    df['location_squared'] = df['location_encoded'] ** 2
    return df

train_df = prepare_enhanced_features(train_df)

feature_cols = [
    'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
    'is_rush_hour', 'is_weekend', 'is_morning', 'is_evening', 'is_night',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'minute_sin', 'minute_cos',
    'location_encoded',
    'rush_x_location', 'hour_x_weekend', 'hour_x_location', 'weekend_x_location',
    'hour_squared', 'location_squared'
]

def build_dataset(df, target_col):
    X = df[feature_cols].values
    y = df[target_col].map(label_to_idx).values
    return X, y

print('[PREPARE] Building datasets...')
X_enter, y_enter = build_dataset(train_df, 'congestion_enter_rating')
X_exit, y_exit = build_dataset(train_df, 'congestion_exit_rating')

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def validate_xgb_lgb(X, y, rating_type):
    print(f'\n[VALIDATE] {rating_type.upper()}')
    
    oof_preds_xgb = np.zeros(len(y), dtype=int)
    oof_probs_xgb = np.zeros((len(y), len(LABELS)))
    oof_preds_lgb = np.zeros(len(y), dtype=int)
    oof_probs_lgb = np.zeros((len(y), len(LABELS)))
    
    fold_f1_xgb = []
    fold_acc_xgb = []
    fold_f1_lgb = []
    fold_acc_lgb = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Compute class weights
        class_counts = np.bincount(y_train, minlength=len(LABELS))
        total = class_counts.sum()
        sample_weights = np.array([total / (len(LABELS) * max(class_counts[y_train[i]], 1)) 
                                   for i in range(len(y_train))])
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multi:softprob',
            num_class=len(LABELS),
            random_state=42 + fold_idx,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
        
        val_preds_xgb = xgb_model.predict(X_val)
        val_probs_xgb = xgb_model.predict_proba(X_val)
        
        oof_preds_xgb[val_idx] = val_preds_xgb
        oof_probs_xgb[val_idx] = val_probs_xgb
        
        f1_xgb = f1_score(y_val, val_preds_xgb, average='macro', zero_division=0)
        acc_xgb = accuracy_score(y_val, val_preds_xgb)
        fold_f1_xgb.append(f1_xgb)
        fold_acc_xgb.append(acc_xgb)
        
        # LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='multiclass',
            num_class=len(LABELS),
            random_state=42 + fold_idx,
            n_jobs=-1,
            verbose=-1
        )
        lgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        val_preds_lgb = lgb_model.predict(X_val)
        val_probs_lgb = lgb_model.predict_proba(X_val)
        
        oof_preds_lgb[val_idx] = val_preds_lgb
        oof_probs_lgb[val_idx] = val_probs_lgb
        
        f1_lgb = f1_score(y_val, val_preds_lgb, average='macro', zero_division=0)
        acc_lgb = accuracy_score(y_val, val_preds_lgb)
        fold_f1_lgb.append(f1_lgb)
        fold_acc_lgb.append(acc_lgb)
        
        print(f'  Fold {fold_idx}: XGB F1={f1_xgb:.4f} Acc={acc_xgb:.4f} | LGB F1={f1_lgb:.4f} Acc={acc_lgb:.4f}')
    
    # OOF metrics
    oof_f1_xgb = f1_score(y, oof_preds_xgb, average='macro', zero_division=0)
    oof_acc_xgb = accuracy_score(y, oof_preds_xgb)
    oof_f1_lgb = f1_score(y, oof_preds_lgb, average='macro', zero_division=0)
    oof_acc_lgb = accuracy_score(y, oof_preds_lgb)
    
    print(f'  OOF XGB: F1={oof_f1_xgb:.4f}, Acc={oof_acc_xgb:.4f}')
    print(f'  OOF LGB: F1={oof_f1_lgb:.4f}, Acc={oof_acc_lgb:.4f}')
    
    # Blend XGB + LGB (average probabilities)
    oof_probs_blend = (oof_probs_xgb + oof_probs_lgb) / 2
    oof_preds_blend = oof_probs_blend.argmax(axis=1)
    oof_f1_blend = f1_score(y, oof_preds_blend, average='macro', zero_division=0)
    oof_acc_blend = accuracy_score(y, oof_preds_blend)
    
    print(f'  OOF BLEND: F1={oof_f1_blend:.4f}, Acc={oof_acc_blend:.4f}')
    
    return {
        'xgb': {'oof_preds': oof_preds_xgb, 'oof_probs': oof_probs_xgb, 'f1': oof_f1_xgb, 'acc': oof_acc_xgb},
        'lgb': {'oof_preds': oof_preds_lgb, 'oof_probs': oof_probs_lgb, 'f1': oof_f1_lgb, 'acc': oof_acc_lgb},
        'blend': {'oof_preds': oof_preds_blend, 'oof_probs': oof_probs_blend, 'f1': oof_f1_blend, 'acc': oof_acc_blend}
    }

# Run validation
print('[RUN] Enter models...')
enter_results = validate_xgb_lgb(X_enter, y_enter, 'enter')

print('\n[RUN] Exit models...')
exit_results = validate_xgb_lgb(X_exit, y_exit, 'exit')

# Combined blend metrics
combined_f1 = (enter_results['blend']['f1'] + exit_results['blend']['f1']) / 2
combined_acc = (enter_results['blend']['acc'] + exit_results['blend']['acc']) / 2

print(f'\n[COMBINED METRICS - XGB+LGB BLEND]')
print(f'  Macro F1 (avg): {combined_f1:.4f}')
print(f'  Accuracy (avg): {combined_acc:.4f}')

# Save
oof_data_gbm = {
    'enter_xgb_probs': enter_results['xgb']['oof_probs'],
    'enter_lgb_probs': enter_results['lgb']['oof_probs'],
    'enter_blend_probs': enter_results['blend']['oof_probs'],
    'exit_xgb_probs': exit_results['xgb']['oof_probs'],
    'exit_lgb_probs': exit_results['lgb']['oof_probs'],
    'exit_blend_probs': exit_results['blend']['oof_probs'],
    'y_enter': y_enter,
    'y_exit': y_exit
}

with open('oof_predictions_gbm.pkl', 'wb') as f:
    pickle.dump(oof_data_gbm, f)

print(f'\n[OK] GBM OOF saved to oof_predictions_gbm.pkl')

print(f'\n[SUMMARY]')
print(f'  Baseline:       F1=0.4189, Acc=0.7461')
print(f'  Enhanced SMOTE: F1=0.4087, Acc=0.7767')
print(f'  XGB+LGB Blend:  F1={combined_f1:.4f}, Acc={combined_acc:.4f}')
print(f'  Target:         F1>0.80, Acc>0.80')
