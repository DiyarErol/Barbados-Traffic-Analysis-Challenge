"""
Development 12.1: Enhanced Models with SMOTE + Advanced Features
Use SMOTE for minority class oversampling + ordinal encoding + richer features.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

print('[DEV 12.1] ENHANCED MODELS: SMOTE + ADVANCED FEATURES')
print('='*80)

LABELS = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
label_to_idx = {l:i for i,l in enumerate(LABELS)}

# Load and prepare data
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

# Enhanced feature engineering
def prepare_enhanced_features(df):
    df = df.copy()
    # Basic temporal
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if h in [7,8,9,16,17,18] else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    df['is_morning'] = df['hour'].apply(lambda h: 1 if h < 12 else 0)
    df['is_evening'] = df['hour'].apply(lambda h: 1 if 17 <= h < 21 else 0)
    df['is_night'] = df['hour'].apply(lambda h: 1 if h < 6 or h >= 22 else 0)
    
    # Cyclical
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    
    # Interactions
    df['rush_x_location'] = df['is_rush_hour'] * df['location_encoded']
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    df['hour_x_location'] = df['hour'] * df['location_encoded']
    df['weekend_x_location'] = df['is_weekend'] * df['location_encoded']
    
    # Polynomial
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

print('[PREPARE] Building enhanced datasets...')
X_enter, y_enter = build_dataset(train_df, 'congestion_enter_rating')
X_exit, y_exit = build_dataset(train_df, 'congestion_exit_rating')

# Validation with SMOTE
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

def validate_with_smote(X, y, rating_type, use_smote=True):
    print(f'\n[VALIDATE] {rating_type.upper()} - SMOTE={use_smote}')
    
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), len(LABELS)))
    
    fold_f1 = []
    fold_acc = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Apply SMOTE to training data
        if use_smote:
            # Check if minority classes exist
            class_counts = np.bincount(y_train, minlength=len(LABELS))
            min_count = class_counts[class_counts > 0].min() if (class_counts > 0).any() else 1
            
            if min_count >= 2:  # SMOTE needs at least 2 samples
                smote = SMOTE(random_state=42 + fold_idx, k_neighbors=min(5, min_count - 1))
                try:
                    X_train, y_train = smote.fit_resample(X_train, y_train)
                except Exception as e:
                    print(f'    Fold {fold_idx}: SMOTE failed ({e}), using original data')
        
        # Compute class weights for remaining imbalance
        class_counts = np.bincount(y_train, minlength=len(LABELS))
        total = class_counts.sum()
        class_weights = {i: total / (len(LABELS) * max(class_counts[i], 1)) for i in range(len(LABELS))}
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=25,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight=class_weights,
            random_state=42 + fold_idx,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)
        
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        
        f1_macro = f1_score(y_val, val_preds, average='macro', zero_division=0)
        acc = accuracy_score(y_val, val_preds)
        
        fold_f1.append(f1_macro)
        fold_acc.append(acc)
        
        print(f'  Fold {fold_idx}: F1={f1_macro:.4f}, Acc={acc:.4f}')
    
    oof_f1 = f1_score(y, oof_preds, average='macro', zero_division=0)
    oof_acc = accuracy_score(y, oof_preds)
    
    print(f'  OOF: F1={oof_f1:.4f}, Acc={oof_acc:.4f}')
    
    # Distribution
    pred_labels = [LABELS[i] for i in oof_preds]
    print(f'\n  Predicted distribution:')
    pred_dist = pd.Series(pred_labels).value_counts(normalize=True) * 100
    for l in LABELS:
        print(f'    {l}: {pred_dist.get(l,0):.1f}%')
    
    return {
        'oof_preds': oof_preds,
        'oof_probs': oof_probs,
        'f1_macro': oof_f1,
        'accuracy': oof_acc
    }

# Run validation
print('\n[RUN] Enter model with SMOTE...')
enter_results = validate_with_smote(X_enter, y_enter, 'enter', use_smote=True)

print('\n[RUN] Exit model with SMOTE...')
exit_results = validate_with_smote(X_exit, y_exit, 'exit', use_smote=True)

# Combined
combined_f1 = (enter_results['f1_macro'] + exit_results['f1_macro']) / 2
combined_acc = (enter_results['accuracy'] + exit_results['accuracy']) / 2

print(f'\n[COMBINED METRICS - ENHANCED]')
print(f'  Macro F1 (avg): {combined_f1:.4f}')
print(f'  Accuracy (avg): {combined_acc:.4f}')

# Save enhanced OOF
oof_data_enhanced = {
    'enter_preds': enter_results['oof_preds'],
    'enter_probs': enter_results['oof_probs'],
    'exit_preds': exit_results['oof_preds'],
    'exit_probs': exit_results['oof_probs'],
    'y_enter': y_enter,
    'y_exit': y_exit,
    'feature_cols': feature_cols
}

with open('oof_predictions_enhanced.pkl', 'wb') as f:
    pickle.dump(oof_data_enhanced, f)

print(f'\n[OK] Enhanced OOF saved to oof_predictions_enhanced.pkl')

print(f'\n[COMPARISON]')
print(f'  Baseline OOF:  F1=0.4189, Acc=0.7461')
print(f'  Enhanced OOF:  F1={combined_f1:.4f}, Acc={combined_acc:.4f}')
improvement_f1 = combined_f1 - 0.4189
improvement_acc = combined_acc - 0.7461
print(f'  Improvement:   F1={improvement_f1:+.4f}, Acc={improvement_acc:+.4f}')
