"""
Development 12: Stratified K-Fold Validation Framework
Build robust local validation aligned with competition metrics (macro F1 + accuracy).
Generate OOF predictions to estimate leaderboard performance.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

print('[DEV 12] STRATIFIED K-FOLD VALIDATION FRAMEWORK')
print('='*80)

# Labels
LABELS = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
label_to_idx = {l:i for i,l in enumerate(LABELS)}

# Load training data
train_df = pd.read_csv('Train.csv')
train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
train_df['hour'] = train_df['datetime'].dt.hour
train_df['minute'] = train_df['datetime'].dt.minute
train_df['day_of_week'] = train_df['datetime'].dt.dayofweek
train_df['day_of_month'] = train_df['datetime'].dt.day
train_df['month'] = train_df['datetime'].dt.month

# Encode location
unique_locations = sorted(train_df['view_label'].unique())
location_encoder = {loc: idx for idx, loc in enumerate(unique_locations)}
train_df['location_encoded'] = train_df['view_label'].map(location_encoder)

# Feature engineering
def prepare_features(df):
    df = df.copy()
    df['is_rush_hour'] = df['hour'].apply(lambda h: 1 if h in [7,8,9,16,17,18] else 0)
    df['is_weekend'] = df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    df['is_morning'] = df['hour'].apply(lambda h: 1 if h < 12 else 0)
    df['is_evening'] = df['hour'].apply(lambda h: 1 if 17 <= h < 21 else 0)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['signal_encoded'] = 0  # placeholder (not used in current models)
    df['rush_x_location'] = df['is_rush_hour'] * df['location_encoded']
    df['hour_x_weekend'] = df['hour'] * df['is_weekend']
    return df

train_df = prepare_features(train_df)

feature_cols = [
    'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
    'is_rush_hour', 'is_weekend', 'is_morning', 'is_evening',
    'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    'location_encoded', 'signal_encoded',
    'rush_x_location', 'hour_x_weekend'
]

# Prepare datasets for enter and exit
def build_dataset(df, target_col):
    X = df[feature_cols].values
    y = df[target_col].map(label_to_idx).values
    return X, y

print('[PREPARE] Building enter/exit datasets...')
X_enter, y_enter = build_dataset(train_df, 'congestion_enter_rating')
X_exit, y_exit = build_dataset(train_df, 'congestion_exit_rating')

# K-Fold validation
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Validation function
def validate_model(X, y, rating_type):
    print(f'\n[VALIDATE] {rating_type.upper()} - {N_SPLITS}-Fold CV')
    
    oof_preds = np.zeros(len(y), dtype=int)
    oof_probs = np.zeros((len(y), len(LABELS)))
    
    fold_f1 = []
    fold_acc = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Compute class weights
        class_counts = np.bincount(y_train, minlength=len(LABELS))
        total = class_counts.sum()
        class_weights = {i: total / (len(LABELS) * max(class_counts[i], 1)) for i in range(len(LABELS))}
        
        # Train RandomForest with class weights
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight=class_weights,
            random_state=42 + fold_idx,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Predict on validation
        val_preds = model.predict(X_val)
        val_probs = model.predict_proba(X_val)
        
        oof_preds[val_idx] = val_preds
        oof_probs[val_idx] = val_probs
        
        # Compute metrics
        f1_macro = f1_score(y_val, val_preds, average='macro', zero_division=0)
        acc = accuracy_score(y_val, val_preds)
        
        fold_f1.append(f1_macro)
        fold_acc.append(acc)
        
        print(f'  Fold {fold_idx}: F1={f1_macro:.4f}, Acc={acc:.4f}')
    
    # Overall OOF metrics
    oof_f1 = f1_score(y, oof_preds, average='macro', zero_division=0)
    oof_acc = accuracy_score(y, oof_preds)
    
    print(f'  OOF: F1={oof_f1:.4f}, Acc={oof_acc:.4f}')
    print(f'  Avg Fold: F1={np.mean(fold_f1):.4f}±{np.std(fold_f1):.4f}, Acc={np.mean(fold_acc):.4f}±{np.std(fold_acc):.4f}')
    
    # Class distribution
    pred_labels = [LABELS[i] for i in oof_preds]
    true_labels = [LABELS[i] for i in y]
    
    print(f'\n  True distribution:')
    true_dist = pd.Series(true_labels).value_counts(normalize=True) * 100
    for l in LABELS:
        print(f'    {l}: {true_dist.get(l,0):.1f}%')
    
    print(f'\n  Predicted distribution:')
    pred_dist = pd.Series(pred_labels).value_counts(normalize=True) * 100
    for l in LABELS:
        print(f'    {l}: {pred_dist.get(l,0):.1f}%')
    
    # Per-class F1
    print(f'\n  Per-class F1:')
    report = classification_report(y, oof_preds, target_names=LABELS, zero_division=0, output_dict=True)
    for l in LABELS:
        print(f'    {l}: {report[l]["f1-score"]:.4f}')
    
    return {
        'oof_preds': oof_preds,
        'oof_probs': oof_probs,
        'f1_macro': oof_f1,
        'accuracy': oof_acc,
        'fold_f1': fold_f1,
        'fold_acc': fold_acc,
        'report': report
    }

# Run validation
enter_results = validate_model(X_enter, y_enter, 'enter')
exit_results = validate_model(X_exit, y_exit, 'exit')

# Combined metrics (average enter + exit)
combined_f1 = (enter_results['f1_macro'] + exit_results['f1_macro']) / 2
combined_acc = (enter_results['accuracy'] + exit_results['accuracy']) / 2

print(f'\n[COMBINED METRICS]')
print(f'  Macro F1 (avg): {combined_f1:.4f}')
print(f'  Accuracy (avg): {combined_acc:.4f}')

# Save OOF predictions
oof_data = {
    'enter_preds': enter_results['oof_preds'],
    'enter_probs': enter_results['oof_probs'],
    'exit_preds': exit_results['oof_preds'],
    'exit_probs': exit_results['oof_probs'],
    'y_enter': y_enter,
    'y_exit': y_exit
}

with open('oof_predictions.pkl', 'wb') as f:
    pickle.dump(oof_data, f)

print(f'\n[OK] OOF predictions saved to oof_predictions.pkl')

# Summary comparison with target
print(f'\n[SUMMARY]')
print(f'  Current leaderboard: F1=0.6675, Acc=0.7682')
print(f'  Target leaderboard:  F1>0.80, Acc>0.80 (to beat 0.801299329)')
print(f'  Local OOF estimate:  F1={combined_f1:.4f}, Acc={combined_acc:.4f}')
if combined_f1 < 0.67:
    print(f'  Status: Below current; needs improvement')
elif combined_f1 < 0.80:
    print(f'  Status: Improvement over current; still below target')
else:
    print(f'  Status: At or above target!')
