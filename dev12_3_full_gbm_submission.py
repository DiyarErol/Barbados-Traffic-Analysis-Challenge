"""
Development 12.3: Train Full XGB+LGB on all data and generate submission
Use best hyperparameters from CV; train on full dataset; generate submission.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import xgboost as xgb
import lightgbm as lgb

print('[DEV 12.3] TRAIN FULL GBM AND GENERATE SUBMISSION')
print('='*80)

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

print('[PREPARE] Building full training datasets...')
X_enter, y_enter = build_dataset(train_df, 'congestion_enter_rating')
X_exit, y_exit = build_dataset(train_df, 'congestion_exit_rating')

# Train models on full data
def train_full_models(X, y, rating_type):
    print(f'\n[TRAIN] {rating_type.upper()} - Full dataset')
    
    # Compute sample weights
    class_counts = np.bincount(y, minlength=len(LABELS))
    total = class_counts.sum()
    sample_weights = np.array([total / (len(LABELS) * max(class_counts[y[i]], 1)) 
                               for i in range(len(y))])
    
    # XGBoost
    print('  Training XGBoost...')
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
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    )
    xgb_model.fit(X, y, sample_weight=sample_weights, verbose=False)
    
    # LightGBM
    print('  Training LightGBM...')
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
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X, y, sample_weight=sample_weights)
    
    return xgb_model, lgb_model

enter_xgb, enter_lgb = train_full_models(X_enter, y_enter, 'enter')
exit_xgb, exit_lgb = train_full_models(X_exit, y_exit, 'exit')

# Save models
print('\n[SAVE] Saving trained models...')
joblib.dump(enter_xgb, 'gbm_enter_xgb.pkl')
joblib.dump(enter_lgb, 'gbm_enter_lgb.pkl')
joblib.dump(exit_xgb, 'gbm_exit_xgb.pkl')
joblib.dump(exit_lgb, 'gbm_exit_lgb.pkl')
joblib.dump({'feature_cols': feature_cols, 'location_encoder': location_encoder}, 'gbm_artifacts.pkl')

print('[OK] Models saved')

# Generate submission
print('\n[GENERATE] Creating submission...')

with open('segment_info.pkl', 'rb') as f:
    segment_info = pickle.load(f)

def interpolate_segment_time(segment_id):
    known_segments = sorted(segment_info.keys())
    if segment_id in segment_info:
        return segment_info[segment_id]
    lower = [s for s in known_segments if s < segment_id]
    if lower:
        seg_lower = max(lower)
        info = segment_info[seg_lower]
        diff = segment_id - seg_lower
        total_minutes = info['hour'] * 60 + info['minute'] + diff
        return {
            'hour': (total_minutes // 60) % 24,
            'minute': total_minutes % 60,
            'day_of_week': info['day_of_week'],
            'location': info.get('location', 'Norman Niles #1')
        }
    return {'hour': 12, 'minute': 0, 'day_of_week': 0, 'location': 'Norman Niles #1'}

def make_features(hour, minute, day_of_week, location_name):
    loc_encoded = location_encoder.get(location_name, 0)
    is_rush_hour = 1 if hour in [7,8,9,16,17,18] else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    is_morning = 1 if hour < 12 else 0
    is_evening = 1 if 17 <= hour < 21 else 0
    is_night = 1 if hour < 6 or hour >= 22 else 0
    
    feats = {
        'hour': hour,
        'minute': minute,
        'day_of_week': day_of_week,
        'day_of_month': 1,
        'month': 1,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend,
        'is_morning': is_morning,
        'is_evening': is_evening,
        'is_night': is_night,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
        'minute_sin': np.sin(2 * np.pi * minute / 60),
        'minute_cos': np.cos(2 * np.pi * minute / 60),
        'location_encoded': loc_encoded,
        'rush_x_location': is_rush_hour * loc_encoded,
        'hour_x_weekend': hour * is_weekend,
        'hour_x_location': hour * loc_encoded,
        'weekend_x_location': is_weekend * loc_encoded,
        'hour_squared': hour ** 2,
        'location_squared': loc_encoded ** 2
    }
    return pd.DataFrame([feats])[feature_cols]

sample_df = pd.read_csv('SampleSubmission.csv')
ids = sample_df['ID'].tolist()

submission_rows = []

for req_id in ids:
    parts = req_id.split('_')
    segment_id = int(parts[2])
    loc_parts = []
    for i in range(3, len(parts)):
        if parts[i] == 'congestion':
            break
        loc_parts.append(parts[i])
    location_name = ' '.join(loc_parts)
    rating_type = parts[-2]
    
    tinfo = interpolate_segment_time(segment_id)
    X = make_features(tinfo['hour'], tinfo['minute'], tinfo['day_of_week'], location_name)
    
    if rating_type == 'enter':
        probs_xgb = enter_xgb.predict_proba(X)[0]
        probs_lgb = enter_lgb.predict_proba(X)[0]
    else:
        probs_xgb = exit_xgb.predict_proba(X)[0]
        probs_lgb = exit_lgb.predict_proba(X)[0]
    
    # Blend
    probs_blend = (probs_xgb + probs_lgb) / 2
    pred_label = LABELS[int(probs_blend.argmax())]
    
    submission_rows.append({
        'ID': req_id,
        'Target': pred_label,
        'Target_Accuracy': pred_label
    })

subm = pd.DataFrame(submission_rows)
subm.to_csv('submission_gbm_blend.csv', index=False)

print('[OK] Saved submission_gbm_blend.csv')

# Distribution
print('\n[STATS] Prediction distribution:')
dist = subm['Target'].value_counts(normalize=True)*100
for l in LABELS:
    print(f'  {l}: {dist.get(l,0):.1f}%')

print(f'\n[INFO] Expected OOF performance: F1≈0.46, Acc≈0.76')
print(f'[INFO] This is above baseline but below target; consider ensemble with Dev11.1')
