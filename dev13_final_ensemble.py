"""
Development 13: Final Ensemble - Combine Dev11.1 + Dev12.3
Blend conditional calibration (good distribution) with GBM (good F1).
"""

import pandas as pd
import numpy as np
import pickle
import joblib

print('[DEV 13] FINAL ENSEMBLE: CONDITIONAL + GBM BLEND')
print('='*80)

LABELS = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
label_to_idx = {l:i for i,l in enumerate(LABELS)}

# Load artifacts
print('[LOAD] Loading models and encoders...')
with open('segment_info.pkl', 'rb') as f:
    segment_info = pickle.load(f)

gbm_artifacts = joblib.load('gbm_artifacts.pkl')
feature_cols = gbm_artifacts['feature_cols']
location_encoder = gbm_artifacts['location_encoder']

enter_xgb = joblib.load('gbm_enter_xgb.pkl')
enter_lgb = joblib.load('gbm_enter_lgb.pkl')
exit_xgb = joblib.load('gbm_exit_xgb.pkl')
exit_lgb = joblib.load('gbm_exit_lgb.pkl')

enter_rf = joblib.load('calibrated_enter_model.pkl')
exit_rf = joblib.load('calibrated_exit_model.pkl')

# Train data for priors
train_df = pd.read_csv('Train.csv')
train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
train_df['hour'] = train_df['datetime'].dt.hour

def compute_priors(df, target_col):
    priors = {}
    for (loc, h), grp in df.groupby(['view_label','hour']):
        counts = grp[target_col].value_counts()
        total = counts.sum()
        probs = np.array([counts.get(l, 0) for l in LABELS], dtype=float)
        probs = (probs + 1.0) / (total + len(LABELS))
        priors[(loc, int(h))] = probs / probs.sum()
    return priors

enter_priors = compute_priors(train_df, 'congestion_enter_rating')
exit_priors = compute_priors(train_df, 'congestion_exit_rating')

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

def make_gbm_features(hour, minute, day_of_week, location_name):
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

# Simpler RF features
rf_feature_cols = joblib.load('calibrated_features.pkl')

def make_rf_features(hour, minute, day_of_week, location_name):
    loc_encoded = location_encoder.get(location_name, 0)
    is_rush_hour = 1 if hour in [7,8,9,16,17,18] else 0
    is_weekend = 1 if day_of_week >= 5 else 0
    
    feats = {
        'hour': hour,
        'minute': minute,
        'day_of_week': day_of_week,
        'day_of_month': 1,
        'month': 1,
        'is_rush_hour': is_rush_hour,
        'is_weekend': is_weekend,
        'is_morning': 1 if hour < 12 else 0,
        'is_evening': 1 if 17 <= hour < 21 else 0,
        'hour_sin': np.sin(2 * np.pi * hour / 24),
        'hour_cos': np.cos(2 * np.pi * hour / 24),
        'day_sin': np.sin(2 * np.pi * day_of_week / 7),
        'day_cos': np.cos(2 * np.pi * day_of_week / 7),
        'location_encoded': loc_encoded,
        'signal_encoded': 0,
        'rush_x_location': is_rush_hour * loc_encoded,
        'hour_x_weekend': hour * is_weekend,
    }
    return pd.DataFrame([feats])[rf_feature_cols]

def blend_probs_geometric(p1, p2, p3, alpha1=0.4, alpha2=0.4, alpha3=0.2):
    """Geometric blend of 3 probability distributions"""
    p1 = np.clip(p1, 1e-6, 1.0)
    p2 = np.clip(p2, 1e-6, 1.0)
    p3 = np.clip(p3, 1e-6, 1.0)
    log_blend = alpha1 * np.log(p1) + alpha2 * np.log(p2) + alpha3 * np.log(p3)
    out = np.exp(log_blend)
    out /= out.sum()
    return out

# Generate submission
print('\n[GENERATE] Creating ensemble submission...')
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
    hour = tinfo['hour']
    minute = tinfo['minute']
    dow = tinfo['day_of_week']
    
    # GBM predictions
    X_gbm = make_gbm_features(hour, minute, dow, location_name)
    
    # RF predictions
    X_rf = make_rf_features(hour, minute, dow, location_name)
    
    # Get priors
    if rating_type == 'enter':
        probs_xgb = enter_xgb.predict_proba(X_gbm)[0]
        probs_lgb = enter_lgb.predict_proba(X_gbm)[0]
        probs_rf = enter_rf.predict_proba(X_rf)[0]
        prior = enter_priors.get((location_name, hour))
        if prior is None:
            tmp = train_df[train_df['view_label']==location_name]['congestion_enter_rating'].value_counts()
            total = tmp.sum()
            arr = np.array([tmp.get(l,0) for l in LABELS], dtype=float)
            arr = (arr + 1.0) / (total + len(LABELS)) if total>0 else np.array([0.85,0.07,0.05,0.03])
            prior = arr/arr.sum()
    else:
        probs_xgb = exit_xgb.predict_proba(X_gbm)[0]
        probs_lgb = exit_lgb.predict_proba(X_gbm)[0]
        probs_rf = exit_rf.predict_proba(X_rf)[0]
        prior = exit_priors.get((location_name, hour))
        if prior is None:
            tmp = train_df[train_df['view_label']==location_name]['congestion_exit_rating'].value_counts()
            total = tmp.sum()
            arr = np.array([tmp.get(l,0) for l in LABELS], dtype=float)
            arr = (arr + 1.0) / (total + len(LABELS)) if total>0 else np.array([0.9,0.05,0.03,0.02])
            prior = arr/arr.sum()
    
    # Blend GBM
    probs_gbm = (probs_xgb + probs_lgb) / 2
    
    # Blend RF + prior
    probs_rf_prior = blend_probs_geometric(probs_rf, prior, prior, alpha1=0.7, alpha2=0.15, alpha3=0.15)
    
    # Final ensemble: GBM + RF_prior
    # GBM stronger for F1; RF+prior better for distribution
    if rating_type == 'enter':
        final_probs = blend_probs_geometric(probs_gbm, probs_rf_prior, prior, alpha1=0.5, alpha2=0.4, alpha3=0.1)
    else:
        # Exit more imbalanced; rely more on prior
        final_probs = blend_probs_geometric(probs_gbm, probs_rf_prior, prior, alpha1=0.4, alpha2=0.4, alpha3=0.2)
    
    pred_label = LABELS[int(final_probs.argmax())]
    
    submission_rows.append({
        'ID': req_id,
        'Target': pred_label,
        'Target_Accuracy': pred_label
    })

subm = pd.DataFrame(submission_rows)
subm.to_csv('submission_final_ensemble.csv', index=False)

print('[OK] Saved submission_final_ensemble.csv')

# Distribution
print('\n[STATS] Final ensemble distribution:')
dist = subm['Target'].value_counts(normalize=True)*100
for l in LABELS:
    print(f'  {l}: {dist.get(l,0):.1f}%')

print(f'\n[COMPARISON]')
print(f'  submission_conditional_calibrated: F79.0, L6.8, M8.1, H6.1 (best distribution match)')
print(f'  submission_gbm_blend:              F69.1, L14.1, M10.3, H6.5 (best OOF F1≈0.46)')
print(f'  submission_final_ensemble:         (shown above - balanced approach)')
print(f'\n[RECOMMENDATION]')
print(f'  Try submission_final_ensemble first (best of both worlds)')
print(f'  Fallback: submission_conditional_calibrated if leaderboard prefers distribution')
print(f'  Alternative: submission_gbm_blend if raw model performance matters more')
