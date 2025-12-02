"""
Development 11.1: Conditional (Location+Hour) Distribution Calibration
Combine model probabilities with location-hour priors for better accuracy.
"""

import pandas as pd
import numpy as np
import pickle
import joblib

print('[DEV 11.1] CONDITIONAL CALIBRATION (Location+Hour priors)')
print('='*80)

# Labels and order
LABELS = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
label_to_idx = {l:i for i,l in enumerate(LABELS)}

# Load models and artifacts from Dev 11
enter_model = joblib.load('calibrated_enter_model.pkl')
exit_model = joblib.load('calibrated_exit_model.pkl')
feature_cols = joblib.load('calibrated_features.pkl')
location_encoder = joblib.load('calibrated_location_encoder.pkl')

# Segment info for time mapping
with open('segment_info.pkl', 'rb') as f:
    segment_info = pickle.load(f)

# Training data for priors
train_df = pd.read_csv('Train.csv')
train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
train_df['hour'] = train_df['datetime'].dt.hour

# Compute priors per (location, hour)
def compute_priors(df, target_col):
    priors = {}
    for (loc, h), grp in df.groupby(['view_label','hour']):
        counts = grp[target_col].value_counts()
        total = counts.sum()
        probs = np.array([counts.get(l, 0) for l in LABELS], dtype=float)
        # Add Laplace smoothing
        probs = (probs + 1.0) / (total + len(LABELS))
        priors[(loc, int(h))] = probs / probs.sum()
    return priors

print('[BUILD] Computing location-hour priors...')
enter_priors = compute_priors(train_df, 'congestion_enter_rating')
exit_priors  = compute_priors(train_df, 'congestion_exit_rating')
print(f'[OK] Priors: enter={len(enter_priors)}, exit={len(exit_priors)}')

# Helper: interpolate time

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

# Build features for RF

def make_features(hour, minute, day_of_week, location_name):
    is_rush_hour = 1 if hour in [7,8,9,16,17,18] else 0
    is_weekend   = 1 if day_of_week >= 5 else 0
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
        'location_encoded': location_encoder.get(location_name, 0),
        # signal not used in Dev 11 models; keep placeholder as 0
        'signal_encoded': 0,
        'rush_x_location': is_rush_hour * location_encoder.get(location_name, 0),
        'hour_x_weekend': hour * is_weekend,
    }
    return pd.DataFrame([feats])[feature_cols]

# Conditional calibration

def blend_probs(model_probs, prior_probs, alpha=0.75):
    # geometric mean in log space to avoid underflow
    p1 = np.clip(model_probs, 1e-6, 1.0)
    p2 = np.clip(prior_probs, 1e-6, 1.0)
    log_blend = alpha * np.log(p1) + (1 - alpha) * np.log(p2)
    out = np.exp(log_blend)
    out /= out.sum()
    return out

print('[LOAD] Loading SampleSubmission IDs...')
sample_df = pd.read_csv('SampleSubmission.csv')
ids = sample_df['ID'].tolist()

submission_rows = []
row_meta = []  # keep metadata and probs for quota adjustment

print('[PREDICT] Predicting with conditional calibration...')
for req_id in ids:
    parts = req_id.split('_')
    segment_id = int(parts[2])
    # location tokens between index 3 and token 'congestion'
    loc_parts = []
    for i in range(3, len(parts)):
        if parts[i] == 'congestion':
            break
        loc_parts.append(parts[i])
    location_name = ' '.join(loc_parts)
    rating_type = parts[-2]  # enter or exit

    tinfo = interpolate_segment_time(segment_id)
    hour = tinfo['hour']
    minute = tinfo['minute']
    dow = tinfo['day_of_week']

    X = make_features(hour, minute, dow, location_name)

    if rating_type == 'enter':
        probs = enter_model.predict_proba(X)[0]
        prior = enter_priors.get((location_name, hour))
        if prior is None:
            # fallback: location-only prior
            tmp = train_df[train_df['view_label']==location_name]['congestion_enter_rating'].value_counts()
            total = tmp.sum()
            arr = np.array([tmp.get(l,0) for l in LABELS], dtype=float)
            arr = (arr + 1.0) / (total + len(LABELS)) if total>0 else np.array([0.85,0.07,0.05,0.03])
            prior = arr/arr.sum()
        final_probs = blend_probs(probs, prior, alpha=0.8)
    else:
        probs = exit_model.predict_proba(X)[0]
        prior = exit_priors.get((location_name, hour))
        if prior is None:
            tmp = train_df[train_df['view_label']==location_name]['congestion_exit_rating'].value_counts()
            total = tmp.sum()
            arr = np.array([tmp.get(l,0) for l in LABELS], dtype=float)
            arr = (arr + 1.0) / (total + len(LABELS)) if total>0 else np.array([0.9,0.05,0.03,0.02])
            prior = arr/arr.sum()
        final_probs = blend_probs(probs, prior, alpha=0.7)

    # low-confidence override (for extreme imbalanced exit)
    if final_probs.max() < 0.55:
        # push slightly towards prior-heavy blend
        final_probs = blend_probs(final_probs, prior, alpha=0.5)

    pred_idx = int(final_probs.argmax())
    pred_label = LABELS[pred_idx]

    submission_rows.append({
        'ID': req_id,
        'Target': pred_label,
        'Target_Accuracy': pred_label
    })
    row_meta.append({
        'ID': req_id,
        'type': rating_type,
        'probs': final_probs,
        'pred_idx': pred_idx
    })

subm = pd.DataFrame(submission_rows)

# Global target distributions from Train (per type)
print('[ADJUST] Applying soft global quota per type...')
train_enter_dist = (train_df['congestion_enter_rating'].value_counts() + 1.0)
train_enter_dist = train_enter_dist / train_enter_dist.sum()
train_exit_dist = (train_df['congestion_exit_rating'].value_counts() + 1.0)
train_exit_dist = train_exit_dist / train_exit_dist.sum()

row_meta_df = pd.DataFrame(row_meta)

for rtype, dist_series in [('enter', train_enter_dist), ('exit', train_exit_dist)]:
    mask = row_meta_df['type'] == rtype
    idxs = row_meta_df.index[mask].tolist()
    desired_counts = {
        label_to_idx[l]: int(round(dist_series.get(l, 0) * len(idxs))) for l in LABELS
    }
    current_preds = [row_meta_df.at[i,'pred_idx'] for i in idxs]
    current_counts = {k: current_preds.count(k) for k in range(len(LABELS))}

    # build candidate pool with margins (confidence gap)
    candidates = []
    for i in idxs:
        probs = row_meta_df.at[i,'probs']
        top2 = probs.argsort()[::-1][:2]
        margin = probs[top2[0]] - probs[top2[1]]
        candidates.append((i, int(top2[0]), int(top2[1]), float(margin)))

    # If a class is over target, move lowest-margin items to most plausible alternative
    for cls in range(len(LABELS)):
        over = current_counts.get(cls,0) - desired_counts.get(cls,0)
        if over > 0:
            pool = [(i,t1,t2,m) for (i,t1,t2,m) in candidates if t1==cls]
            pool.sort(key=lambda x: x[3])  # ascending margin
            for (i,t1,t2,m) in pool[:over]:
                # reassign to t2 if it improves towards quota; else pick best class with room
                assign_to = t2
                # find any class with shortage
                shortages = [c for c in range(len(LABELS)) if current_counts.get(c,0) < desired_counts.get(c,0)]
                if shortages and assign_to not in shortages:
                    # choose among shortages the class with highest prob for this row
                    probs = row_meta_df.at[i,'probs']
                    assign_to = max(shortages, key=lambda c: probs[c])
                # update counts and row
                current_counts[cls] -= 1
                current_counts[assign_to] = current_counts.get(assign_to,0) + 1
                row_meta_df.at[i,'pred_idx'] = assign_to

# Write adjusted predictions back
subm['Target'] = [LABELS[row_meta_df.at[i,'pred_idx']] for i in row_meta_df.index]
subm['Target_Accuracy'] = subm['Target']
subm.to_csv('submission_conditional_calibrated.csv', index=False)

print('[OK] Saved submission_conditional_calibrated.csv')
# Show distribution
print('\n[STATS] Prediction distribution:')
dist = subm['Target'].value_counts(normalize=True)*100
for l in LABELS:
    print(f'  {l}: {dist.get(l,0):.1f}%')
