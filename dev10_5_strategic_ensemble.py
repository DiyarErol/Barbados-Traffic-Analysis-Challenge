"""
Development 10.5: Strategic Ensemble with Class Balancing
Combining Neural Network + Stacking + Rule-based with smart weighting
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler

print('[DEV 10.5] STRATEGIC ENSEMBLE WITH CLASS BALANCING')
print('='*80)

# Load all models
print('\n[LOAD] Loading all trained models...')

# 1. Neural Network
import tensorflow as tf
nn_model = tf.keras.models.load_model('neural_network_model.h5')
nn_scaler = joblib.load('neural_network_scaler.pkl')
nn_features = joblib.load('neural_network_features.pkl')
nn_encoder = joblib.load('neural_network_location_encoder.pkl')
nn_label_map = joblib.load('neural_network_label_map.pkl')

# 2. Stacking Ensemble
stacking_enter = joblib.load('stacking_enter_model.pkl')
stacking_exit = joblib.load('stacking_exit_model.pkl')
stacking_features = joblib.load('stacking_features.pkl')
stacking_encoder = joblib.load('stacking_location_encoder.pkl')
stacking_label_map = joblib.load('stacking_label_map.pkl')

# 3. Rule-based
with open('location_hour_rules.pkl', 'rb') as f:
    location_rules = pickle.load(f)

# 4. Segment info
with open('segment_info.pkl', 'rb') as f:
    segment_info = pickle.load(f)

print('[OK] All models loaded!')

# Reverse label maps
nn_reverse = {v: k for k, v in nn_label_map.items()}
stacking_reverse = {v: k for k, v in stacking_label_map.items()}


def interpolate_segment_time(segment_id):
    """Interpolate time for unknown segments"""
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


def get_neural_network_prediction(features_dict, rating_type):
    """Get NN prediction with probabilities"""
    # Prepare features (24 features)
    hour = features_dict['hour']
    minute = features_dict['minute']
    day_of_week = features_dict['day_of_week']
    location_enc = features_dict['location_encoded_nn']
    signal_enc = features_dict['signal_encoded']
    
    feature_vector = [
        hour, minute, day_of_week, 1, 1,  # day_of_month, month
        features_dict['is_rush_hour'], features_dict['is_weekend'],
        features_dict['is_morning'], features_dict['is_evening'],
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * minute / 60), np.cos(2 * np.pi * minute / 60),
        np.sin(2 * np.pi * day_of_week / 7), np.cos(2 * np.pi * day_of_week / 7),
        location_enc, signal_enc,
        hour**2, hour**3,
        features_dict['is_rush_hour'] * location_enc,
        hour * location_enc,
        features_dict['is_weekend'] * features_dict['is_rush_hour'],
        features_dict['is_morning'] * features_dict['is_rush_hour'],
        features_dict['is_evening'] * features_dict['is_rush_hour']
    ]
    
    # Scale
    X = nn_scaler.transform([feature_vector])
    
    # Predict
    predictions = nn_model.predict(X, verbose=0)
    enter_probs = predictions[0][0]
    exit_probs = predictions[1][0]
    
    if rating_type == 'enter':
        return enter_probs
    else:
        return exit_probs


def get_stacking_prediction(features_dict, rating_type):
    """Get stacking prediction with probabilities"""
    feature_vector = [
        features_dict['hour'], features_dict['minute'],
        features_dict['day_of_week'], 1, 1,  # day_of_month, month
        features_dict['is_rush_hour'], features_dict['is_weekend'],
        features_dict['is_morning'], features_dict['is_evening'],
        features_dict['hour_sin'], features_dict['hour_cos'],
        features_dict['day_sin'], features_dict['day_cos'],
        features_dict['location_encoded_stacking'],
        features_dict['signal_encoded'],
        features_dict['rush_x_location']
    ]
    
    X = pd.DataFrame([feature_vector], columns=stacking_features)
    
    if rating_type == 'enter':
        probs = stacking_enter.predict_proba(X)[0]
    else:
        probs = stacking_exit.predict_proba(X)[0]
    
    return probs


def get_rule_based_prediction(location, hour, rating_type):
    """Get rule-based prediction"""
    location_key = (location, hour)
    if location_key in location_rules:
        rules = location_rules[location_key]
        if rating_type == 'enter' and 'enter_probabilities' in rules:
            return rules['enter_probabilities']
        elif rating_type == 'exit' and 'exit_probabilities' in rules:
            return rules['exit_probabilities']
    
    # Default: mostly free flowing
    return {'free flowing': 0.75, 'light delay': 0.15, 'moderate delay': 0.07, 'heavy delay': 0.03}


def smart_ensemble(nn_probs, stacking_probs, rule_probs, hour, is_rush_hour):
    """Smart weighted ensemble with dynamic weights"""
    
    # Convert rule probs dict to array
    label_order = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
    rule_probs_array = np.array([rule_probs.get(label, 0.0) for label in label_order])
    
    # Dynamic weights based on time
    if is_rush_hour:
        # Rush hour: trust models more, reduce rule weight
        weights = [0.35, 0.45, 0.20]  # NN, Stacking, Rules
    else:
        # Non-rush: balance all sources
        weights = [0.30, 0.40, 0.30]
    
    # Weighted average
    combined_probs = (
        weights[0] * nn_probs +
        weights[1] * stacking_probs +
        weights[2] * rule_probs_array
    )
    
    # Apply class balancing boost
    # Boost non-free-flowing classes slightly
    boost = np.array([1.0, 1.3, 1.5, 1.7])  # Boost light, moderate, heavy
    combined_probs = combined_probs * boost
    
    # Normalize
    combined_probs = combined_probs / combined_probs.sum()
    
    return combined_probs


def generate_strategic_submission():
    """Generate submission with strategic ensemble"""
    
    print('\n[SUBMIT] Generating Strategic Ensemble Submission...')
    print('-'*80)
    
    # Load sample
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = sample_df['ID'].tolist()
    
    # Signal map
    train_df = pd.read_csv('Train.csv')
    signal_map_data = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    location_signals = train_df.groupby('view_label')['signaling'].apply(
        lambda x: signal_map_data.get(x.mode()[0] if len(x.mode()) > 0 else 'none', 0)
    ).to_dict()
    
    submission_data = []
    
    for req_id in required_ids:
        parts = req_id.split('_')
        segment_id = int(parts[2])
        
        location_parts = []
        for i in range(3, len(parts)):
            if parts[i] == 'congestion':
                break
            location_parts.append(parts[i])
        location_name = ' '.join(location_parts)
        rating_type = parts[-2]
        
        # Get time
        time_info = interpolate_segment_time(segment_id)
        hour = time_info['hour']
        minute = time_info['minute']
        day_of_week = time_info['day_of_week']
        
        # Features
        is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
        features = {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'is_rush_hour': is_rush_hour,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_morning': 1 if hour < 12 else 0,
            'is_evening': 1 if 17 <= hour < 21 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'location_encoded_nn': nn_encoder.get(location_name, 0),
            'location_encoded_stacking': stacking_encoder.get(location_name, 0),
            'signal_encoded': location_signals.get(location_name, 0),
            'rush_x_location': is_rush_hour * stacking_encoder.get(location_name, 0)
        }
        
        # Get predictions from all models
        nn_probs = get_neural_network_prediction(features, rating_type)
        stacking_probs = get_stacking_prediction(features, rating_type)
        rule_probs = get_rule_based_prediction(location_name, hour, rating_type)
        
        # Smart ensemble
        final_probs = smart_ensemble(nn_probs, stacking_probs, rule_probs, hour, is_rush_hour)
        
        # Choose class
        pred_class = np.argmax(final_probs)
        label_order = ['free flowing', 'light delay', 'moderate delay', 'heavy delay']
        prediction = label_order[pred_class]
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission_strategic.csv', index=False)
    
    print(f'\n[OK] Strategic submission saved: submission_strategic.csv')
    
    # Distribution
    print(f'\n[STATS] Strategic Prediction Distribution:')
    dist = submission_df['Target'].value_counts(normalize=True) * 100
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        pct = dist.get(label, 0)
        print(f'  {label}: {pct:.1f}%')
    
    # Compare with target
    target_dist = {
        'free flowing': 79.94,
        'light delay': 6.24,
        'moderate delay': 8.58,
        'heavy delay': 5.24
    }
    
    print(f'\n[COMPARE] Comparison with Target Distribution:')
    total_error = 0
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        actual = dist.get(label, 0)
        target = target_dist[label]
        error = abs(actual - target)
        total_error += error
        print(f'  {label}: {actual:.2f}% (target: {target:.2f}%, error: {error:.2f}%)')
    
    print(f'\n[FINAL] Total Distribution Error: {total_error:.2f}%')
    
    return submission_df


if __name__ == '__main__':
    submission = generate_strategic_submission()
    
    print('\n' + '='*80)
    print('[OK] GELISTIRME 10.5 TAMAMLANDI')
    print('   -> Neural Network + XGBoost+LightGBM + Rules ensemble')
    print('   -> Dynamic weighting based on rush hour')
    print('   -> Class balancing boost applied')
    print('   -> submission_strategic.csv generated')
    print('='*80)
