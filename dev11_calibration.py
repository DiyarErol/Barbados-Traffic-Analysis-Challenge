"""
Development 11: Distribution Calibration Model
Train with proper class weights and apply target distribution constraints
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

print('[DEV 11] DISTRIBUTION CALIBRATION MODEL')
print('='*80)

# Target distribution (from problem description)
TARGET_DISTRIBUTION = {
    'free flowing': 0.7994,
    'light delay': 0.0624,
    'moderate delay': 0.0858,
    'heavy delay': 0.0524
}

def prepare_training_data_with_balanced_sampling():
    """Prepare training data with balanced sampling"""
    
    print('\n[LOAD] Loading and preparing training data...')
    
    train_df = pd.read_csv('Train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
    
    # Time features
    train_df['hour'] = train_df['datetime'].dt.hour
    train_df['minute'] = train_df['datetime'].dt.minute
    train_df['day_of_week'] = train_df['datetime'].dt.dayofweek
    train_df['day_of_month'] = train_df['datetime'].dt.day
    train_df['month'] = train_df['datetime'].dt.month
    
    # Boolean features
    train_df['is_rush_hour'] = train_df['hour'].apply(
        lambda h: 1 if h in [7, 8, 9, 16, 17, 18] else 0
    )
    train_df['is_weekend'] = train_df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    train_df['is_morning'] = train_df['hour'].apply(lambda h: 1 if h < 12 else 0)
    train_df['is_evening'] = train_df['hour'].apply(lambda h: 1 if 17 <= h < 21 else 0)
    
    # Cyclical encoding
    train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
    train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)
    train_df['day_sin'] = np.sin(2 * np.pi * train_df['day_of_week'] / 7)
    train_df['day_cos'] = np.cos(2 * np.pi * train_df['day_of_week'] / 7)
    
    # Location encoding
    location_encoder = {loc: idx for idx, loc in enumerate(train_df['view_label'].unique())}
    train_df['location_encoded'] = train_df['view_label'].map(location_encoder)
    
    # Signal encoding
    signal_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    train_df['signal_encoded'] = train_df['signaling'].map(signal_map).fillna(0)
    
    # Interaction features
    train_df['rush_x_location'] = train_df['is_rush_hour'] * train_df['location_encoded']
    train_df['hour_x_weekend'] = train_df['hour'] * train_df['is_weekend']
    
    print(f'[OK] Features prepared: {train_df.shape[1]} columns')
    
    return train_df, location_encoder


def train_calibrated_models():
    """Train models with proper class weights"""
    
    print('\n[TRAIN] Training Calibrated Models...')
    print('-'*80)
    
    # Prepare data
    train_df, location_encoder = prepare_training_data_with_balanced_sampling()
    
    feature_cols = [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
        'is_rush_hour', 'is_weekend', 'is_morning', 'is_evening',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'location_encoded', 'signal_encoded', 'rush_x_location', 'hour_x_weekend'
    ]
    
    # Encode labels
    label_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    
    train_df['enter_encoded'] = train_df['congestion_enter_rating'].map(label_map)
    train_df['exit_encoded'] = train_df['congestion_exit_rating'].map(label_map)
    
    # Filter valid samples
    valid_samples = train_df[train_df['enter_encoded'].notna() & train_df['exit_encoded'].notna()]
    X = valid_samples[feature_cols].values
    y_enter = valid_samples['enter_encoded'].values
    y_exit = valid_samples['exit_encoded'].values
    
    print(f'\n[STATS] Dataset:')
    print(f'  Samples: {len(X):,}')
    print(f'  Features: {len(feature_cols)}')
    
    # Check class distribution
    print(f'\n[STATS] Enter Class Distribution:')
    for label, code in label_map.items():
        count = np.sum(y_enter == code)
        pct = count / len(y_enter) * 100
        print(f'  {label}: {count:,} ({pct:.1f}%)')
    
    print(f'\n[STATS] Exit Class Distribution:')
    for label, code in label_map.items():
        count = np.sum(y_exit == code)
        pct = count / len(y_exit) * 100
        print(f'  {label}: {count:,} ({pct:.1f}%)')
    
    # Compute class weights (inverse frequency)
    enter_class_weights = compute_class_weight('balanced', classes=np.unique(y_enter), y=y_enter)
    exit_class_weights = compute_class_weight('balanced', classes=np.unique(y_exit), y=y_exit)
    
    enter_class_weight_dict = {i: w for i, w in enumerate(enter_class_weights)}
    exit_class_weight_dict = {i: w for i, w in enumerate(exit_class_weights)}
    
    print(f'\n[STATS] Enter Class Weights:')
    for label, code in label_map.items():
        print(f'  {label}: {enter_class_weight_dict[code]:.2f}')
    
    print(f'\n[STATS] Exit Class Weights:')
    for label, code in label_map.items():
        print(f'  {label}: {exit_class_weight_dict[code]:.2f}')
    
    # Train/test split
    X_train, X_test, y_enter_train, y_enter_test, y_exit_train, y_exit_test = train_test_split(
        X, y_enter, y_exit, test_size=0.2, random_state=42, stratify=y_enter
    )
    
    print(f'\n[STATS] Split:')
    print(f'  Training: {len(X_train):,}')
    print(f'  Testing: {len(X_test):,}')
    
    # ===== ENTER MODEL with class weights =====
    print(f'\n[TRAIN] Training ENTER model with balanced class weights...')
    
    enter_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=enter_class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    
    enter_model.fit(X_train, y_enter_train)
    
    # Evaluate
    y_enter_pred = enter_model.predict(X_test)
    enter_acc = accuracy_score(y_enter_test, y_enter_pred)
    print(f'[EVAL] Enter Accuracy: {enter_acc*100:.2f}%')
    
    # Check prediction distribution
    enter_pred_dist = pd.Series(y_enter_pred).value_counts(normalize=True).sort_index()
    print(f'[EVAL] Enter Prediction Distribution on Test:')
    for label, code in label_map.items():
        pct = enter_pred_dist.get(code, 0) * 100
        print(f'  {label}: {pct:.1f}%')
    
    # ===== EXIT MODEL with class weights =====
    print(f'\n[TRAIN] Training EXIT model with balanced class weights...')
    
    exit_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight=exit_class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    
    exit_model.fit(X_train, y_exit_train)
    
    # Evaluate
    y_exit_pred = exit_model.predict(X_test)
    exit_acc = accuracy_score(y_exit_test, y_exit_pred)
    print(f'[EVAL] Exit Accuracy: {exit_acc*100:.2f}%')
    
    # Check prediction distribution
    exit_pred_dist = pd.Series(y_exit_pred).value_counts(normalize=True).sort_index()
    print(f'[EVAL] Exit Prediction Distribution on Test:')
    for label, code in label_map.items():
        pct = exit_pred_dist.get(code, 0) * 100
        print(f'  {label}: {pct:.1f}%')
    
    # Save models
    joblib.dump(enter_model, 'calibrated_enter_model.pkl')
    joblib.dump(exit_model, 'calibrated_exit_model.pkl')
    joblib.dump(feature_cols, 'calibrated_features.pkl')
    joblib.dump(location_encoder, 'calibrated_location_encoder.pkl')
    joblib.dump(label_map, 'calibrated_label_map.pkl')
    joblib.dump(enter_class_weight_dict, 'calibrated_enter_class_weights.pkl')
    joblib.dump(exit_class_weight_dict, 'calibrated_exit_class_weights.pkl')
    
    print(f'\n[OK] Models saved!')
    
    return enter_model, exit_model, feature_cols, location_encoder, label_map


def apply_distribution_constraints(predictions, target_dist):
    """Apply target distribution constraints via probability calibration"""
    
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame(predictions, columns=['ID', 'prediction', 'probability'])
    
    # Get current distribution
    current_dist = pred_df['prediction'].value_counts(normalize=True).to_dict()
    
    print('\n[CALIBRATE] Applying Distribution Constraints...')
    print(f'Current distribution:')
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        curr = current_dist.get(label, 0) * 100
        target = target_dist[label] * 100
        print(f'  {label}: {curr:.1f}% (target: {target:.1f}%)')
    
    # Sort by probability ascending (less confident predictions first)
    pred_df = pred_df.sort_values('probability')
    
    # Calculate target counts
    total = len(pred_df)
    target_counts = {label: int(total * prob) for label, prob in target_dist.items()}
    
    # Adjust to ensure sum equals total
    diff = total - sum(target_counts.values())
    target_counts['free flowing'] += diff  # Add difference to majority class
    
    print(f'\n[CALIBRATE] Target Counts:')
    for label, count in target_counts.items():
        print(f'  {label}: {count}')
    
    # Reassign predictions to match target distribution
    # Strategy: Keep high-confidence predictions, adjust low-confidence ones
    final_predictions = pred_df['prediction'].copy()
    
    # For each class, ensure we have target count
    for label in ['heavy delay', 'moderate delay', 'light delay', 'free flowing']:
        current_count = (final_predictions == label).sum()
        target_count = target_counts[label]
        
        if current_count < target_count:
            # Need more of this class - convert some predictions
            needed = target_count - current_count
            
            # Find candidates (predictions that are NOT this class and have low probability)
            candidates_mask = (final_predictions != label) & (pred_df['probability'] < 0.7)
            candidates_indices = pred_df[candidates_mask].index[:needed]
            
            final_predictions.loc[candidates_indices] = label
        
        elif current_count > target_count:
            # Need fewer of this class - convert to free flowing (majority class)
            excess = current_count - target_count
            
            # Find low-confidence predictions of this class
            candidates_mask = (final_predictions == label) & (pred_df['probability'] < 0.6)
            candidates_indices = pred_df[candidates_mask].index[:excess]
            
            final_predictions.loc[candidates_indices] = 'free flowing'
    
    # Update predictions
    pred_df['prediction'] = final_predictions
    
    # Check final distribution
    final_dist = pred_df['prediction'].value_counts(normalize=True).to_dict()
    print(f'\n[CALIBRATE] Final Distribution After Constraints:')
    total_error = 0
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        final = final_dist.get(label, 0) * 100
        target = target_dist[label] * 100
        error = abs(final - target)
        total_error += error
        print(f'  {label}: {final:.1f}% (target: {target:.1f}%, error: {error:.2f}%)')
    
    print(f'\n[CALIBRATE] Total Distribution Error: {total_error:.2f}%')
    
    return pred_df['prediction'].values


def generate_calibrated_submission():
    """Generate submission with calibrated predictions"""
    
    print('\n[SUBMIT] Generating Calibrated Submission...')
    print('-'*80)
    
    # Load models
    enter_model = joblib.load('calibrated_enter_model.pkl')
    exit_model = joblib.load('calibrated_exit_model.pkl')
    feature_cols = joblib.load('calibrated_features.pkl')
    location_encoder = joblib.load('calibrated_location_encoder.pkl')
    label_map = joblib.load('calibrated_label_map.pkl')
    
    reverse_label_map = {v: k for k, v in label_map.items()}
    
    # Load segment info
    with open('segment_info.pkl', 'rb') as f:
        segment_info = pickle.load(f)
    
    # Load sample
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = sample_df['ID'].tolist()
    
    # Signal map
    train_df = pd.read_csv('Train.csv')
    signal_map_data = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    location_signals = train_df.groupby('view_label')['signaling'].apply(
        lambda x: signal_map_data.get(x.mode()[0] if len(x.mode()) > 0 else 'none', 0)
    ).to_dict()
    
    # Interpolation
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
    
    print(f'\n[PREDICT] Predicting with calibrated models...')
    
    enter_predictions = []
    exit_predictions = []
    
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
        is_weekend = 1 if day_of_week >= 5 else 0
        
        features = {
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
            'signal_encoded': location_signals.get(location_name, 0),
            'rush_x_location': is_rush_hour * location_encoder.get(location_name, 0),
            'hour_x_weekend': hour * is_weekend
        }
        
        X = pd.DataFrame([features])[feature_cols]
        
        # Predict with probability
        if rating_type == 'enter':
            probs = enter_model.predict_proba(X)[0]
            pred_class = np.argmax(probs)
            max_prob = probs[pred_class]
            prediction = reverse_label_map[pred_class]
            
            enter_predictions.append([req_id, prediction, max_prob])
        else:
            probs = exit_model.predict_proba(X)[0]
            pred_class = np.argmax(probs)
            max_prob = probs[pred_class]
            prediction = reverse_label_map[pred_class]
            
            exit_predictions.append([req_id, prediction, max_prob])
    
    # Apply distribution constraints separately for enter and exit
    print('\n[CALIBRATE] Applying constraints to ENTER predictions...')
    enter_calibrated = apply_distribution_constraints(enter_predictions, TARGET_DISTRIBUTION)
    
    print('\n[CALIBRATE] Applying constraints to EXIT predictions...')
    exit_calibrated = apply_distribution_constraints(exit_predictions, TARGET_DISTRIBUTION)
    
    # Combine
    submission_data = []
    enter_idx = 0
    exit_idx = 0
    
    for req_id in required_ids:
        rating_type = req_id.split('_')[-2]
        
        if rating_type == 'enter':
            prediction = enter_calibrated[enter_idx]
            enter_idx += 1
        else:
            prediction = exit_calibrated[exit_idx]
            exit_idx += 1
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission_calibrated.csv', index=False)
    
    print(f'\n[OK] Calibrated submission saved: submission_calibrated.csv')
    
    # Overall distribution
    print(f'\n[STATS] Overall Prediction Distribution:')
    dist = submission_df['Target'].value_counts(normalize=True) * 100
    total_error = 0
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        actual = dist.get(label, 0)
        target = TARGET_DISTRIBUTION[label] * 100
        error = abs(actual - target)
        total_error += error
        print(f'  {label}: {actual:.1f}% (target: {target:.1f}%, error: {error:.2f}%)')
    
    print(f'\n[FINAL] Total Distribution Error: {total_error:.2f}%')
    
    return submission_df


if __name__ == '__main__':
    # Train models
    enter_model, exit_model, features, loc_enc, label_map = train_calibrated_models()
    
    # Generate submission with distribution constraints
    submission = generate_calibrated_submission()
    
    print('\n' + '='*80)
    print('[OK] GELISTIRME 11 TAMAMLANDI')
    print('   -> RandomForest with balanced class weights')
    print('   -> Distribution calibration applied')
    print('   -> submission_calibrated.csv generated')
    print('='*80)
