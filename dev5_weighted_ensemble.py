"""
Development 5: Weighted Ensemble - Rules + Model Predictions
Combines probabilistic rules with ML model predictions for better accuracy
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

def generate_improved_submission_v5():
    """Generate submission with weighted ensemble of rules and model"""
    
    print('🔍 GELİŞTİRME 5: Weighted Ensemble (Rules + Model)')
    print('='*60)
    
    # Load all resources
    print('\n📦 Kaynaklar yükleniyor...')
    
    with open('segment_info.pkl', 'rb') as f:
        segment_info = pickle.load(f)
    with open('location_hour_rules.pkl', 'rb') as f:
        location_hour_rules = pickle.load(f)
    
    enter_model = joblib.load('time_based_enter_model.pkl')
    exit_model = joblib.load('time_based_exit_model.pkl')
    label_encoders = joblib.load('time_based_label_encoders.pkl')
    feature_cols = joblib.load('time_based_features.pkl')
    
    le_enter = label_encoders['enter']
    le_exit = label_encoders['exit']
    
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = sample_df['ID'].tolist()
    
    train_df = pd.read_csv('Train.csv')
    signal_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    location_signals = train_df.groupby('view_label')['signaling'].apply(
        lambda x: signal_map.get(x.mode()[0] if len(x.mode()) > 0 else 'none', 0)
    ).to_dict()
    
    print(f'✓ Kaynaklar yüklendi')
    
    # Severity mapping
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    # Interpolation function
    def interpolate_segment_time(segment_id):
        known_segments = sorted(segment_info.keys())
        
        if segment_id in segment_info:
            return segment_info[segment_id]
        
        lower = [s for s in known_segments if s < segment_id]
        upper = [s for s in known_segments if s > segment_id]
        
        if lower and upper:
            seg_lower = max(lower)
            seg_upper = min(upper)
            info_lower = segment_info[seg_lower]
            info_upper = segment_info[seg_upper]
            
            ratio = (segment_id - seg_lower) / (seg_upper - seg_lower)
            lower_minutes = info_lower['hour'] * 60 + info_lower['minute']
            upper_minutes = info_upper['hour'] * 60 + info_upper['minute']
            
            if upper_minutes < lower_minutes:
                upper_minutes += 1440
            
            interpolated_minutes = int(lower_minutes + ratio * (upper_minutes - lower_minutes))
            interpolated_minutes = interpolated_minutes % 1440
            
            return {
                'hour': interpolated_minutes // 60,
                'minute': interpolated_minutes % 60,
                'day_of_week': info_lower['day_of_week'],
                'location': info_lower.get('location', 'Norman Niles #1')
            }
        elif lower:
            seg_lower = max(lower)
            info = segment_info[seg_lower].copy()
            diff = segment_id - seg_lower
            total_minutes = info['hour'] * 60 + info['minute'] + diff
            info['hour'] = (total_minutes // 60) % 24
            info['minute'] = total_minutes % 60
            return info
        else:
            return {'hour': 12, 'minute': 0, 'day_of_week': 0, 'location': 'Norman Niles #1'}
    
    # Generate predictions with ensemble
    print('\n🔮 Ensemble predictions oluşturuluyor...')
    np.random.seed(42)
    
    submission_data = []
    ensemble_used = 0
    rule_only = 0
    model_only = 0
    
    for req_id in required_ids:
        try:
            parts = req_id.split('_')
            segment_id = int(parts[2])
            
            location_parts = []
            for i in range(3, len(parts)):
                if parts[i] == 'congestion':
                    break
                location_parts.append(parts[i])
            location_name = ' '.join(location_parts)
            rating_type = parts[-2]
            
            # Get time info
            time_info = interpolate_segment_time(segment_id)
            hour = time_info['hour']
            minute = time_info['minute']
            day_of_week = time_info['day_of_week']
            
            # Get rule-based prediction
            rule_prediction = None
            rule_confidence = 0.0
            
            if location_name in location_hour_rules:
                rules = location_hour_rules[location_name][rating_type]
                
                if hour in rules and len(rules[hour]) > 0:
                    # Get distribution
                    dist = rules[hour]
                    
                    # Sample from distribution for probabilistic prediction
                    labels = list(dist.keys())
                    probs = list(dist.values())
                    rule_prediction = np.random.choice(labels, p=probs)
                    
                    # Confidence = probability of top prediction
                    rule_confidence = max(probs)
            
            # Get model prediction
            is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
            is_weekend = 1 if day_of_week >= 5 else 0
            is_morning = 1 if hour < 12 else 0
            is_evening = 1 if (hour >= 16 and hour < 20) else 0
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            signal_encoded = location_signals.get(location_name, 0)
            
            features_dict = {
                'hour': hour,
                'minute': minute,
                'day_of_week': day_of_week,
                'is_rush_hour': is_rush_hour,
                'is_weekend': is_weekend,
                'is_morning': is_morning,
                'is_evening': is_evening,
                'hour_sin': hour_sin,
                'hour_cos': hour_cos,
                'signal_encoded': signal_encoded
            }
            
            X = pd.DataFrame([features_dict])[feature_cols]
            
            if rating_type == 'enter':
                pred_encoded = enter_model.predict(X)[0]
                model_prediction = le_enter.inverse_transform([pred_encoded])[0]
                
                # Get model confidence (probability)
                model_proba = enter_model.predict_proba(X)[0]
                model_confidence = model_proba[pred_encoded]
            else:
                pred_encoded = exit_model.predict(X)[0]
                model_prediction = le_exit.inverse_transform([pred_encoded])[0]
                
                model_proba = exit_model.predict_proba(X)[0]
                model_confidence = model_proba[pred_encoded]
            
            # Ensemble decision
            if rule_prediction is not None:
                # We have both predictions
                # Weight: 70% rules (from actual data), 30% model
                rule_weight = 0.70
                model_weight = 0.30
                
                # Convert to severity
                rule_severity = severity_map[rule_prediction]
                model_severity = severity_map[model_prediction]
                
                # Weighted average
                weighted_severity = rule_weight * rule_severity + model_weight * model_severity
                final_severity = int(round(weighted_severity))
                
                # Clip to valid range
                final_severity = max(0, min(3, final_severity))
                
                prediction = reverse_map[final_severity]
                ensemble_used += 1
                
            elif rule_prediction is not None:
                # Only rule available
                prediction = rule_prediction
                rule_only += 1
            else:
                # Only model available
                prediction = model_prediction
                model_only += 1
            
        except Exception as e:
            prediction = 'free flowing'
            model_only += 1
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    print(f'\n✅ Ensemble predictions tamamlandı')
    print(f'📊 Ensemble (rules+model): {ensemble_used} ({ensemble_used/len(required_ids)*100:.1f}%)')
    print(f'📋 Rule only: {rule_only} ({rule_only/len(required_ids)*100:.1f}%)')
    print(f'🤖 Model only: {model_only} ({model_only/len(required_ids)*100:.1f}%)')
    
    # Apply temporal smoothing on ensemble results
    print('\n🔧 Temporal smoothing uygulanıyor...')
    submission_df = apply_temporal_smoothing(submission_df.copy(), window_size=3)
    
    # Apply anomaly detection
    print('\n🔧 Anomaly detection uygulanıyor...')
    submission_df = apply_anomaly_detection(submission_df.copy())
    
    # Save
    submission_df.to_csv('submission.csv', index=False)
    
    print(f'\n✅ Submission güncellendi: submission.csv')
    print(f'📊 Toplam: {len(submission_df):,} satır')
    
    # Distribution
    print(f'\n📈 Final Tahmin Dağılımı:')
    dist = submission_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(submission_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    # Compare with training
    print('\n📊 Training vs Submission:')
    train_enter_dist = train_df['congestion_enter_rating'].value_counts(normalize=True) * 100
    train_exit_dist = train_df['congestion_exit_rating'].value_counts(normalize=True) * 100
    
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        enter_pct = train_enter_dist.get(label, 0)
        exit_pct = train_exit_dist.get(label, 0)
        train_avg = (enter_pct + exit_pct) / 2
        
        submit_pct = dist.get(label, 0) / len(submission_df) * 100
        diff = submit_pct - train_avg
        
        print(f'  {label}:')
        print(f'    Training: {train_avg:.1f}% | Submission: {submit_pct:.1f}% | Diff: {diff:+.1f}%')
    
    return submission_df


def apply_temporal_smoothing(df, window_size=3):
    """Apply temporal smoothing"""
    
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    df['segment_id'] = df['ID'].apply(lambda x: int(x.split('_')[2]))
    df['location'] = df['ID'].apply(
        lambda x: ' '.join([p for i, p in enumerate(x.split('_')) 
                           if i >= 3 and p != 'congestion'])
    )
    df['rating_type'] = df['ID'].apply(lambda x: x.split('_')[-2])
    df['severity'] = df['Target'].map(severity_map)
    
    smoothed_data = []
    changes = 0
    
    for (location, rating_type), group in df.groupby(['location', 'rating_type']):
        group = group.sort_values('segment_id').copy()
        
        if len(group) >= window_size:
            original = group['severity'].values.copy()
            smoothed = group['severity'].rolling(
                window=window_size, center=True, min_periods=1
            ).median().round().astype(int).values
            
            changes += (original != smoothed).sum()
            group['severity'] = smoothed
        
        group['Target'] = group['severity'].map(reverse_map)
        group['Target_Accuracy'] = group['Target']
        smoothed_data.append(group)
    
    result = pd.concat(smoothed_data, ignore_index=True).sort_index()
    print(f'  ✓ {changes} prediction değiştirildi')
    
    return result[['ID', 'Target', 'Target_Accuracy']]


def apply_anomaly_detection(df):
    """Detect and fix anomalies"""
    
    severity_map = {
        'free flowing': 0,
        'light delay': 1,
        'moderate delay': 2,
        'heavy delay': 3
    }
    reverse_map = {v: k for k, v in severity_map.items()}
    
    df['segment_id'] = df['ID'].apply(lambda x: int(x.split('_')[2]))
    df['location'] = df['ID'].apply(
        lambda x: ' '.join([p for i, p in enumerate(x.split('_')) 
                           if i >= 3 and p != 'congestion'])
    )
    df['rating_type'] = df['ID'].apply(lambda x: x.split('_')[-2])
    df['severity'] = df['Target'].map(severity_map)
    
    fixed_data = []
    anomalies = 0
    
    for (location, rating_type), group in df.groupby(['location', 'rating_type']):
        group = group.sort_values('segment_id').copy()
        
        if len(group) >= 3:
            severities = group['severity'].values
            
            for i in range(1, len(severities) - 1):
                if severities[i] >= severities[i-1] + 2 and severities[i] >= severities[i+1] + 2:
                    severities[i] = int(round((severities[i-1] + severities[i+1]) / 2))
                    anomalies += 1
            
            group['severity'] = severities
        
        group['Target'] = group['severity'].map(reverse_map)
        group['Target_Accuracy'] = group['Target']
        fixed_data.append(group)
    
    result = pd.concat(fixed_data, ignore_index=True).sort_index()
    print(f'  ✓ {anomalies} anomaly düzeltildi')
    
    return result[['ID', 'Target', 'Target_Accuracy']]


if __name__ == '__main__':
    submission = generate_improved_submission_v5()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 5 TAMAMLANDI')
    print('   → Weighted ensemble (70% rules, 30% model)')
    print('   → Temporal smoothing uygulandı')
    print('   → Anomaly detection yapıldı')
    print('   → En iyi tahminler oluşturuldu')
    print('='*60)
