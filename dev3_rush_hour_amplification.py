"""
Development 3: Rush Hour Congestion Amplification
Increases congestion predictions during peak hours with location-specific rules
"""

import pandas as pd
import numpy as np
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

def analyze_training_distribution():
    """Analyze training data to understand real congestion distribution"""
    
    print('🔍 GELİŞTİRME 3: Rush Hour Congestion Amplification')
    print('='*60)
    
    train_df = pd.read_csv('Train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
    train_df['hour'] = train_df['datetime'].dt.hour
    
    print('\n📊 Training Data Gerçek Dağılımı:')
    print('\nEnter Rating:')
    enter_dist = train_df['congestion_enter_rating'].value_counts(normalize=True) * 100
    for label, pct in enter_dist.items():
        print(f'  {label}: {pct:.1f}%')
    
    print('\nExit Rating:')
    exit_dist = train_df['congestion_exit_rating'].value_counts(normalize=True) * 100
    for label, pct in exit_dist.items():
        print(f'  {label}: {pct:.1f}%')
    
    # Rush hour analysis
    print('\n🚦 Rush Hour (07-09, 16-18) Analizi:')
    rush_hours = [7, 8, 9, 16, 17, 18]
    rush_data = train_df[train_df['hour'].isin(rush_hours)]
    
    print('\nRush Hour Enter:')
    rush_enter = rush_data['congestion_enter_rating'].value_counts(normalize=True) * 100
    for label, pct in rush_enter.items():
        print(f'  {label}: {pct:.1f}%')
    
    print('\nRush Hour Exit:')
    rush_exit = rush_data['congestion_exit_rating'].value_counts(normalize=True) * 100
    for label, pct in rush_exit.items():
        print(f'  {label}: {pct:.1f}%')
    
    return train_df


def create_enhanced_rules(train_df):
    """Create location and time specific congestion rules"""
    
    print('\n📋 Enhanced Congestion Rules Oluşturuluyor...')
    
    # Location-hour specific congestion probabilities
    location_hour_rules = {}
    
    for location in train_df['view_label'].unique():
        loc_data = train_df[train_df['view_label'] == location]
        location_hour_rules[location] = {'enter': {}, 'exit': {}}
        
        for hour in range(24):
            hour_data = loc_data[loc_data['hour'] == hour]
            
            if len(hour_data) > 0:
                # Enter distribution for this hour
                enter_dist = hour_data['congestion_enter_rating'].value_counts(normalize=True).to_dict()
                location_hour_rules[location]['enter'][hour] = enter_dist
                
                # Exit distribution
                exit_dist = hour_data['congestion_exit_rating'].value_counts(normalize=True).to_dict()
                location_hour_rules[location]['exit'][hour] = exit_dist
    
    # Save rules
    with open('location_hour_rules.pkl', 'wb') as f:
        pickle.dump(location_hour_rules, f)
    
    print(f'✓ {len(location_hour_rules)} lokasyon için probabilistic rules oluşturuldu')
    
    # Show sample
    if 'Norman Niles #1' in location_hour_rules:
        print('\n📋 Örnek: Norman Niles #1, Saat 17 (Rush Hour)')
        rules = location_hour_rules['Norman Niles #1']
        if 17 in rules['enter']:
            print('  Enter distribution:')
            for label, prob in sorted(rules['enter'][17].items(), key=lambda x: x[1], reverse=True):
                print(f'    {label}: {prob*100:.1f}%')
    
    return location_hour_rules


def generate_improved_submission_v3():
    """Generate submission with probabilistic rush hour congestion"""
    
    print('\n' + '='*60)
    print('📝 SUBMISSION V3: Probabilistic Rush Hour Congestion')
    print('='*60)
    
    # Load patterns and rules
    try:
        with open('segment_info.pkl', 'rb') as f:
            segment_info = pickle.load(f)
        with open('location_hour_rules.pkl', 'rb') as f:
            location_hour_rules = pickle.load(f)
        print(f'\n✓ Rules yüklendi: {len(segment_info)} segment, {len(location_hour_rules)} lokasyon')
    except:
        print('\n⚠️ Rules bulunamadı, oluşturuluyor...')
        train_df = analyze_training_distribution()
        location_hour_rules = create_enhanced_rules(train_df)
        with open('segment_info.pkl', 'rb') as f:
            segment_info = pickle.load(f)
    
    # Load models (fallback)
    enter_model = joblib.load('time_based_enter_model.pkl')
    exit_model = joblib.load('time_based_exit_model.pkl')
    label_encoders = joblib.load('time_based_label_encoders.pkl')
    feature_cols = joblib.load('time_based_features.pkl')
    
    le_enter = label_encoders['enter']
    le_exit = label_encoders['exit']
    
    # Load sample
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = sample_df['ID'].tolist()
    
    # Load train for signal patterns
    train_df = pd.read_csv('Train.csv')
    signal_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    location_signals = train_df.groupby('view_label')['signaling'].apply(
        lambda x: signal_map.get(x.mode()[0] if len(x.mode()) > 0 else 'none', 0)
    ).to_dict()
    
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
            
            hour = interpolated_minutes // 60
            minute = interpolated_minutes % 60
            
            return {
                'hour': hour,
                'minute': minute,
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
    
    # Generate predictions
    print('\n🔮 Tahminler oluşturuluyor (probabilistic rules)...')
    np.random.seed(42)  # For reproducibility
    
    submission_data = []
    rule_used = 0
    model_used = 0
    
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
            
            # Try probabilistic rules
            prediction = None
            if location_name in location_hour_rules:
                rules = location_hour_rules[location_name][rating_type]
                
                if hour in rules and len(rules[hour]) > 0:
                    # Sample from distribution
                    labels = list(rules[hour].keys())
                    probs = list(rules[hour].values())
                    prediction = np.random.choice(labels, p=probs)
                    rule_used += 1
            
            # Fallback to model
            if prediction is None:
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
                    prediction = le_enter.inverse_transform([pred_encoded])[0]
                else:
                    pred_encoded = exit_model.predict(X)[0]
                    prediction = le_exit.inverse_transform([pred_encoded])[0]
                
                model_used += 1
            
        except Exception as e:
            prediction = 'free flowing'
            model_used += 1
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission.csv', index=False)
    
    print(f'\n✅ Submission güncellendi: submission.csv')
    print(f'📊 Toplam: {len(submission_df):,} satır')
    print(f'🎯 Probabilistic rules: {rule_used} ({rule_used/len(required_ids)*100:.1f}%)')
    print(f'🔢 Model fallback: {model_used} ({model_used/len(required_ids)*100:.1f}%)')
    
    # Distribution
    print(f'\n📈 Tahmin Dağılımı:')
    dist = submission_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(submission_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    # Compare with training distribution
    print('\n📊 Training vs Submission Karşılaştırması:')
    train_enter_dist = train_df['congestion_enter_rating'].value_counts(normalize=True) * 100
    train_exit_dist = train_df['congestion_exit_rating'].value_counts(normalize=True) * 100
    
    # Average training distribution
    all_labels = set(list(train_enter_dist.index) + list(train_exit_dist.index))
    print('\nTraining Avg:')
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        if label in all_labels:
            enter_pct = train_enter_dist.get(label, 0)
            exit_pct = train_exit_dist.get(label, 0)
            avg_pct = (enter_pct + exit_pct) / 2
            print(f'  {label}: {avg_pct:.1f}%')
    
    return submission_df


if __name__ == '__main__':
    # Analyze training
    train_df = analyze_training_distribution()
    
    # Create rules
    location_hour_rules = create_enhanced_rules(train_df)
    
    # Generate submission
    submission = generate_improved_submission_v3()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 3 TAMAMLANDI')
    print('='*60)
