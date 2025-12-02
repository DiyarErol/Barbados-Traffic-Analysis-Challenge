"""
Development 2: Advanced Temporal Pattern Learning
Uses location-specific time patterns and segment interpolation
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

def extract_temporal_patterns():
    """Extract location-specific temporal patterns from training data"""
    
    print('🔍 GELİŞTİRME 2: Lokasyon-Spesifik Temporal Pattern')
    print('='*60)
    
    # Load train data
    train_df = pd.read_csv('Train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
    train_df['hour'] = train_df['datetime'].dt.hour
    train_df['minute'] = train_df['datetime'].dt.minute
    train_df['day_of_week'] = train_df['datetime'].dt.dayofweek
    
    # Create detailed segment mapping
    segment_info = {}
    for _, row in train_df.iterrows():
        seg_id = row['time_segment_id']
        segment_info[seg_id] = {
            'hour': row['hour'],
            'minute': row['minute'],
            'day_of_week': row['day_of_week'],
            'location': row['view_label']
        }
    
    print(f'✓ {len(segment_info)} segment bilgisi toplandı')
    print(f'✓ Segment range: {min(segment_info.keys())} - {max(segment_info.keys())}')
    
    # Location-time congestion patterns
    print('\n📊 Lokasyon-Zaman-Congestion Pattern Analizi...')
    
    location_time_patterns = {}
    
    for location in train_df['view_label'].unique():
        loc_data = train_df[train_df['view_label'] == location]
        
        # For each hour, what's the most common congestion?
        hour_patterns_enter = {}
        hour_patterns_exit = {}
        
        for hour in range(24):
            hour_data = loc_data[loc_data['hour'] == hour]
            
            if len(hour_data) > 0:
                # Enter patterns
                enter_data = hour_data[hour_data['congestion_enter_rating'].notna()]
                if len(enter_data) > 0:
                    hour_patterns_enter[hour] = enter_data['congestion_enter_rating'].mode()[0]
                else:
                    hour_patterns_enter[hour] = 'free flowing'
                
                # Exit patterns
                exit_data = hour_data[hour_data['congestion_exit_rating'].notna()]
                if len(exit_data) > 0:
                    hour_patterns_exit[hour] = exit_data['congestion_exit_rating'].mode()[0]
                else:
                    hour_patterns_exit[hour] = 'free flowing'
        
        location_time_patterns[location] = {
            'enter': hour_patterns_enter,
            'exit': hour_patterns_exit
        }
    
    print(f'✓ {len(location_time_patterns)} lokasyon için pattern çıkarıldı')
    
    # Show sample
    print('\n📋 Örnek Pattern (Norman Niles #1):')
    if 'Norman Niles #1' in location_time_patterns:
        patterns = location_time_patterns['Norman Niles #1']
        print('  Enter patterns (rush hours):')
        for hour in [7, 8, 9, 16, 17, 18]:
            if hour in patterns['enter']:
                print(f'    {hour:02d}:00 -> {patterns["enter"][hour]}')
    
    # Save patterns
    import pickle
    with open('segment_info.pkl', 'wb') as f:
        pickle.dump(segment_info, f)
    with open('location_time_patterns.pkl', 'wb') as f:
        pickle.dump(location_time_patterns, f)
    
    print('\n✓ segment_info.pkl kaydedildi')
    print('✓ location_time_patterns.pkl kaydedildi')
    
    return segment_info, location_time_patterns


def interpolate_segment_time(segment_id, segment_info):
    """Interpolate time for segments not in training data"""
    
    known_segments = sorted(segment_info.keys())
    
    if segment_id in segment_info:
        return segment_info[segment_id]
    
    # Find nearest segments
    lower = [s for s in known_segments if s < segment_id]
    upper = [s for s in known_segments if s > segment_id]
    
    if lower and upper:
        # Interpolate between two known segments
        seg_lower = max(lower)
        seg_upper = min(upper)
        
        info_lower = segment_info[seg_lower]
        info_upper = segment_info[seg_upper]
        
        # Calculate position ratio
        ratio = (segment_id - seg_lower) / (seg_upper - seg_lower)
        
        # Interpolate time
        lower_minutes = info_lower['hour'] * 60 + info_lower['minute']
        upper_minutes = info_upper['hour'] * 60 + info_upper['minute']
        
        # Handle day wraparound
        if upper_minutes < lower_minutes:
            upper_minutes += 1440  # Add 24 hours
        
        interpolated_minutes = int(lower_minutes + ratio * (upper_minutes - lower_minutes))
        interpolated_minutes = interpolated_minutes % 1440  # Wrap around
        
        hour = interpolated_minutes // 60
        minute = interpolated_minutes % 60
        
        # Use day_of_week from nearest segment
        day_of_week = info_lower['day_of_week']
        
        return {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'location': info_lower.get('location', 'Norman Niles #1')
        }
    
    elif lower:
        # Extrapolate from last known
        seg_lower = max(lower)
        info = segment_info[seg_lower].copy()
        diff = segment_id - seg_lower
        total_minutes = info['hour'] * 60 + info['minute'] + diff
        info['hour'] = (total_minutes // 60) % 24
        info['minute'] = total_minutes % 60
        return info
    
    elif upper:
        # Extrapolate from first known
        seg_upper = min(upper)
        info = segment_info[seg_upper].copy()
        diff = seg_upper - segment_id
        total_minutes = info['hour'] * 60 + info['minute'] - diff
        if total_minutes < 0:
            total_minutes += 1440
        info['hour'] = (total_minutes // 60) % 24
        info['minute'] = total_minutes % 60
        return info
    
    else:
        # No known segments, use default
        return {
            'hour': 12,
            'minute': 0,
            'day_of_week': 0,
            'location': 'Norman Niles #1'
        }


def generate_improved_submission_v2():
    """Generate submission with temporal patterns and interpolation"""
    
    print('\n' + '='*60)
    print('📝 SUBMISSION V2: Temporal Pattern + Interpolation')
    print('='*60)
    
    # Load patterns
    import pickle
    try:
        with open('segment_info.pkl', 'rb') as f:
            segment_info = pickle.load(f)
        with open('location_time_patterns.pkl', 'rb') as f:
            location_time_patterns = pickle.load(f)
        print(f'\n✓ Patterns yüklendi: {len(segment_info)} segment, {len(location_time_patterns)} lokasyon')
    except:
        print('\n⚠️ Patterns bulunamadı, oluşturuluyor...')
        segment_info, location_time_patterns = extract_temporal_patterns()
    
    # Load models
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
    
    # Generate predictions
    print('\n🔮 Tahminler oluşturuluyor (temporal pattern + interpolation)...')
    submission_data = []
    pattern_used = 0
    interpolated = 0
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
            
            # Get time info with interpolation
            time_info = interpolate_segment_time(segment_id, segment_info)
            hour = time_info['hour']
            minute = time_info['minute']
            day_of_week = time_info['day_of_week']
            
            if segment_id in segment_info:
                interpolated += 0  # Direct match
            else:
                interpolated += 1
            
            # Try location-time pattern first
            prediction = None
            if location_name in location_time_patterns:
                patterns = location_time_patterns[location_name]
                
                if rating_type == 'enter' and hour in patterns['enter']:
                    prediction = patterns['enter'][hour]
                    pattern_used += 1
                elif rating_type == 'exit' and hour in patterns['exit']:
                    prediction = patterns['exit'][hour]
                    pattern_used += 1
            
            # If no pattern match, use model
            if prediction is None:
                # Features
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
    print(f'🎯 Pattern kullanıldı: {pattern_used} ({pattern_used/len(required_ids)*100:.1f}%)')
    print(f'🔢 Model kullanıldı: {model_used} ({model_used/len(required_ids)*100:.1f}%)')
    print(f'📐 Interpolasyon: {interpolated} segment')
    
    # Distribution
    print(f'\n📈 Tahmin Dağılımı:')
    dist = submission_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(submission_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    return submission_df


if __name__ == '__main__':
    # Extract patterns
    segment_info, location_time_patterns = extract_temporal_patterns()
    
    # Generate submission
    submission = generate_improved_submission_v2()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 2 TAMAMLANDI')
    print('='*60)
