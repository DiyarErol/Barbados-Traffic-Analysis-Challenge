"""
Development 1: Improved Time Mapping from Segment IDs
Creates accurate time mapping using train data patterns
"""

import pandas as pd
import numpy as np
import joblib

def create_segment_time_mapping():
    """Create mapping from segment_id to actual time"""
    
    print('🔍 GELİŞTİRME 1: Segment-Zaman Mapping İyileştirmesi')
    print('='*60)
    
    # Load train data
    train_df = pd.read_csv('Train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
    
    # Create mapping: segment_id -> (hour, minute, day_of_week)
    segment_mapping = {}
    
    for _, row in train_df.iterrows():
        seg_id = row['time_segment_id']
        dt = row['datetime']
        
        if seg_id not in segment_mapping:
            segment_mapping[seg_id] = {
                'hour': dt.hour,
                'minute': dt.minute,
                'day_of_week': dt.dayofweek,
                'date': dt.date()
            }
    
    print(f'✓ {len(segment_mapping)} segment için gerçek zaman bulundu')
    print(f'✓ Segment aralığı: {min(segment_mapping.keys())} - {max(segment_mapping.keys())}')
    
    # Save mapping
    import pickle
    with open('segment_time_mapping.pkl', 'wb') as f:
        pickle.dump(segment_mapping, f)
    
    print('✓ segment_time_mapping.pkl kaydedildi')
    
    # Show sample
    print('\n📋 Örnek Mapping:')
    for seg_id in list(segment_mapping.keys())[:5]:
        info = segment_mapping[seg_id]
        print(f'  Segment {seg_id}: {info["hour"]:02d}:{info["minute"]:02d}, Gün {info["day_of_week"]}')
    
    return segment_mapping


def generate_improved_submission_v1():
    """Generate submission with improved time mapping"""
    
    print('\n' + '='*60)
    print('📝 SUBMISSION V1: İyileştirilmiş Zaman Mapping')
    print('='*60)
    
    # Load segment mapping
    try:
        import pickle
        with open('segment_time_mapping.pkl', 'rb') as f:
            segment_mapping = pickle.load(f)
        print(f'\n✓ Segment mapping yüklendi: {len(segment_mapping)} segment')
    except:
        print('\n⚠️ Mapping bulunamadı, oluşturuluyor...')
        segment_mapping = create_segment_time_mapping()
    
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
    print('\n🔮 Tahminler oluşturuluyor (geliştirilmiş zaman bilgisi ile)...')
    submission_data = []
    mapped_count = 0
    
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
            
            # Use real mapping if available
            if segment_id in segment_mapping:
                time_info = segment_mapping[segment_id]
                hour = time_info['hour']
                minute = time_info['minute']
                day_of_week = time_info['day_of_week']
                mapped_count += 1
            else:
                # Fallback to estimation
                total_minutes = segment_id
                hour = (total_minutes // 60) % 24
                minute = total_minutes % 60
                day_of_week = (total_minutes // 1440) % 7
            
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
            
        except Exception as e:
            prediction = 'free flowing'
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission.csv', index=False)
    
    print(f'\n✅ Submission güncellendi: submission.csv')
    print(f'📊 Toplam: {len(submission_df):,} satır')
    print(f'🎯 Gerçek mapping kullanıldı: {mapped_count} / {len(required_ids)} (%{mapped_count/len(required_ids)*100:.1f})')
    
    # Distribution
    print(f'\n📈 Tahmin Dağılımı:')
    dist = submission_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(submission_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    return submission_df


if __name__ == '__main__':
    # Create mapping
    segment_mapping = create_segment_time_mapping()
    
    # Generate submission
    submission = generate_improved_submission_v1()
    
    print('\n' + '='*60)
    print('✅ GELİŞTİRME 1 TAMAMLANDI')
    print('='*60)
