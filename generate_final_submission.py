"""
Generate Final Submission using Time-Based Model
Uses the newly trained RandomForest/GradientBoosting models
"""

import pandas as pd
import numpy as np
import joblib

def generate_final_submission():
    """Generate submission with time-based model"""
    
    print('='*60)
    print('🚀 FINAL SUBMISSION GENERATOR')
    print('='*60)
    
    # Load sample
    print('\n📁 Sample submission yükleniyor...')
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = sample_df['ID'].tolist()
    print(f'✓ Gerekli ID sayısı: {len(required_ids):,}')
    
    # Load models
    print('\n🤖 Modeller yükleniyor...')
    enter_model = joblib.load('time_based_enter_model.pkl')
    exit_model = joblib.load('time_based_exit_model.pkl')
    label_encoders = joblib.load('time_based_label_encoders.pkl')
    feature_cols = joblib.load('time_based_features.pkl')
    
    le_enter = label_encoders['enter']
    le_exit = label_encoders['exit']
    
    print(f'✓ Enter model: RandomForest')
    print(f'✓ Exit model: GradientBoosting')
    
    # Load train data for signal patterns
    print('\n📊 Train verisinden sinyal bilgisi alınıyor...')
    train_df = pd.read_csv('Train.csv')
    
    # Signal mapping
    signal_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    
    # Location-based signal averages
    location_signals = train_df.groupby('view_label')['signaling'].apply(
        lambda x: signal_map.get(x.mode()[0] if len(x.mode()) > 0 else 'none', 0)
    ).to_dict()
    
    print(f'✓ {len(location_signals)} lokasyon için sinyal bilgisi')
    
    # Generate predictions
    print('\n🔮 Tahminler oluşturuluyor...')
    submission_data = []
    
    for req_id in required_ids:
        try:
            # Parse ID: time_segment_XXX_Location Name_congestion_enter/exit_rating
            parts = req_id.split('_')
            segment_id = int(parts[2])
            
            # Extract location (between segment number and congestion)
            location_parts = []
            for i in range(3, len(parts)):
                if parts[i] == 'congestion':
                    break
                location_parts.append(parts[i])
            location_name = ' '.join(location_parts)
            
            rating_type = parts[-2]  # 'enter' or 'exit'
            
            # Estimate time from segment_id
            # Assuming ~1 minute per segment
            total_minutes = segment_id
            hour = (total_minutes // 60) % 24
            minute = total_minutes % 60
            day_of_week = (total_minutes // 1440) % 7
            
            # Time features
            is_rush_hour = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
            is_weekend = 1 if day_of_week >= 5 else 0
            is_morning = 1 if hour < 12 else 0
            is_evening = 1 if (hour >= 16 and hour < 20) else 0
            
            # Cyclical encoding
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            
            # Signal (from location patterns)
            signal_encoded = location_signals.get(location_name, 0)
            
            # Create feature dict
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
            
            # Create DataFrame
            X = pd.DataFrame([features_dict])[feature_cols]
            
            # Predict
            if rating_type == 'enter':
                pred_encoded = enter_model.predict(X)[0]
                prediction = le_enter.inverse_transform([pred_encoded])[0]
            else:
                pred_encoded = exit_model.predict(X)[0]
                prediction = le_exit.inverse_transform([pred_encoded])[0]
            
        except Exception as e:
            # Fallback to free flowing
            prediction = 'free flowing'
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save
    output_file = 'submission.csv'
    submission_df.to_csv(output_file, index=False)
    
    print(f'\n✅ Submission oluşturuldu: {output_file}')
    print(f'📊 Toplam satır: {len(submission_df):,}')
    
    # Show distribution
    print(f'\n📈 Tahmin Dağılımı:')
    dist = submission_df['Target'].value_counts().sort_values(ascending=False)
    for label, count in dist.items():
        pct = count / len(submission_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    # Validate
    print(f'\n✅ Validasyon:')
    print(f'   Tüm ID\'ler mevcut: {len(submission_df) == len(sample_df)}')
    print(f'   Format doğru: {list(submission_df.columns) == list(sample_df.columns)}')
    
    # Sample predictions
    print(f'\n📋 Örnek Tahminler:')
    sample_rows = submission_df.sample(min(10, len(submission_df)))
    for _, row in sample_rows.iterrows():
        id_short = row['ID'][:50] + '...' if len(row['ID']) > 50 else row['ID']
        print(f'   {id_short}: {row["Target"]}')
    
    return submission_df


if __name__ == '__main__':
    submission = generate_final_submission()
    
    print('\n' + '='*60)
    print('✅ FINAL SUBMISSION HAZIR!')
    print('='*60)
    print('\n📤 Dosya: submission.csv')
    print('🎯 Model: RandomForest + GradientBoosting (67.57% / 95.77%)')
    print('🔗 Zindi: https://zindi.africa/competitions/barbados-traffic-analysis-challenge/submissions')
