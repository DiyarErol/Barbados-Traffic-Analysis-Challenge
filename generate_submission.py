"""
Generate Submission File for Zindi Competition
Creates submission.csv with ID, Target, Target_Accuracy columns
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def generate_submission():
    """Generate submission file for test data"""
    
    print('='*60)
    print('🎯 ZINDI SUBMISSION GENERATOR')
    print('='*60)
    
    # Load sample submission to get required IDs
    print('\n📁 Sample submission yükleniyor...')
    sample_df = pd.read_csv('SampleSubmission.csv')
    required_ids = sample_df['ID'].tolist()
    print(f'✓ Gerekli ID sayısı: {len(required_ids):,}')
    
    # Load test data
    print('\n📁 Test verisi yükleniyor...')
    test_df = pd.read_csv('TestInputSegments.csv')
    print(f'✓ Toplam test kaydı: {len(test_df):,}')
    
    # Basic features
    test_df['datetime'] = pd.to_datetime(test_df['datetimestamp_start'])
    test_df['hour'] = test_df['datetime'].dt.hour
    test_df['day_of_week'] = test_df['datetime'].dt.dayofweek
    test_df['is_weekend'] = (test_df['day_of_week'] >= 5).astype(int)
    test_df['is_rush_hour'] = test_df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
    # Synthetic features (dummy values for missing columns)
    for col in ['vehicle_count', 'avg_speed', 'traffic_density']:
        if col not in test_df.columns:
            test_df[col] = 0
    
    test_df['vehicle_variance'] = 0
    test_df['speed_variance'] = 0
    
    # Load model
    print('\n🤖 Model yükleniyor...')
    try:
        enter_model = joblib.load('voting_ensemble_enter_model.pkl')
        exit_model = joblib.load('voting_ensemble_exit_model.pkl')
        model_name = 'Voting Ensemble'
        print(f'✓ Model yüklendi: {model_name}')
    except Exception as e:
        print(f'❌ Model yüklenemedi: {e}')
        return None
    
    # Feature columns
    feature_cols = ['vehicle_count', 'avg_speed', 'traffic_density', 
                    'vehicle_variance', 'speed_variance', 'hour', 
                    'is_rush_hour', 'day_of_week', 'is_weekend']
    
    X_test = test_df[feature_cols].fillna(0)
    
    # Make predictions
    print('\n🔮 Tahminler yapılıyor...')
    pred_enter = enter_model.predict(X_test)
    pred_exit = exit_model.predict(X_test)
    
    # Convert numeric predictions to labels
    congestion_map = {
        0: 'free flowing',
        1: 'light delay',
        2: 'moderate delay',
        3: 'heavy delay'
    }
    
    pred_enter_labels = [congestion_map.get(p, 'free flowing') for p in pred_enter]
    pred_exit_labels = [congestion_map.get(p, 'free flowing') for p in pred_exit]
    
    # Create ID to prediction mapping
    id_to_prediction = {}
    
    for idx, row in test_df.iterrows():
        id_to_prediction[row['ID_enter']] = pred_enter_labels[idx]
        id_to_prediction[row['ID_exit']] = pred_exit_labels[idx]
    
    # Create submission dataframe using required IDs from sample
    print('\n📝 Submission dosyası oluşturuluyor...')
    submission_data = []
    
    for req_id in required_ids:
        # Get prediction if available, otherwise use default
        if req_id in id_to_prediction:
            prediction = id_to_prediction[req_id]
        else:
            # Default prediction for missing IDs
            prediction = 'free flowing'
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction  # Same as Target for submission
        })
    
    submission_df = pd.DataFrame(submission_data)
    
    # Save submission file
    output_file = 'submission.csv'
    submission_df.to_csv(output_file, index=False)
    
    print(f'\n✅ Submission dosyası oluşturuldu: {output_file}')
    print(f'📊 Toplam satır: {len(submission_df):,}')
    print(f'📋 Kolon sayısı: {len(submission_df.columns)}')
    
    # Show prediction distribution
    print(f'\n📈 Tahmin Dağılımı:')
    target_dist = submission_df['Target'].value_counts()
    for label, count in target_dist.items():
        pct = count / len(submission_df) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    # Check coverage
    matched_ids = sum(1 for req_id in required_ids if req_id in id_to_prediction)
    print(f'\n✓ Tahmin Kapsamı:')
    print(f'   Test verisinde bulunan: {matched_ids:,} / {len(required_ids):,} ({matched_ids/len(required_ids)*100:.1f}%)')
    print(f'   Default kullanılan: {len(required_ids) - matched_ids:,}')
    
    # Show sample
    print(f'\n📋 İlk 10 Satır:')
    print(submission_df.head(10).to_string(index=False))
    
    # Verify format matches sample
    print(f'\n✅ Format Kontrolü:')
    sample_df = pd.read_csv('SampleSubmission.csv')
    print(f'   Sample sütunlar: {list(sample_df.columns)}')
    print(f'   Submission sütunlar: {list(submission_df.columns)}')
    print(f'   Format eşleşiyor: {list(sample_df.columns) == list(submission_df.columns)}')
    
    return submission_df


if __name__ == '__main__':
    submission = generate_submission()
    
    if submission is not None:
        print('\n' + '='*60)
        print('✅ SUBMISSION HAZIR!')
        print('='*60)
        print('\n💡 Sonraki adım:')
        print('   1. submission.csv dosyasını Zindi\'ye yükleyin')
        print('   2. https://zindi.africa/competitions/barbados-traffic-analysis-challenge/submissions')
    else:
        print('\n' + '='*60)
        print('❌ SUBMISSION OLUŞTURULAMADI!')
        print('='*60)
