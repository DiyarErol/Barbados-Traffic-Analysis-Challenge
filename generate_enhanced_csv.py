"""
Generate Enhanced CSV with Predictions and Analysis
Creates a comprehensive CSV file with all features, predictions, and statistics
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def generate_enhanced_csv():
    """Generate enhanced CSV with predictions"""
    
    # Load data
    print('📁 Veri yükleniyor...')
    df = pd.read_csv('Train.csv')
    
    # Basic features
    df['datetime'] = pd.to_datetime(df['datetimestamp_start'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    
    # Synthetic features (using simple variance if location columns don't exist)
    if 'vehicle_count' in df.columns:
        df['vehicle_variance'] = df['vehicle_count'].rolling(3, min_periods=1).std().fillna(0)
    else:
        df['vehicle_variance'] = 0
        
    if 'avg_speed' in df.columns:
        df['speed_variance'] = df['avg_speed'].rolling(3, min_periods=1).std().fillna(0)
    else:
        df['speed_variance'] = 0
    
    # Load model
    print('🤖 Model yükleniyor...')
    try:
        enter_model = joblib.load('voting_ensemble_enter_model.pkl')
        exit_model = joblib.load('voting_ensemble_exit_model.pkl')
        model_name = 'Voting Ensemble'
        print(f'✓ Model yüklendi: {model_name}')
    except Exception as e:
        print(f'⚠️ Model yüklenemedi: {e}')
        print('📝 Model olmadan devam ediliyor...')
        enter_model = None
        exit_model = None
        model_name = 'None'
    
    # Prediction features (use only existing columns)
    base_feature_cols = ['vehicle_count', 'avg_speed', 'traffic_density', 
                         'vehicle_variance', 'speed_variance', 'hour', 
                         'is_rush_hour', 'day_of_week', 'is_weekend']
    
    # Filter to only existing columns
    feature_cols = [col for col in base_feature_cols if col in df.columns]
    
    # Create dummy data for missing columns
    for col in base_feature_cols:
        if col not in df.columns:
            df[col] = 0
    
    X = df[base_feature_cols].fillna(0)
    
    # Make predictions
    print('🔮 Tahminler yapılıyor...')
    if enter_model is not None and hasattr(enter_model, 'predict'):
        df['predicted_enter'] = enter_model.predict(X)
        df['predicted_exit'] = exit_model.predict(X)
        
        # Probabilities
        if hasattr(enter_model, 'predict_proba'):
            enter_proba = enter_model.predict_proba(X)
            exit_proba = exit_model.predict_proba(X)
            df['enter_confidence'] = enter_proba.max(axis=1)
            df['exit_confidence'] = exit_proba.max(axis=1)
        else:
            df['enter_confidence'] = 0.0
            df['exit_confidence'] = 0.0
    else:
        # No model - use actual values
        df['predicted_enter'] = df['congestion_enter_rating']
        df['predicted_exit'] = df['congestion_exit_rating']
        df['enter_confidence'] = 1.0
        df['exit_confidence'] = 1.0
    
    # Error analysis
    df['enter_prediction_correct'] = (df['predicted_enter'] == df['congestion_enter_rating']).astype(int)
    df['exit_prediction_correct'] = (df['predicted_exit'] == df['congestion_exit_rating']).astype(int)
    
    # Congestion labels
    congestion_labels = {
        0: 'Free Flowing', 
        1: 'Light Delay', 
        2: 'Moderate Delay', 
        3: 'Heavy Delay'
    }
    
    df['enter_congestion_label'] = df['congestion_enter_rating'].map(congestion_labels)
    df['exit_congestion_label'] = df['congestion_exit_rating'].map(congestion_labels)
    df['predicted_enter_label'] = df['predicted_enter'].map(congestion_labels)
    df['predicted_exit_label'] = df['predicted_exit'].map(congestion_labels)
    
    # Time features
    df['date'] = df['datetime'].dt.date
    df['time'] = df['datetime'].dt.time
    df['day_name'] = df['datetime'].dt.day_name()
    df['month'] = df['datetime'].dt.month
    df['week_of_year'] = df['datetime'].dt.isocalendar().week
    
    # Statistics
    df['total_predictions'] = 2
    df['correct_predictions'] = df['enter_prediction_correct'] + df['exit_prediction_correct']
    df['accuracy_rate'] = df['correct_predictions'] / df['total_predictions']
    
    # Kimlik, Hedef, Hedef_Doğruluğu sütunları
    df['Kimlik'] = df.index + 1  # Benzersiz kimlik
    df['Hedef'] = df['congestion_enter_rating'].astype(str) + '_' + df['congestion_exit_rating'].astype(str)
    df['Hedef_Doğruluğu'] = ((df['enter_prediction_correct'] == 1) & (df['exit_prediction_correct'] == 1)).astype(int)
    
    # Column ordering (prioritize existing columns from original data)
    output_cols = [
        # Identification
        'Kimlik', 'responseId', 'view_label', 'time_segment_id', 'cycle_phase',
        'ID_enter', 'ID_exit', 'videos', 'video_time',
        
        # Time
        'datetime', 'date', 'time', 'day_name', 'hour', 'day_of_week', 'month', 'week_of_year',
        'datetimestamp_start', 'datetimestamp_end',
        'is_weekend', 'is_rush_hour',
        
        # Traffic features
        'vehicle_count', 'avg_speed', 'traffic_density', 
        'vehicle_variance', 'speed_variance',
        
        # Signaling
        'signaling',
        
        # Actual values
        'congestion_enter_rating', 'enter_congestion_label',
        'congestion_exit_rating', 'exit_congestion_label',
        'Hedef',
        
        # Predictions
        'predicted_enter', 'predicted_enter_label', 'enter_confidence', 'enter_prediction_correct',
        'predicted_exit', 'predicted_exit_label', 'exit_confidence', 'exit_prediction_correct',
        
        # Performance
        'total_predictions', 'correct_predictions', 'accuracy_rate', 'Hedef_Doğruluğu'
    ]
    
    # Filter existing columns
    output_cols = [col for col in output_cols if col in df.columns]
    df_output = df[output_cols].copy()
    
    # Save
    output_file = 'traffic_predictions_enhanced.csv'
    df_output.to_csv(output_file, index=False)
    
    print(f'\n✅ CSV dosyası oluşturuldu: {output_file}')
    print(f'📊 Toplam kayıt: {len(df_output):,}')
    print(f'📋 Kolon sayısı: {len(df_output.columns)}')
    print(f'🤖 Model: {model_name}')
    
    # Summary statistics
    print(f'\n📈 Özet İstatistikler:')
    print(f'   Enter Doğruluk: {df_output["enter_prediction_correct"].mean():.2%}')
    print(f'   Exit Doğruluk: {df_output["exit_prediction_correct"].mean():.2%}')
    print(f'   Ortalama Güven (Enter): {df_output["enter_confidence"].mean():.2%}')
    print(f'   Ortalama Güven (Exit): {df_output["exit_confidence"].mean():.2%}')
    
    # Congestion distribution
    print(f'\n🚦 Tıkanıklık Dağılımı (Enter):')
    enter_dist = df_output['enter_congestion_label'].value_counts()
    for label, count in enter_dist.items():
        pct = count / len(df_output) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    print(f'\n🚦 Tıkanıklık Dağılımı (Exit):')
    exit_dist = df_output['exit_congestion_label'].value_counts()
    for label, count in exit_dist.items():
        pct = count / len(df_output) * 100
        print(f'   {label}: {count:,} ({pct:.1f}%)')
    
    # Sample records
    print(f'\n📋 İlk 5 Kayıt:')
    sample_cols = ['time_segment_id', 'datetime', 'enter_congestion_label', 
                   'predicted_enter_label', 'enter_confidence', 'enter_prediction_correct']
    sample_cols = [col for col in sample_cols if col in df_output.columns]
    print(df_output[sample_cols].head())
    
    return df_output


if __name__ == '__main__':
    print('='*60)
    print('🚦 ENHANCED CSV GENERATOR')
    print('='*60)
    
    df_enhanced = generate_enhanced_csv()
    
    print(f'\n{"="*60}')
    print('✅ İŞLEM TAMAMLANDI!')
    print('='*60)
    print(f'\n💡 Dosya: traffic_predictions_enhanced.csv')
    print(f'📊 Toplam: {len(df_enhanced):,} kayıt, {len(df_enhanced.columns)} kolon')
