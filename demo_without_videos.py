"""
Video dosyaları olmadan test etmek için basitleştirilmiş demo
Sentetik özellikler kullanır
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("BARBADOS TRAFFIC ANALYSIS - BASİTLEŞTİRİLMİŞ DEMO")
print("(Video dosyası gerektirmez - Sentetik özellikler kullanır)")
print("=" * 80)

# 1. Veriyi yükle
print("\n1. Veri yükleniyor...")
train_df = pd.read_csv('Train.csv')
print(f"   Toplam eğitim örnekleri: {len(train_df)}")

# 2. Sentetik özellikler oluştur (video yerine)
print("\n2. Sentetik özellikler oluşturuluyor...")
print("   (Gerçek uygulamada video işleme yapılacak)")

np.random.seed(42)

# Zaman özellikleri
train_df['datetime'] = pd.to_datetime(train_df['video_time'])
train_df['hour'] = train_df['datetime'].dt.hour
train_df['minute'] = train_df['datetime'].dt.minute
train_df['day_of_week'] = train_df['datetime'].dt.dayofweek

# Rush hour
train_df['is_rush_hour'] = train_df['hour'].apply(
    lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0
)

# Sinyal encoding
signal_mapping = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
train_df['signaling_encoded'] = train_df['signaling'].map(signal_mapping).fillna(0)

# Sentetik video özellikleri (gerçekte video işlemeden gelecek)
print("   Sentetik video özellikleri ekleniyor...")

# Tıkanıklığa göre farklı değerler üret
def generate_synthetic_features(congestion_level):
    """Tıkanıklık seviyesine göre gerçekçi sentetik özellikler"""
    if congestion_level == 'free flowing':
        vehicle_count = np.random.uniform(5, 15)
        density = np.random.uniform(0.1, 0.3)
        movement = np.random.uniform(0.5, 0.8)
    elif congestion_level == 'light delay':
        vehicle_count = np.random.uniform(12, 25)
        density = np.random.uniform(0.25, 0.45)
        movement = np.random.uniform(0.3, 0.6)
    elif congestion_level == 'moderate delay':
        vehicle_count = np.random.uniform(20, 35)
        density = np.random.uniform(0.4, 0.6)
        movement = np.random.uniform(0.2, 0.4)
    else:  # heavy delay
        vehicle_count = np.random.uniform(30, 50)
        density = np.random.uniform(0.55, 0.8)
        movement = np.random.uniform(0.05, 0.25)
    
    return {
        'vehicle_count_mean': vehicle_count,
        'vehicle_count_max': vehicle_count * 1.3,
        'vehicle_count_std': vehicle_count * 0.2,
        'density_mean': density,
        'density_max': density * 1.2,
        'movement_mean': movement,
        'movement_std': movement * 0.3
    }

# Her satır için sentetik özellikler üret
synthetic_features = []
for idx, row in train_df.iterrows():
    features = generate_synthetic_features(row['congestion_enter_rating'])
    features['time_segment_id'] = row['time_segment_id']
    synthetic_features.append(features)
    
    if (idx + 1) % 2000 == 0:
        print(f"   İşlendi: {idx + 1}/{len(train_df)}")

synthetic_df = pd.DataFrame(synthetic_features)
train_df = train_df.merge(synthetic_df, on='time_segment_id', how='left')

# Döngüsel özellikler
train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)

# Lagged features (basit versiyon)
feature_cols = ['vehicle_count_mean', 'density_mean', 'movement_mean']
for col in feature_cols:
    train_df[f'{col}_lag_1'] = train_df[col].shift(1)
    train_df[f'{col}_lag_2'] = train_df[col].shift(2)

# Rolling features
for col in feature_cols:
    train_df[f'{col}_rolling_mean_5'] = train_df[col].rolling(window=5, min_periods=1).mean()
    train_df[f'{col}_rolling_std_5'] = train_df[col].rolling(window=5, min_periods=1).std()

train_df = train_df.fillna(0)

print(f"   Toplam özellik sayısı: {len([c for c in train_df.columns if c not in ['responseId', 'view_label', 'ID_enter', 'ID_exit', 'videos', 'video_time', 'datetimestamp_start', 'datetimestamp_end', 'date', 'congestion_enter_rating', 'congestion_exit_rating', 'cycle_phase', 'datetime', 'signaling']])}")

# 3. Model eğitimi
print("\n3. Model eğitimi başlıyor...")

# Özellik sütunları
exclude_cols = [
    'responseId', 'view_label', 'ID_enter', 'ID_exit', 'videos',
    'video_time', 'datetimestamp_start', 'datetimestamp_end',
    'date', 'congestion_enter_rating', 'congestion_exit_rating',
    'cycle_phase', 'datetime', 'signaling'
]

feature_columns = [col for col in train_df.columns if col not in exclude_cols]

X = train_df[feature_columns].values
le = LabelEncoder()
y_enter = le.fit_transform(train_df['congestion_enter_rating'])
y_exit = le.transform(train_df['congestion_exit_rating'])

# Train-test split
X_train, X_test, y_train_enter, y_test_enter = train_test_split(
    X, y_enter, test_size=0.2, random_state=42, stratify=y_enter
)
_, _, y_train_exit, y_test_exit = train_test_split(
    X, y_exit, test_size=0.2, random_state=42, stratify=y_exit
)

print(f"   Eğitim seti: {len(X_train)} örnek")
print(f"   Test seti: {len(X_test)} örnek")
print(f"   Özellik sayısı: {X_train.shape[1]}")

# Enter modeli
print("\n   Enter modeli eğitiliyor...")
model_enter = GradientBoostingClassifier(
    n_estimators=100,  # Demo için daha az
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    subsample=0.8,
    verbose=0
)
model_enter.fit(X_train, y_train_enter)

y_pred_enter = model_enter.predict(X_test)
acc_enter = accuracy_score(y_test_enter, y_pred_enter)

print(f"   ✓ Enter modeli eğitildi - Test Accuracy: {acc_enter:.4f}")

# Exit modeli
print("\n   Exit modeli eğitiliyor...")
model_exit = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42,
    subsample=0.8,
    verbose=0
)
model_exit.fit(X_train, y_train_exit)

y_pred_exit = model_exit.predict(X_test)
acc_exit = accuracy_score(y_test_exit, y_pred_exit)

print(f"   ✓ Exit modeli eğitildi - Test Accuracy: {acc_exit:.4f}")

# 4. Detaylı sonuçlar
print("\n" + "=" * 80)
print("SONUÇLAR")
print("=" * 80)

print("\nEnter Congestion Classification Report:")
print(classification_report(
    y_test_enter, y_pred_enter,
    target_names=le.classes_,
    zero_division=0
))

print("\nExit Congestion Classification Report:")
print(classification_report(
    y_test_exit, y_pred_exit,
    target_names=le.classes_,
    zero_division=0
))

# 5. En önemli özellikler
print("\n" + "=" * 80)
print("EN ÖNEMLİ 10 ÖZELLİK")
print("=" * 80)

feature_importance = sorted(
    zip(feature_columns, model_enter.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nEnter için:")
for i, (feat, imp) in enumerate(feature_importance, 1):
    print(f"{i:2d}. {feat:40s} : {imp:.4f}")

print("\n" + "=" * 80)
print("ÖZET")
print("=" * 80)
print(f"""
✅ Model Başarıyla Eğitildi!

Performans:
- Enter Accuracy: {acc_enter:.2%}
- Exit Accuracy:  {acc_exit:.2%}

Veri:
- Eğitim örnekleri: {len(X_train)}
- Test örnekleri: {len(X_test)}
- Özellik sayısı: {X_train.shape[1]}

NOT: 
- Bu demo SENTETİK özellikler kullanır
- Gerçek uygulamada video işleme yapılmalıdır
- Video dosyaları hazırlandığında traffic_analysis_solution.py çalıştırın

Sonraki Adımlar:
1. Video dosyalarını videos/ klasörüne koyun
2. python traffic_analysis_solution.py (gerçek eğitim)
3. python test_prediction.py (tahmin)
4. python analyze_results.py (analiz)
""")

print("=" * 80)
