"""
Simplified demo for testing without video files
Uses synthetic features
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
print("BARBADOS TRAFFIC ANALYSIS - SIMPLIFIED DEMO")
print("(No video files required - Uses synthetic features)")
print("=" * 80)

# 1. Load data
print("\n1. Loading data...")
train_df = pd.read_csv('Train.csv')
print(f"   Total training samples: {len(train_df)}")

# 2. Create synthetic features (instead of video)
print("\n2. Creating synthetic features...")
print("   (In production, video processing will be used)")

np.random.seed(42)

# Time features
train_df['datetime'] = pd.to_datetime(train_df['video_time'])
train_df['hour'] = train_df['datetime'].dt.hour
train_df['minute'] = train_df['datetime'].dt.minute
train_df['day_of_week'] = train_df['datetime'].dt.dayofweek

# Rush hour
train_df['is_rush_hour'] = train_df['hour'].apply(
    lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0
)

# Signal encoding
signal_mapping = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
train_df['signaling_encoded'] = train_df['signaling'].map(signal_mapping).fillna(0)

# Synthetic video features (would come from video processing)
print("   Adding synthetic video features...")

# Tıkanıklığa göre farklı değerler üret
def generate_synthetic_features(congestion_level):
    """Realistic synthetic features based on congestion level"""
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

# Generate synthetic features per row
synthetic_features = []
for idx, row in train_df.iterrows():
    features = generate_synthetic_features(row['congestion_enter_rating'])
    features['time_segment_id'] = row['time_segment_id']
    synthetic_features.append(features)
    
    if (idx + 1) % 2000 == 0:
        print(f"   Processed: {idx + 1}/{len(train_df)}")

synthetic_df = pd.DataFrame(synthetic_features)
train_df = train_df.merge(synthetic_df, on='time_segment_id', how='left')

# Cyclical features
train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)

# Lagged features (simple version)
feature_cols = ['vehicle_count_mean', 'density_mean', 'movement_mean']
for col in feature_cols:
    train_df[f'{col}_lag_1'] = train_df[col].shift(1)
    train_df[f'{col}_lag_2'] = train_df[col].shift(2)

# Rolling features
for col in feature_cols:
    train_df[f'{col}_rolling_mean_5'] = train_df[col].rolling(window=5, min_periods=1).mean()
    train_df[f'{col}_rolling_std_5'] = train_df[col].rolling(window=5, min_periods=1).std()

train_df = train_df.fillna(0)

print(f"   Total feature count: {len([c for c in train_df.columns if c not in ['responseId', 'view_label', 'ID_enter', 'ID_exit', 'videos', 'video_time', 'datetimestamp_start', 'datetimestamp_end', 'date', 'congestion_enter_rating', 'congestion_exit_rating', 'cycle_phase', 'datetime', 'signaling']])}")

# 3. Model eğitimi
print("\n3. Starting model training...")

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

print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")
print(f"   Feature count: {X_train.shape[1]}")

# Enter modeli
print("\n   Training Enter model...")
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

print(f"   ✓ Enter model trained - Test Accuracy: {acc_enter:.4f}")

# Exit modeli
print("\n   Training Exit model...")
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

print(f"   ✓ Exit model trained - Test Accuracy: {acc_exit:.4f}")

# 4. Detaylı sonuçlar
print("\n" + "=" * 80)
print("RESULTS")
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
print("TOP 10 FEATURES")
print("=" * 80)

feature_importance = sorted(
    zip(feature_columns, model_enter.feature_importances_),
    key=lambda x: x[1],
    reverse=True
)[:10]

print("\nFor Enter:")
for i, (feat, imp) in enumerate(feature_importance, 1):
    print(f"{i:2d}. {feat:40s} : {imp:.4f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"""
✅ Model Trained Successfully!

Performance:
- Enter Accuracy: {acc_enter:.2%}
- Exit Accuracy:  {acc_exit:.2%}

Data:
- Training samples: {len(X_train)}
- Test samples: {len(X_test)}
- Feature count: {X_train.shape[1]}

NOTE: 
- This demo uses SYNTHETIC features
- In production, video processing should be implemented
- Once video files are ready, run traffic_analysis_solution.py

Next Steps:
1. Place video files under videos/
2. python traffic_analysis_solution.py (full training)
3. python test_prediction.py (prediction)
4. python analyze_results.py (analysis)
""")

print("=" * 80)
