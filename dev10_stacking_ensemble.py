"""
Development 10: XGBoost + LightGBM Stacked Ensemble
Advanced gradient boosting with meta-learner stacking
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    print(f'[OK] XGBoost {xgb.__version__} available')
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print('[WARNING] XGBoost not available, will install...')

try:
    import lightgbm as lgb
    print(f'[OK] LightGBM {lgb.__version__} available')
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print('[WARNING] LightGBM not available, will install...')

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def install_boosting_libraries():
    """Install XGBoost and LightGBM if not available"""
    import subprocess
    
    if not XGBOOST_AVAILABLE:
        print('\\n[INSTALL] Installing XGBoost...')
        subprocess.run(['pip', 'install', 'xgboost'], check=True)
        print('[OK] XGBoost installed!')
    
    if not LIGHTGBM_AVAILABLE:
        print('\\n[INSTALL] Installing LightGBM...')
        subprocess.run(['pip', 'install', 'lightgbm'], check=True)
        print('[OK] LightGBM installed!')


def prepare_training_data():
    """Prepare training data with enhanced features"""
    
    print('\\n[DEV 10] GELISTIRME 10: XGBoost + LightGBM Stacking')
    print('='*80)
    
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
    
    print(f'[OK] Features prepared: {train_df.shape[1]} columns')
    
    return train_df, location_encoder


def train_xgboost_lightgbm_ensemble():
    """Train XGBoost + LightGBM stacked ensemble"""
    
    print('\\n[TRAIN] Training XGBoost + LightGBM Ensemble...')
    print('-'*80)
    
    # Prepare data
    train_df, location_encoder = prepare_training_data()
    
    feature_cols = [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month',
        'is_rush_hour', 'is_weekend', 'is_morning', 'is_evening',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'location_encoded', 'signal_encoded', 'rush_x_location'
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
    
    print(f'\\n[STATS] Dataset:')
    print(f'  Samples: {len(X):,}')
    print(f'  Features: {len(feature_cols)}')
    
    # Train/test split
    X_train, X_test, y_enter_train, y_enter_test, y_exit_train, y_exit_test = train_test_split(
        X, y_enter, y_exit, test_size=0.2, random_state=42, stratify=y_enter
    )
    
    print(f'\\n[STATS] Split:')
    print(f'  Training: {len(X_train):,}')
    print(f'  Testing: {len(X_test):,}')
    
    # ===== ENTER MODEL =====
    print(f'\\n[TRAIN] Training ENTER model...')
    
    # XGBoost for enter
    xgb_enter = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    
    # LightGBM for enter
    lgb_enter = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=4,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # RandomForest for diversity
    rf_enter = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Stacking ensemble
    estimators_enter = [
        ('xgb', xgb_enter),
        ('lgb', lgb_enter),
        ('rf', rf_enter)
    ]
    
    stacking_enter = StackingClassifier(
        estimators=estimators_enter,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    stacking_enter.fit(X_train, y_enter_train)
    
    # Evaluate enter
    y_enter_pred = stacking_enter.predict(X_test)
    enter_acc = accuracy_score(y_enter_test, y_enter_pred)
    print(f'\\n[EVAL] Enter Accuracy: {enter_acc*100:.2f}%')
    
    # ===== EXIT MODEL =====
    print(f'\\n[TRAIN] Training EXIT model...')
    
    # XGBoost for exit
    xgb_exit = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softmax',
        num_class=4,
        random_state=42,
        n_jobs=-1
    )
    
    # LightGBM for exit
    lgb_exit = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multiclass',
        num_class=4,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    # RandomForest for diversity
    rf_exit = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Stacking ensemble
    estimators_exit = [
        ('xgb', xgb_exit),
        ('lgb', lgb_exit),
        ('rf', rf_exit)
    ]
    
    stacking_exit = StackingClassifier(
        estimators=estimators_exit,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5,
        n_jobs=-1,
        verbose=1
    )
    
    stacking_exit.fit(X_train, y_exit_train)
    
    # Evaluate exit
    y_exit_pred = stacking_exit.predict(X_test)
    exit_acc = accuracy_score(y_exit_test, y_exit_pred)
    print(f'\\n[EVAL] Exit Accuracy: {exit_acc*100:.2f}%')
    
    # Save models
    joblib.dump(stacking_enter, 'stacking_enter_model.pkl')
    joblib.dump(stacking_exit, 'stacking_exit_model.pkl')
    joblib.dump(feature_cols, 'stacking_features.pkl')
    joblib.dump(location_encoder, 'stacking_location_encoder.pkl')
    joblib.dump(label_map, 'stacking_label_map.pkl')
    
    print(f'\\n[OK] Models saved:')
    print(f'  * stacking_enter_model.pkl')
    print(f'  * stacking_exit_model.pkl')
    print(f'  * stacking_features.pkl')
    print(f'  * stacking_location_encoder.pkl')
    print(f'  * stacking_label_map.pkl')
    
    return stacking_enter, stacking_exit, feature_cols, location_encoder, label_map, (enter_acc, exit_acc)


def generate_stacking_submission():
    """Generate submission using stacking ensemble"""
    
    print('\\n[SUBMIT] Generating Stacking Ensemble Submission...')
    print('-'*80)
    
    # Load models
    stacking_enter = joblib.load('stacking_enter_model.pkl')
    stacking_exit = joblib.load('stacking_exit_model.pkl')
    feature_cols = joblib.load('stacking_features.pkl')
    location_encoder = joblib.load('stacking_location_encoder.pkl')
    label_map = joblib.load('stacking_label_map.pkl')
    
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
    
    print(f'\\n[PREDICT] Predicting with stacking ensemble...')
    
    submission_data = []
    
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
        features = {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'day_of_month': 1,
            'month': 1,
            'is_rush_hour': 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_morning': 1 if hour < 12 else 0,
            'is_evening': 1 if 17 <= hour < 21 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'location_encoded': location_encoder.get(location_name, 0),
            'signal_encoded': location_signals.get(location_name, 0),
            'rush_x_location': (1 if hour in [7, 8, 9, 16, 17, 18] else 0) * location_encoder.get(location_name, 0)
        }
        
        X = pd.DataFrame([features])[feature_cols]
        
        # Predict
        if rating_type == 'enter':
            pred_class = stacking_enter.predict(X)[0]
        else:
            pred_class = stacking_exit.predict(X)[0]
        
        prediction = reverse_label_map[pred_class]
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission_stacking.csv', index=False)
    
    print(f'\\n[OK] Stacking submission saved: submission_stacking.csv')
    
    # Distribution
    print(f'\\n[STATS] Stacking Prediction Distribution:')
    dist = submission_df['Target'].value_counts(normalize=True) * 100
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        pct = dist.get(label, 0)
        print(f'  {label}: {pct:.1f}%')
    
    return submission_df


if __name__ == '__main__':
    # Install libraries if needed
    install_boosting_libraries()
    
    # Re-import after installation
    if not XGBOOST_AVAILABLE or not LIGHTGBM_AVAILABLE:
        import xgboost as xgb
        import lightgbm as lgb
    
    # Train ensemble
    enter_model, exit_model, features, loc_enc, label_map, accuracies = train_xgboost_lightgbm_ensemble()
    
    print(f'\\n[FINAL] Final Accuracies:')
    print(f'  Enter: {accuracies[0]*100:.2f}%')
    print(f'  Exit: {accuracies[1]*100:.2f}%')
    
    # Generate submission
    submission = generate_stacking_submission()
    
    print('\\n' + '='*80)
    print('[OK] GELISTIRME 10 TAMAMLANDI')
    print('   -> XGBoost + LightGBM + RandomForest stacking')
    print('   -> Logistic regression meta-learner')
    print('   -> submission_stacking.csv generated')
    print('='*80)
