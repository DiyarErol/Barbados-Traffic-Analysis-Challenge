"""
Train Better Model with Available Features
Uses only time-based features that we can extract from segment IDs
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

def train_time_based_model():
    """Train model using only time-based features"""
    
    print('='*60)
    print('🎓 TIME-BASED MODEL TRAINING')
    print('='*60)
    
    # Load data
    print('\n📁 Loading Data...')
    df = pd.read_csv('Train.csv')
    print(f'✓ Total records: {len(df):,}')
    
    # Time features
    df['datetime'] = pd.to_datetime(df['datetimestamp_start'])
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_morning'] = (df['hour'] < 12).astype(int)
    df['is_evening'] = ((df['hour'] >= 16) & (df['hour'] < 20)).astype(int)
    
    # Cyclical encoding for hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Signal encoding
    signal_map = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}
    df['signal_encoded'] = df['signaling'].map(signal_map).fillna(0)
    
    # Features
    feature_cols = [
        'hour', 'minute', 'day_of_week',
        'is_rush_hour', 'is_weekend', 
        'is_morning', 'is_evening',
        'hour_sin', 'hour_cos',
        'signal_encoded'
    ]
    
    X = df[feature_cols]
    
    # Encode targets
    le_enter = LabelEncoder()
    le_exit = LabelEncoder()
    
    y_enter = le_enter.fit_transform(df['congestion_enter_rating'])
    y_exit = le_exit.fit_transform(df['congestion_exit_rating'])
    
    print(f'\n📊 Class Distribution:')
    print(f'\nEnter:')
    for label, count in zip(*np.unique(y_enter, return_counts=True)):
        print(f'  {le_enter.inverse_transform([label])[0]}: {count:,}')
    
    print(f'\nExit:')
    for label, count in zip(*np.unique(y_exit, return_counts=True)):
        print(f'  {le_exit.inverse_transform([label])[0]}: {count:,}')
    
    # Split data
    X_train, X_test, y_enter_train, y_enter_test, y_exit_train, y_exit_test = train_test_split(
        X, y_enter, y_exit, test_size=0.2, random_state=42, stratify=y_enter
    )
    
    print(f'\n🔧 Training set: {len(X_train):,}')
    print(f'🧪 Test set: {len(X_test):,}')
    
    # Train models
    print(f'\n🎯 Training model...')
    
    # Enter model - RandomForest with balanced class weights
    print('  Enter model (RandomForest)...')
    enter_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    enter_model.fit(X_train, y_enter_train)
    
    # Exit model
    print('  Exit model (GradientBoosting)...')
    exit_model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=10,
        learning_rate=0.1,
        random_state=42
    )
    exit_model.fit(X_train, y_exit_train)
    
    # Evaluate
    print(f'\n📊 MODEL PERFORMANCE:')
    
    # Enter
    enter_pred = enter_model.predict(X_test)
    enter_acc = accuracy_score(y_enter_test, enter_pred)
    print(f'\n🚦 Enter Congestion:')
    print(f'  Accuracy: {enter_acc:.4f} ({enter_acc*100:.2f}%)')
    print('\nDetailed Report:')
    print(classification_report(y_enter_test, enter_pred, 
                                target_names=le_enter.classes_, 
                                zero_division=0))
    
    # Exit
    exit_pred = exit_model.predict(X_test)
    exit_acc = accuracy_score(y_exit_test, exit_pred)
    print(f'\n🚦 Exit Congestion:')
    print(f'  Accuracy: {exit_acc:.4f} ({exit_acc*100:.2f}%)')
    print('\nDetailed Report:')
    print(classification_report(y_exit_test, exit_pred, 
                                target_names=le_exit.classes_,
                                zero_division=0))
    
    # Feature importance
    print(f'\n📈 Top Features (Enter):')
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': enter_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(5).iterrows():
        print(f'  {row["feature"]}: {row["importance"]:.4f}')
    
    # Save models
    print(f'\n💾 Saving models...')
    joblib.dump(enter_model, 'time_based_enter_model.pkl')
    joblib.dump(exit_model, 'time_based_exit_model.pkl')
    joblib.dump({'enter': le_enter, 'exit': le_exit}, 'time_based_label_encoders.pkl')
    joblib.dump(feature_cols, 'time_based_features.pkl')
    
    print('  ✓ time_based_enter_model.pkl')
    print('  ✓ time_based_exit_model.pkl')
    print('  ✓ time_based_label_encoders.pkl')
    print('  ✓ time_based_features.pkl')
    
    return enter_model, exit_model, le_enter, le_exit, feature_cols


if __name__ == '__main__':
    models = train_time_based_model()
    
    print('\n' + '='*60)
    print('✅ MODEL TRAINING COMPLETED!')
    print('='*60)
    print('\n💡 Now create a submission with generate_final_submission.py')
