"""
Development 9: Multi-Output Deep Neural Network
Advanced deep learning approach with shared layers and separate output heads
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Check if TensorFlow is available
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    from sklearn.preprocessing import StandardScaler
    TENSORFLOW_AVAILABLE = True
    print(f'[OK] TensorFlow {tf.__version__} available')
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print('[WARNING] TensorFlow not available, will install...')

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib


def install_tensorflow():
    """Install TensorFlow if not available"""
    if not TENSORFLOW_AVAILABLE:
        import subprocess
        print('\n[INSTALL] Installing TensorFlow...')
        subprocess.run(['pip', 'install', 'tensorflow'], check=True)
        print('[OK] TensorFlow installed!')
        return True
    return True


from sklearn.preprocessing import StandardScaler


def prepare_advanced_features():
    """Prepare enhanced features for neural network"""
    
    print('\n[DEV 9] GELISTIRME 9: Multi-Output Neural Network')
    print('='*80)
    
    # Load data
    train_df = pd.read_csv('Train.csv')
    train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
    
    # Extract time features
    train_df['hour'] = train_df['datetime'].dt.hour
    train_df['minute'] = train_df['datetime'].dt.minute
    train_df['day_of_week'] = train_df['datetime'].dt.dayofweek
    train_df['day_of_month'] = train_df['datetime'].dt.day
    train_df['month'] = train_df['datetime'].dt.month
    train_df['week_of_year'] = train_df['datetime'].dt.isocalendar().week
    
    # Boolean features
    train_df['is_rush_hour'] = train_df['hour'].apply(
        lambda h: 1 if h in [7, 8, 9, 16, 17, 18] else 0
    )
    train_df['is_weekend'] = train_df['day_of_week'].apply(lambda d: 1 if d >= 5 else 0)
    train_df['is_morning'] = train_df['hour'].apply(lambda h: 1 if h < 12 else 0)
    train_df['is_afternoon'] = train_df['hour'].apply(lambda h: 1 if 12 <= h < 17 else 0)
    train_df['is_evening'] = train_df['hour'].apply(lambda h: 1 if 17 <= h < 21 else 0)
    train_df['is_night'] = train_df['hour'].apply(lambda h: 1 if h >= 21 or h < 6 else 0)
    
    # Cyclical encoding
    train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
    train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)
    train_df['minute_sin'] = np.sin(2 * np.pi * train_df['minute'] / 60)
    train_df['minute_cos'] = np.cos(2 * np.pi * train_df['minute'] / 60)
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
    train_df['hour_x_location'] = train_df['hour'] * train_df['location_encoded']
    
    # Polynomial features (hour)
    train_df['hour_squared'] = train_df['hour'] ** 2
    train_df['hour_cubed'] = train_df['hour'] ** 3
    
    print(f'[OK] Enhanced feature engineering completed')
    print(f'[OK] Total features: {len([c for c in train_df.columns if c not in ["datetime", "view_label", "signaling"]])}')
    
    return train_df, location_encoder


def build_multi_output_network(input_dim, n_classes_enter, n_classes_exit):
    """
    Build multi-output neural network with shared layers
    
    Architecture:
    - Shared layers: Extract common temporal patterns
    - Enter head: Specialized for entry congestion
    - Exit head: Specialized for exit congestion
    """
    
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Shared layers - extract common patterns
    shared = layers.Dense(128, activation='relu', name='shared_1')(inputs)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    shared = layers.Dense(64, activation='relu', name='shared_2')(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.3)(shared)
    
    shared = layers.Dense(32, activation='relu', name='shared_3')(shared)
    shared = layers.BatchNormalization()(shared)
    shared = layers.Dropout(0.2)(shared)
    
    # Enter congestion head
    enter_branch = layers.Dense(32, activation='relu', name='enter_dense_1')(shared)
    enter_branch = layers.Dropout(0.2)(enter_branch)
    enter_branch = layers.Dense(16, activation='relu', name='enter_dense_2')(enter_branch)
    enter_output = layers.Dense(n_classes_enter, activation='softmax', name='enter_output')(enter_branch)
    
    # Exit congestion head
    exit_branch = layers.Dense(32, activation='relu', name='exit_dense_1')(shared)
    exit_branch = layers.Dropout(0.2)(exit_branch)
    exit_branch = layers.Dense(16, activation='relu', name='exit_dense_2')(exit_branch)
    exit_output = layers.Dense(n_classes_exit, activation='softmax', name='exit_output')(exit_branch)
    
    # Create model
    model = models.Model(
        inputs=inputs,
        outputs=[enter_output, exit_output],
        name='multi_output_traffic_network'
    )
    
    # Compile with separate losses and metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'enter_output': 'sparse_categorical_crossentropy',
            'exit_output': 'sparse_categorical_crossentropy'
        },
        loss_weights={
            'enter_output': 1.0,
            'exit_output': 1.0
        },
        metrics={
            'enter_output': ['accuracy'],
            'exit_output': ['accuracy']
        }
    )
    
    return model


def train_neural_network():
    """Train the multi-output neural network"""
    
    print('\n🧠 Training Multi-Output Neural Network...')
    print('-'*80)
    
    # Prepare features
    train_df, location_encoder = prepare_advanced_features()
    
    # Feature columns
    feature_cols = [
        'hour', 'minute', 'day_of_week', 'day_of_month', 'month', 'week_of_year',
        'is_rush_hour', 'is_weekend', 'is_morning', 'is_afternoon', 'is_evening', 'is_night',
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'day_sin', 'day_cos',
        'location_encoded', 'signal_encoded',
        'rush_x_location', 'hour_x_location',
        'hour_squared', 'hour_cubed'
    ]
    
    # Prepare X, y
    X = train_df[feature_cols].values
    
    # Encode labels
    enter_labels = train_df['congestion_enter_rating'].dropna()
    exit_labels = train_df['congestion_exit_rating'].dropna()
    
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
    
    print(f'\n📊 Dataset:')
    print(f'  Samples: {len(X):,}')
    print(f'  Features: {len(feature_cols)}')
    print(f'  Enter classes: {len(np.unique(y_enter))}')
    print(f'  Exit classes: {len(np.unique(y_exit))}')
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train/validation split
    X_train, X_val, y_enter_train, y_enter_val, y_exit_train, y_exit_val = train_test_split(
        X_scaled, y_enter, y_exit, test_size=0.2, random_state=42, stratify=y_enter
    )
    
    print(f'\n📊 Split:')
    print(f'  Training: {len(X_train):,}')
    print(f'  Validation: {len(X_val):,}')
    
    # Build model
    model = build_multi_output_network(
        input_dim=len(feature_cols),
        n_classes_enter=len(np.unique(y_enter)),
        n_classes_exit=len(np.unique(y_exit))
    )
    
    print(f'\n🏗️ Model Architecture:')
    model.summary()
    
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1
    )
    
    # Train
    print(f'\n🚀 Training...')
    history = model.fit(
        X_train,
        {'enter_output': y_enter_train, 'exit_output': y_exit_train},
        validation_data=(
            X_val,
            {'enter_output': y_enter_val, 'exit_output': y_exit_val}
        ),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate
    print(f'\n📈 Evaluation:')
    
    enter_pred = model.predict(X_val)[0].argmax(axis=1)
    exit_pred = model.predict(X_val)[1].argmax(axis=1)
    
    enter_acc = accuracy_score(y_enter_val, enter_pred)
    exit_acc = accuracy_score(y_exit_val, exit_pred)
    
    print(f'\n  Enter Accuracy: {enter_acc*100:.2f}%')
    print(f'  Exit Accuracy: {exit_acc*100:.2f}%')
    
    # Save model and artifacts
    model.save('neural_network_model.h5')
    joblib.dump(scaler, 'neural_network_scaler.pkl')
    joblib.dump(feature_cols, 'neural_network_features.pkl')
    joblib.dump(location_encoder, 'neural_network_location_encoder.pkl')
    joblib.dump(label_map, 'neural_network_label_map.pkl')
    
    print(f'\n✅ Model saved:')
    print(f'  • neural_network_model.h5')
    print(f'  • neural_network_scaler.pkl')
    print(f'  • neural_network_features.pkl')
    print(f'  • neural_network_location_encoder.pkl')
    print(f'  • neural_network_label_map.pkl')
    
    return model, scaler, feature_cols, location_encoder, label_map, (enter_acc, exit_acc)


def generate_nn_submission():
    """Generate submission using neural network predictions"""
    
    print('\n📝 Generating Neural Network Submission...')
    print('-'*80)
    
    # Load model and artifacts
    model = keras.models.load_model('neural_network_model.h5')
    scaler = joblib.load('neural_network_scaler.pkl')
    feature_cols = joblib.load('neural_network_features.pkl')
    location_encoder = joblib.load('neural_network_location_encoder.pkl')
    label_map = joblib.load('neural_network_label_map.pkl')
    
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
    
    # Interpolation function
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
    
    # Generate predictions
    print(f'\n🔮 Predicting with Neural Network...')
    
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
        
        # Get time info
        time_info = interpolate_segment_time(segment_id)
        hour = time_info['hour']
        minute = time_info['minute']
        day_of_week = time_info['day_of_week']
        
        # Create features
        features = {
            'hour': hour,
            'minute': minute,
            'day_of_week': day_of_week,
            'day_of_month': 1,
            'month': 1,
            'week_of_year': 1,
            'is_rush_hour': 1 if hour in [7, 8, 9, 16, 17, 18] else 0,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_morning': 1 if hour < 12 else 0,
            'is_afternoon': 1 if 12 <= hour < 17 else 0,
            'is_evening': 1 if 17 <= hour < 21 else 0,
            'is_night': 1 if hour >= 21 or hour < 6 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'minute_sin': np.sin(2 * np.pi * minute / 60),
            'minute_cos': np.cos(2 * np.pi * minute / 60),
            'day_sin': np.sin(2 * np.pi * day_of_week / 7),
            'day_cos': np.cos(2 * np.pi * day_of_week / 7),
            'location_encoded': location_encoder.get(location_name, 0),
            'signal_encoded': location_signals.get(location_name, 0),
            'rush_x_location': (1 if hour in [7, 8, 9, 16, 17, 18] else 0) * location_encoder.get(location_name, 0),
            'hour_x_location': hour * location_encoder.get(location_name, 0),
            'hour_squared': hour ** 2,
            'hour_cubed': hour ** 3
        }
        
        X = pd.DataFrame([features])[feature_cols].values
        X_scaled = scaler.transform(X)
        
        # Predict
        enter_pred, exit_pred = model.predict(X_scaled, verbose=0)
        
        if rating_type == 'enter':
            pred_class = enter_pred[0].argmax()
        else:
            pred_class = exit_pred[0].argmax()
        
        prediction = reverse_label_map[pred_class]
        
        submission_data.append({
            'ID': req_id,
            'Target': prediction,
            'Target_Accuracy': prediction
        })
    
    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv('submission_nn.csv', index=False)
    
    print(f'\n✅ Neural Network submission saved: submission_nn.csv')
    
    # Distribution
    print(f'\n📊 NN Prediction Distribution:')
    dist = submission_df['Target'].value_counts(normalize=True) * 100
    for label in ['free flowing', 'light delay', 'moderate delay', 'heavy delay']:
        pct = dist.get(label, 0)
        print(f'  {label}: {pct:.1f}%')
    
    return submission_df


if __name__ == '__main__':
    # Install TensorFlow if needed
    if install_tensorflow():
        # Re-import after installation
        if not TENSORFLOW_AVAILABLE:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers, models, callbacks
        
        # Train neural network
        model, scaler, features, loc_enc, label_map, accuracies = train_neural_network()
        
        print(f'\n📊 Final Accuracies:')
        print(f'  Enter: {accuracies[0]*100:.2f}%')
        print(f'  Exit: {accuracies[1]*100:.2f}%')
        
        # Generate submission
        submission = generate_nn_submission()
        
        print('\n' + '='*80)
        print('✅ GELİŞTİRME 9 TAMAMLANDI')
        print('   → Multi-output neural network trained')
        print('   → Shared layers + separate heads')
        print('   → submission_nn.csv generated')
        print('='*80)
