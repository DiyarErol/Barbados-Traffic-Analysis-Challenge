"""
Ensemble Model System for Traffic Congestion Prediction
Combines multiple models for improved accuracy and robustness
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import joblib
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class EnsembleTrafficPredictor:
    """
    Advanced ensemble model combining multiple classifiers
    for improved traffic congestion prediction
    """
    
    def __init__(self, ensemble_type: str = 'voting', n_jobs: int = -1):
        """
        Args:
            ensemble_type: 'voting' (hard/soft) or 'stacking'
            n_jobs: Number of parallel jobs
        """
        self.ensemble_type = ensemble_type
        self.n_jobs = n_jobs
        self.enter_model = None
        self.exit_model = None
        self.feature_names = None
        
    def _create_base_models(self) -> List[Tuple[str, object]]:
        """Create base models for ensemble"""
        
        # Gradient Boosting - Best for feature interactions
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            random_state=42,
            verbose=0
        )
        
        # Random Forest - Good for non-linear patterns
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        # Extra Trees - More randomization, less overfitting
        et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=self.n_jobs,
            verbose=0
        )
        
        return [
            ('gradient_boosting', gb),
            ('random_forest', rf),
            ('extra_trees', et)
        ]
    
    def _create_voting_ensemble(self, voting: str = 'soft') -> VotingClassifier:
        """
        Create voting ensemble
        
        Args:
            voting: 'hard' (majority vote) or 'soft' (average probabilities)
        """
        base_models = self._create_base_models()
        
        return VotingClassifier(
            estimators=base_models,
            voting=voting,
            n_jobs=self.n_jobs,
            verbose=False
        )
    
    def _create_stacking_ensemble(self) -> StackingClassifier:
        """Create stacking ensemble with meta-learner"""
        base_models = self._create_base_models()
        
        # Logistic Regression as meta-learner
        meta_learner = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=self.n_jobs
        )
        
        return StackingClassifier(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5,
            n_jobs=self.n_jobs,
            verbose=0
        )
    
    def train(self, X_train: pd.DataFrame, y_enter: pd.Series, y_exit: pd.Series):
        """
        Train ensemble models
        
        Args:
            X_train: Training features
            y_enter: Enter congestion labels
            y_exit: Exit congestion labels
        """
        print(f"🔧 Ensemble Tipi: {self.ensemble_type.upper()}")
        print(f"📊 Eğitim Örnekleri: {len(X_train)}")
        print(f"🎯 Özellik Sayısı: {X_train.shape[1]}")
        print()
        
        self.feature_names = X_train.columns.tolist()
        
        # Create ensemble models
        if self.ensemble_type == 'voting':
            print("🗳️  Voting Ensemble (Soft Voting) Oluşturuluyor...")
            self.enter_model = self._create_voting_ensemble(voting='soft')
            self.exit_model = self._create_voting_ensemble(voting='soft')
        elif self.ensemble_type == 'stacking':
            print("📚 Stacking Ensemble (Meta-Learner) Oluşturuluyor...")
            self.enter_model = self._create_stacking_ensemble()
            self.exit_model = self._create_stacking_ensemble()
        else:
            raise ValueError(f"Unknown ensemble_type: {self.ensemble_type}")
        
        # Train Enter model
        print("\n🚦 Enter Congestion Model Eğitiliyor...")
        self.enter_model.fit(X_train, y_enter)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.enter_model, X_train, y_enter,
            cv=5, scoring='accuracy', n_jobs=self.n_jobs
        )
        print(f"   ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Train Exit model
        print("\n🚦 Exit Congestion Model Eğitiliyor...")
        self.exit_model.fit(X_train, y_exit)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.exit_model, X_train, y_exit,
            cv=5, scoring='accuracy', n_jobs=self.n_jobs
        )
        print(f"   ✓ CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("\n✅ Ensemble Model Eğitimi Tamamlandı!")
    
    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict congestion levels
        
        Args:
            X: Features
            
        Returns:
            Tuple of (enter_predictions, exit_predictions)
        """
        if self.enter_model is None or self.exit_model is None:
            raise ValueError("Model henüz eğitilmedi! Önce train() çağırın.")
        
        enter_pred = self.enter_model.predict(X)
        exit_pred = self.exit_model.predict(X)
        
        return enter_pred, exit_pred
    
    def predict_proba(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict class probabilities
        
        Args:
            X: Features
            
        Returns:
            Tuple of (enter_probabilities, exit_probabilities)
        """
        if self.enter_model is None or self.exit_model is None:
            raise ValueError("Model henüz eğitilmedi! Önce train() çağırın.")
        
        enter_proba = self.enter_model.predict_proba(X)
        exit_proba = self.exit_model.predict_proba(X)
        
        return enter_proba, exit_proba
    
    def evaluate(self, X_test: pd.DataFrame, y_enter_test: pd.Series, 
                 y_exit_test: pd.Series) -> Dict:
        """
        Evaluate ensemble model performance
        
        Args:
            X_test: Test features
            y_enter_test: True enter labels
            y_exit_test: True exit labels
            
        Returns:
            Dictionary with metrics
        """
        enter_pred, exit_pred = self.predict(X_test)
        
        # Calculate accuracies
        enter_acc = accuracy_score(y_enter_test, enter_pred)
        exit_acc = accuracy_score(y_exit_test, exit_pred)
        
        print("\n" + "="*60)
        print("📊 ENSEMBLE MODEL DEĞERLENDİRME")
        print("="*60)
        
        print(f"\n🎯 Enter Congestion Accuracy: {enter_acc:.4f} ({enter_acc*100:.2f}%)")
        print("\nDetaylı Rapor (Enter):")
        print(classification_report(y_enter_test, enter_pred))
        
        print(f"\n🎯 Exit Congestion Accuracy: {exit_acc:.4f} ({exit_acc*100:.2f}%)")
        print("\nDetaylı Rapor (Exit):")
        print(classification_report(y_exit_test, exit_pred))
        
        return {
            'enter_accuracy': enter_acc,
            'exit_accuracy': exit_acc,
            'enter_predictions': enter_pred,
            'exit_predictions': exit_pred
        }
    
    def get_model_weights(self) -> Dict:
        """Get individual model accuracies (for voting ensemble)"""
        if self.ensemble_type != 'voting':
            return {}
        
        weights = {}
        for name, model in self.enter_model.named_estimators_.items():
            weights[name] = {
                'enter_model': model,
                'exit_model': self.exit_model.named_estimators_[name]
            }
        
        return weights
    
    def save_models(self, enter_path: str = 'ensemble_enter_model.pkl',
                    exit_path: str = 'ensemble_exit_model.pkl'):
        """Save trained models"""
        if self.enter_model is None or self.exit_model is None:
            raise ValueError("Model henüz eğitilmedi!")
        
        joblib.dump(self.enter_model, enter_path)
        joblib.dump(self.exit_model, exit_path)
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'ensemble_type': self.ensemble_type
        }
        joblib.dump(metadata, 'ensemble_metadata.pkl')
        
        print(f"\n💾 Modeller Kaydedildi:")
        print(f"   - {enter_path}")
        print(f"   - {exit_path}")
        print(f"   - ensemble_metadata.pkl")
    
    def load_models(self, enter_path: str = 'ensemble_enter_model.pkl',
                    exit_path: str = 'ensemble_exit_model.pkl'):
        """Load trained models"""
        self.enter_model = joblib.load(enter_path)
        self.exit_model = joblib.load(exit_path)
        
        # Load metadata
        metadata = joblib.load('ensemble_metadata.pkl')
        self.feature_names = metadata['feature_names']
        self.ensemble_type = metadata['ensemble_type']
        
        print(f"\n📂 Modeller Yüklendi:")
        print(f"   - {enter_path}")
        print(f"   - {exit_path}")
        print(f"   - Ensemble Type: {self.ensemble_type}")


def demo_ensemble():
    """Demo function to test ensemble model"""
    print("\n" + "="*60)
    print("🎯 ENSEMBLE MODEL DEMO")
    print("="*60)
    
    # Load training data
    print("\n📁 Veri Yükleniyor...")
    train_df = pd.read_csv('Train.csv')
    
    print(f"✓ Toplam Örnek: {len(train_df)}")
    print(f"✓ Sınıf Dağılımı (Enter):")
    print(train_df['congestion_enter_rating'].value_counts().sort_index())
    
    # Generate synthetic features for demo
    print("\n🔧 Sentetik Özellikler Oluşturuluyor...")
    np.random.seed(42)
    n_samples = len(train_df)
    
    # Create features based on congestion levels
    features = pd.DataFrame()
    for idx, row in train_df.iterrows():
        congestion_level = row['congestion_enter_rating']
        
        if congestion_level == 0:  # Free flowing
            vehicle_count = np.random.randint(5, 15)
            speed = np.random.uniform(45, 60)
            density = np.random.uniform(0.1, 0.3)
        elif congestion_level == 1:  # Light delay
            vehicle_count = np.random.randint(12, 25)
            speed = np.random.uniform(30, 45)
            density = np.random.uniform(0.25, 0.5)
        elif congestion_level == 2:  # Moderate delay
            vehicle_count = np.random.randint(22, 38)
            speed = np.random.uniform(15, 30)
            density = np.random.uniform(0.45, 0.7)
        else:  # Heavy delay
            vehicle_count = np.random.randint(35, 55)
            speed = np.random.uniform(5, 18)
            density = np.random.uniform(0.65, 0.95)
        
        features.loc[idx, 'vehicle_count'] = vehicle_count
        features.loc[idx, 'avg_speed'] = speed
        features.loc[idx, 'traffic_density'] = density
        features.loc[idx, 'vehicle_variance'] = np.random.uniform(0, vehicle_count * 0.2)
        features.loc[idx, 'speed_variance'] = np.random.uniform(0, 10)
    
    # Add temporal features
    train_df['datetime'] = pd.to_datetime(train_df['datetimestamp_start'])
    features['hour'] = train_df['datetime'].dt.hour
    features['is_rush_hour'] = features['hour'].isin([7, 8, 9, 16, 17, 18]).astype(int)
    features['day_of_week'] = train_df['datetime'].dt.dayofweek
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_enter_train, y_enter_test, y_exit_train, y_exit_test = train_test_split(
        features,
        train_df['congestion_enter_rating'],
        train_df['congestion_exit_rating'],
        test_size=0.2,
        random_state=42,
        stratify=train_df['congestion_enter_rating']
    )
    
    print(f"✓ Eğitim Seti: {len(X_train)} örnek")
    print(f"✓ Test Seti: {len(X_test)} örnek")
    
    # Test both ensemble types
    for ensemble_type in ['voting', 'stacking']:
        print("\n" + "="*60)
        print(f"🎯 {ensemble_type.upper()} ENSEMBLE TEST")
        print("="*60)
        
        # Create and train ensemble
        ensemble = EnsembleTrafficPredictor(ensemble_type=ensemble_type)
        ensemble.train(X_train, y_enter_train, y_exit_train)
        
        # Evaluate
        results = ensemble.evaluate(X_test, y_enter_test, y_exit_test)
        
        # Save models
        ensemble.save_models(
            enter_path=f'{ensemble_type}_ensemble_enter_model.pkl',
            exit_path=f'{ensemble_type}_ensemble_exit_model.pkl'
        )
    
    print("\n" + "="*60)
    print("✅ ENSEMBLE DEMO TAMAMLANDI!")
    print("="*60)
    print("\nKullanım:")
    print("  from traffic_ensemble import EnsembleTrafficPredictor")
    print("  ensemble = EnsembleTrafficPredictor(ensemble_type='voting')")
    print("  ensemble.load_models('voting_ensemble_enter_model.pkl', ...)")
    print("  predictions = ensemble.predict(new_features)")


if __name__ == "__main__":
    demo_ensemble()
