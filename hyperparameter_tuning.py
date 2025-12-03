"""
Hyperparameter Tuning System for Traffic Congestion Models
Automates parameter optimization using GridSearch and RandomizedSearch
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score
import joblib
from typing import Dict, Any, Tuple
import time
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Automated hyperparameter optimization for traffic prediction models
    """
    
    def __init__(self, model_type: str = 'gradient_boosting', 
                 search_type: str = 'grid', n_jobs: int = -1,
                 cv_folds: int = 5, random_state: int = 42):
        """
        Args:
            model_type: 'gradient_boosting' or 'random_forest'
            search_type: 'grid' (exhaustive) or 'random' (faster)
            n_jobs: Number of parallel jobs
            cv_folds: Cross-validation folds
            random_state: Random seed
        """
        self.model_type = model_type
        self.search_type = search_type
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.best_model_enter = None
        self.best_model_exit = None
        self.best_params_enter = None
        self.best_params_exit = None
        self.search_results = {}
    
    def _get_parameter_grid(self) -> Dict[str, list]:
        """Get parameter grid for the specified model type"""
        
        if self.model_type == 'gradient_boosting':
            if self.search_type == 'grid':
                # Smaller grid for exhaustive search
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.7, 0.8, 0.9],
                    'min_samples_split': [10, 20],
                    'min_samples_leaf': [4, 8]
                }
            else:  # random search
                # Wider range for random search
                return {
                    'n_estimators': [50, 100, 150, 200, 250, 300],
                    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'max_depth': [3, 5, 7, 9, 11],
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                    'min_samples_split': [5, 10, 15, 20, 30],
                    'min_samples_leaf': [2, 4, 6, 8, 10]
                }
        
        elif self.model_type == 'random_forest':
            if self.search_type == 'grid':
                return {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20],
                    'min_samples_split': [5, 10, 15],
                    'min_samples_leaf': [2, 4, 6],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                }
            else:  # random search
                return {
                    'n_estimators': [50, 100, 150, 200, 250, 300, 400],
                    'max_depth': [5, 10, 15, 20, 25, 30],
                    'min_samples_split': [2, 5, 10, 15, 20],
                    'min_samples_leaf': [1, 2, 4, 6, 8],
                    'max_features': ['sqrt', 'log2', 0.5, 0.7],
                    'bootstrap': [True, False]
                }
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def _get_base_model(self):
        """Get base model for tuning"""
        if self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(random_state=self.random_state, verbose=0)
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state, 
                                         n_jobs=self.n_jobs, verbose=0)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def tune_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   target_name: str = 'enter', n_iter: int = 50) -> Dict:
        """
        Tune hyperparameters for a single target
        
        Args:
            X_train: Training features
            y_train: Training labels
            target_name: 'enter' or 'exit' for logging
            n_iter: Number of iterations for RandomizedSearch
            
        Returns:
            Dictionary with best parameters and CV score
        """
        print(f"\n[START] {target_name.upper()} model tuning")
        print(f"   Model type: {self.model_type}")
        print(f"   Search method: {self.search_type}")
        print(f"   CV folds: {self.cv_folds}")
        
        # Get base model and parameter grid
        base_model = self._get_base_model()
        param_grid = self._get_parameter_grid()
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, 
                            random_state=self.random_state)
        
        # Setup scorer (weighted F1 for imbalanced classes)
        scorer = make_scorer(f1_score, average='weighted')
        
        # Perform search
        start_time = time.time()
        
        if self.search_type == 'grid':
            print(f"   Grid size: {np.prod([len(v) for v in param_grid.values()]):,} combinations")
            search = GridSearchCV(
                base_model, param_grid, cv=cv,
                scoring=scorer, n_jobs=self.n_jobs,
                verbose=1, refit=True
            )
        else:  # random search
            print(f"   Random search iterations: {n_iter}")
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, cv=cv,
                scoring=scorer, n_jobs=self.n_jobs,
                verbose=1, refit=True, random_state=self.random_state
            )
        
        # Fit
        search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        # Store results
        results = {
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'best_model': search.best_estimator_,
            'cv_results': search.cv_results_,
            'elapsed_time': elapsed_time
        }
        
        print(f"\n[DONE] Tuning completed in {elapsed_time:.1f} seconds")
        print(f"   Best CV score: {search.best_score_:.4f}")
        print(f"   Best parameters:")
        for param, value in search.best_params_.items():
            print(f"      {param}: {value}")
        
        return results
    
    def tune_both_targets(self, X_train: pd.DataFrame,
                         y_enter: pd.Series, y_exit: pd.Series,
                         n_iter: int = 50):
        """
        Tune hyperparameters for both enter and exit models
        
        Args:
            X_train: Training features
            y_enter: Enter congestion labels
            y_exit: Exit congestion labels
            n_iter: Number of iterations for RandomizedSearch
        """
        print("\n" + "="*60)
        print(f"HYPERPARAMETER TUNING - {self.model_type.upper()}")
        print("="*60)
        print(f"Training samples: {len(X_train)}")
        print(f"Feature count: {X_train.shape[1]}")
        
        # Tune Enter model
        enter_results = self.tune_model(X_train, y_enter, 'enter', n_iter)
        self.best_model_enter = enter_results['best_model']
        self.best_params_enter = enter_results['best_params']
        self.search_results['enter'] = enter_results
        
        # Tune Exit model
        exit_results = self.tune_model(X_train, y_exit, 'exit', n_iter)
        self.best_model_exit = exit_results['best_model']
        self.best_params_exit = exit_results['best_params']
        self.search_results['exit'] = exit_results
        
        print("\n" + "="*60)
        print("TUNING COMPLETED")
        print("="*60)
        
        # Summary
        print("\nResults:")
        print(f"   Enter CV score: {enter_results['best_cv_score']:.4f} ({enter_results['elapsed_time']:.1f}s)")
        print(f"   Exit CV score: {exit_results['best_cv_score']:.4f} ({exit_results['elapsed_time']:.1f}s)")
    
    def evaluate_on_test(self, X_test: pd.DataFrame,
                        y_enter_test: pd.Series, y_exit_test: pd.Series) -> Dict:
        """
        Evaluate tuned models on test set
        
        Args:
            X_test: Test features
            y_enter_test: True enter labels
            y_exit_test: True exit labels
            
        Returns:
            Dictionary with test metrics
        """
        if self.best_model_enter is None or self.best_model_exit is None:
            raise ValueError("Model has not been tuned yet!")
        
        # Predict
        enter_pred = self.best_model_enter.predict(X_test)
        exit_pred = self.best_model_exit.predict(X_test)
        
        # Calculate metrics
        enter_acc = accuracy_score(y_enter_test, enter_pred)
        exit_acc = accuracy_score(y_exit_test, exit_pred)
        
        enter_f1 = f1_score(y_enter_test, enter_pred, average='weighted')
        exit_f1 = f1_score(y_exit_test, exit_pred, average='weighted')
        
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(f"\nEnter congestion:")
        print(f"   Accuracy: {enter_acc:.4f} ({enter_acc*100:.2f}%)")
        print(f"   F1 score: {enter_f1:.4f}")

        print(f"\nExit congestion:")
        print(f"   Accuracy: {exit_acc:.4f} ({exit_acc*100:.2f}%)")
        print(f"   F1 score: {exit_f1:.4f}")
        
        return {
            'enter_accuracy': enter_acc,
            'exit_accuracy': exit_acc,
            'enter_f1': enter_f1,
            'exit_f1': exit_f1
        }
    
    def save_tuned_models(self, enter_path: str = 'tuned_enter_model.pkl',
                         exit_path: str = 'tuned_exit_model.pkl',
                         params_path: str = 'tuned_params.pkl'):
        """Save tuned models and parameters"""
        if self.best_model_enter is None or self.best_model_exit is None:
            raise ValueError("Model has not been tuned yet!")
        
        joblib.dump(self.best_model_enter, enter_path)
        joblib.dump(self.best_model_exit, exit_path)
        
        params = {
            'enter': self.best_params_enter,
            'exit': self.best_params_exit,
            'model_type': self.model_type,
            'search_type': self.search_type
        }
        joblib.dump(params, params_path)
        
        print(f"\nModels saved:")
        print(f"   - {enter_path}")
        print(f"   - {exit_path}")
        print(f"   - {params_path}")
    
    def get_parameter_importance(self) -> pd.DataFrame:
        """Analyze parameter importance from CV results"""
        if not self.search_results:
            raise ValueError("Tuning has not been performed yet!")
        
        results_list = []
        
        for target in ['enter', 'exit']:
            cv_results = self.search_results[target]['cv_results']
            
            # Convert to DataFrame
            df = pd.DataFrame(cv_results)
            
            # Extract mean test scores and parameters
            score_col = 'mean_test_score'
            param_cols = [col for col in df.columns if col.startswith('param_')]
            
            for param_col in param_cols:
                param_name = param_col.replace('param_', '')
                param_scores = df.groupby(param_col)[score_col].mean().sort_values(ascending=False)
                
                results_list.append({
                    'target': target,
                    'parameter': param_name,
                    'best_value': param_scores.index[0],
                    'best_score': param_scores.values[0],
                    'score_range': param_scores.max() - param_scores.min()
                })
        
        return pd.DataFrame(results_list)


def demo_tuning():
    """Demo function for hyperparameter tuning"""
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING DEMO")
    print("="*60)

    # Load training data
    print("\nLoading data...")
    # Prefer sample_data for quick demo runs
    csv_path = Path('sample_data/Train.csv') if Path('sample_data/Train.csv').exists() else Path('Train.csv')
    train_df = pd.read_csv(csv_path)

    print(f"Total samples: {len(train_df)}")

    # Generate synthetic features (same as ensemble demo)
    print("\nCreating synthetic features...")
    np.random.seed(42)
    
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
    
    # Use subset for faster demo (5000 samples)
    from sklearn.model_selection import train_test_split
    
    X_subset, _, y_enter_subset, _, y_exit_subset, _ = train_test_split(
        features, train_df['congestion_enter_rating'],
        train_df['congestion_exit_rating'],
        train_size=5000, random_state=42,
        stratify=train_df['congestion_enter_rating']
    )
    
    # Further split for train/test
    X_train, X_test, y_enter_train, y_enter_test, y_exit_train, y_exit_test = train_test_split(
        X_subset, y_enter_subset, y_exit_subset,
        test_size=0.2, random_state=42,
        stratify=y_enter_subset
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    
    # Test RandomizedSearch (faster)
    print("\n" + "="*60)
    print("Testing RandomizedSearch (20 iterations)...")
    print("="*60)
    
    tuner = HyperparameterTuner(
        model_type='gradient_boosting',
        search_type='random',
        cv_folds=3,  # Faster for demo
        n_jobs=-1
    )
    
    tuner.tune_both_targets(X_train, y_enter_train, y_exit_train, n_iter=20)
    
    # Evaluate on test set
    test_results = tuner.evaluate_on_test(X_test, y_enter_test, y_exit_test)
    
    # Save tuned models
    tuner.save_tuned_models()
    
    # Analyze parameter importance
    print("\n📊 Parameter Importance Analysis:")
    param_importance = tuner.get_parameter_importance()
    print(param_importance.to_string(index=False))
    
    print("\n" + "="*60)
    print("✅ TUNING DEMO COMPLETED!")
    print("="*60)
    print("\nUsage:")
    print("  from hyperparameter_tuning import HyperparameterTuner")
    print("  tuner = HyperparameterTuner(model_type='gradient_boosting', search_type='random')")
    print("  tuner.tune_both_targets(X_train, y_enter, y_exit, n_iter=50)")
    print("  tuner.save_tuned_models()")


if __name__ == "__main__":
    demo_tuning()
