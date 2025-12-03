"""
Model Configuration
==================
Central configuration for all model parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """Configuration for model training and inference."""
    
    # Model type
    model_type: str = "gradient_boosting"  # Options: gradient_boosting, random_forest, xgboost, lightgbm
    
    # Gradient Boosting parameters
    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 7
    min_samples_split: int = 50
    min_samples_leaf: int = 20
    subsample: float = 0.8
    max_features: str = "sqrt"
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    stratify: bool = True
    
    # Class weights
    use_class_weights: bool = True
    class_weights: Optional[Dict[str, float]] = None
    
    # Ensemble settings
    use_ensemble: bool = False
    ensemble_models: List[str] = field(default_factory=lambda: ["gradient_boosting", "random_forest"])
    ensemble_weights: Optional[List[float]] = None
    
    # Training settings
    validation_split: float = 0.2
    early_stopping: bool = False
    early_stopping_rounds: int = 10
    
    # Output
    save_model: bool = True
    model_dir: str = "models"
    
    def to_sklearn_params(self) -> Dict:
        """Convert to sklearn-compatible parameters."""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'max_features': self.max_features,
            'random_state': self.random_state
        }
    
    def get_class_weights(self) -> Optional[str]:
        """Get class weight setting for sklearn."""
        if self.use_class_weights and self.class_weights is None:
            return 'balanced'
        return None


@dataclass
class HybridModelConfig:
    """Configuration for hybrid tree + deep learning models."""
    
    # Tree model config
    tree_model_config: ModelConfig = field(default_factory=ModelConfig)
    
    # Deep learning config
    use_deep_features: bool = False
    deep_model_type: str = "resnet"  # Options: resnet, efficientnet, custom_cnn
    pretrained: bool = True
    freeze_backbone: bool = True
    
    # Hybrid fusion
    fusion_method: str = "late"  # Options: early, late, stacking
    tree_weight: float = 0.7
    deep_weight: float = 0.3
    
    # Deep learning training
    deep_epochs: int = 50
    deep_batch_size: int = 32
    deep_learning_rate: float = 0.001
    deep_optimizer: str = "adam"
    
    # GPU settings
    use_gpu: bool = True
    gpu_id: int = 0
    mixed_precision: bool = False
