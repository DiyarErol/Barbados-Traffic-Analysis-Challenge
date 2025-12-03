from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class ModelTraining:
    def train_simple(self, df: pd.DataFrame, feature_cols: list, target_col: str,
                     params: Dict[str, Any]) -> Tuple[GradientBoostingClassifier, float]:
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = f1_score(y_val, pred, average='macro')
        return model, score
