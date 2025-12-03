import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from typing import Dict, Any

class Evaluation:
    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
        return {
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
