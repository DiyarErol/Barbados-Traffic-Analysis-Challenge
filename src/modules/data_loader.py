from pathlib import Path
from typing import Optional
import pandas as pd

class DataLoader:
    """Loads training, test and auxiliary data.
    Paths are defined in `config/config.yaml`.
    """
    def __init__(self, train_csv: Path, test_csv: Optional[Path] = None):
        self.train_csv = Path(train_csv)
        self.test_csv = Path(test_csv) if test_csv else None

    def load_train(self) -> pd.DataFrame:
        return pd.read_csv(self.train_csv)

    def load_test(self) -> Optional[pd.DataFrame]:
        if self.test_csv and self.test_csv.exists():
            return pd.read_csv(self.test_csv)
        return None
