import pandas as pd
from typing import Tuple, Optional
from src.config import DATA_DIR_CANDIDATES

def locate_data_dir() -> str:
    for c in DATA_DIR_CANDIDATES:
        if c.exists() and (c / "train.csv").exists():
            return c
    raise FileNotFoundError("Could not locate Kaggle dataset 'playground-series-s6e9'")

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    data_dir = locate_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    
    orig_path = data_dir / "EV_Adoption_and_Range_Anxiety_Dataset.csv"
    orig = pd.read_csv(orig_path) if orig_path.exists() else None
    
    return train, test, orig

def get_targets(train_df: pd.DataFrame, target_col: str = "Will_Buy_EV") -> pd.Series:
    return train_df[target_col].map({"Yes": 1, "No": 0, "1": 1, "0": 0}).astype(int)
