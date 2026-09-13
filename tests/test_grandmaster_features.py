"""
Unit tests for Grandmaster Feature Engineering module.
"""

import pandas as pd
import numpy as np
from features.grandmaster_features import build_grandmaster_features, TARGET


def test_build_grandmaster_features_smoke():
    # Synthetic small dataset matching competition schema
    np.random.seed(42)
    n = 200
    data = {
        "id": range(n),
        "Age": np.random.randint(20, 70, size=n),
        "Annual_Income_USD": np.random.uniform(30000, 180000, size=n),
        "Daily_Commute_km": np.random.uniform(5, 100, size=n),
        "Number_of_Cars_Owned": np.random.randint(1, 5, size=n),
        "Charging_Stations_Near_Home": np.random.randint(0, 15, size=n),
        "Charging_Stations_Near_Work": np.random.randint(0, 20, size=n),
        "Environmental_Concern_Level": np.random.randint(1, 6, size=n),
        "Gender": np.random.choice(["Male", "Female", "Other"], size=n),
        "City_Type": np.random.choice(["Urban", "Suburban", "Rural"], size=n),
        "Current_Car_Type": np.random.choice(["Sedan", "SUV", "Hatchback", "Truck"], size=n),
        "Home_Charging_Possible": np.random.choice(["Yes", "No"], size=n),
        "Subsidy_Available": np.random.choice(["Yes", "No"], size=n),
        "Range_Anxiety_Level": np.random.choice(["Low", "Medium", "High"], size=n),
        TARGET: np.random.choice([0, 1], size=n),
    }
    train_df = pd.DataFrame(data)
    test_df = train_df.drop(columns=[TARGET]).copy()
    test_df["id"] = range(n, 2 * n)

    orig_df = train_df.iloc[:20].copy()
    orig_df[TARGET] = ["Yes" if x == 1 else "No" for x in orig_df[TARGET]]

    tr_feat, te_feat, features, te_cols = build_grandmaster_features(train_df, test_df, orig_df)

    assert len(tr_feat) == n
    assert len(te_feat) == n
    assert TARGET in tr_feat.columns
    assert TARGET not in te_feat.columns
    assert len(features) > 20
    assert len(te_cols) > 0

    # Ensure no NaN values in numeric features
    num_cols = [c for c in features if c not in te_cols]
    assert not tr_feat[num_cols].isnull().any().any(), "Train features contain NaNs!"
    assert not te_feat[num_cols].isnull().any().any(), "Test features contain NaNs!"
