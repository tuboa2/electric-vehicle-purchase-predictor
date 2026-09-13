import pandas as pd
import numpy as np
from experiments.templates.features.domain_features import generate_domain_features


def test_domain_features_generation():
    train_df = pd.DataFrame({
        "id": [1, 2, 3],
        "Charging_Stations_Near_Home": [2, 0, 5],
        "Charging_Stations_Near_Work": [3, 1, 0],
        "Daily_Commute_km": [25.0, 50.0, 10.0],
        "Annual_Income_USD": [80000.0, 45000.0, 120000.0],
        "Number_of_Cars_Owned": [1, 2, 1],
        "Subsidy_Available": ["Yes", "No", "Yes"],
        "Environmental_Concern_Level": [4, 2, 5],
        "Range_Anxiety_Level": ["Low", "High", "Medium"],
        "City_Type": ["Urban", "Rural", "Suburban"],
        "Current_Car_Type": ["Sedan", "SUV", "Sedan"],
        "Will_Buy_EV": ["Yes", "No", "Yes"]
    })

    test_df = train_df.drop(columns=["Will_Buy_EV"]).copy()

    tr_feat, te_feat, new_cols = generate_domain_features(train_df, test_df)

    assert len(tr_feat) == len(train_df)
    assert len(te_feat) == len(test_df)
    for col in new_cols:
        assert col in tr_feat.columns
        assert col in te_feat.columns
        assert not tr_feat[col].isnull().any()
        assert not te_feat[col].isnull().any()
