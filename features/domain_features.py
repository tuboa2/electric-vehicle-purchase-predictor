"""
Domain Feature Generator for playground-series-s6e9 (EV Purchase Prediction).
Implements hypothesis-driven feature transformations:
1. Infrastructure Ratios:
   - Total Charging Availability: Home + Work stations
   - Commute-to-Charging Ratio: Daily distance normalized by local infrastructure
2. Economic Capacity:
   - Economic Cushion per Vehicle: Annual income divided by owned fleet size
   - Income relative to Age: Income per year of working age
3. Behavioral Gating & Psychology:
   - Subsidy Available flag multiplying Environmental Concern
   - Commute Distance amplified by Range Anxiety Severity
   - Environmental Concern vs Range Anxiety tension ratio
4. Higher-Order Categorical Compound Tuples:
   - City x Car Type
   - Car Type x Home Charging
   - Car Type x Subsidy
   - City x Subsidy
   - Full Profile: City x Car x Subsidy
5. Frequency Encoding:
   - Normalized prevalence of single and compound categorical combinations
6. Peer Group Residuals:
   - Deviations from City_Type x Current_Car_Type mean income and commute
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_domain_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Applies deterministic domain interaction, frequency encoding, and aggregation features.
    Guaranteed zero target leakage: only unsupervised and row-wise features.
    Train-set peer statistics and frequencies are strictly applied to test-set without re-estimation.
    """
    df_tr = train_df.copy()
    df_te = test_df.copy()

    anxiety_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    new_cols = []

    for df in [df_tr, df_te]:
        # 1. Total Charging Availability (Physical infrastructure density)
        df["feat_total_charging"] = (
            df["Charging_Stations_Near_Home"].astype(float)
            + df["Charging_Stations_Near_Work"].astype(float)
        )

        # 2. Commute to Charging Ratio (Vulnerability to range depletion)
        df["feat_commute_charging_ratio"] = (
            df["Daily_Commute_km"].astype(float) / (df["feat_total_charging"] + 1.0)
        )

        # 3. Economic Capacity per Owned Vehicle (Discretionary capital flexibility)
        df["feat_income_per_car"] = (
            df["Annual_Income_USD"].astype(float) / (df["Number_of_Cars_Owned"].astype(float) + 1.0)
        )

        # 4. Income normalized by Age (Earning maturity)
        df["feat_income_per_age"] = (
            df["Annual_Income_USD"].astype(float) / df["Age"].astype(float).clip(lower=18.0)
        )

        # 5. Behavioral Gating: Subsidy x Environmental Concern
        # EDA proved Subsidy='No' drops adoption to 0.58%; environmental desire only acts when financially unlocked
        subsidy_flag = (df["Subsidy_Available"] == "Yes").astype(float)
        df["feat_subsidy_env_gate"] = subsidy_flag * df["Environmental_Concern_Level"].astype(float)

        # 6. Commute x Range Anxiety Penalty
        anxiety_score = df["Range_Anxiety_Level"].map(anxiety_map).fillna(1.0).astype(float)
        df["feat_commute_anxiety_penalty"] = df["Daily_Commute_km"].astype(float) * anxiety_score

        # 7. Psychological Tension: Environmental Motivation vs Range Anxiety Pushback
        df["feat_env_anxiety_ratio"] = (
            df["Environmental_Concern_Level"].astype(float) / (anxiety_score + 0.5)
        )

        # 8. Compound Categorical Tuples
        df["feat_city_car"] = df["City_Type"].astype(str) + "_" + df["Current_Car_Type"].astype(str)
        df["feat_car_charging"] = df["Current_Car_Type"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)
        df["feat_car_subsidy"] = df["Current_Car_Type"].astype(str) + "_" + df["Subsidy_Available"].astype(str)
        df["feat_city_subsidy"] = df["City_Type"].astype(str) + "_" + df["Subsidy_Available"].astype(str)
        df["feat_city_car_subsidy"] = (
            df["City_Type"].astype(str) + "_" + df["Current_Car_Type"].astype(str) + "_" + df["Subsidy_Available"].astype(str)
        )

    new_cols.extend([
        "feat_total_charging",
        "feat_commute_charging_ratio",
        "feat_income_per_car",
        "feat_income_per_age",
        "feat_subsidy_env_gate",
        "feat_commute_anxiety_penalty",
        "feat_env_anxiety_ratio",
        "feat_city_car",
        "feat_car_charging",
        "feat_car_subsidy",
        "feat_city_subsidy",
        "feat_city_car_subsidy",
    ])

    # 9. Frequency Encoding: Compute normalized value counts on train, map to test
    freq_cols = [
        "City_Type", "Current_Car_Type",
        "feat_city_car", "feat_car_charging", "feat_car_subsidy", "feat_city_car_subsidy"
    ]
    for fc in freq_cols:
        freq_name = f"freq_{fc}"
        val_counts = df_tr[fc].value_counts(normalize=True).to_dict()
        df_tr[freq_name] = df_tr[fc].map(val_counts).fillna(0.0).astype(float)
        df_te[freq_name] = df_te[fc].map(val_counts).fillna(0.0).astype(float)
        new_cols.append(freq_name)

    # 10. Groupby Peer Aggregations (Residuals relative to City_Type x Current_Car_Type)
    group_cols = ["City_Type", "Current_Car_Type"]
    group_income_stats = df_tr.groupby(group_cols)["Annual_Income_USD"].agg(["mean", "std"]).reset_index()
    group_income_stats.columns = group_cols + ["peer_income_mean", "peer_income_std"]

    group_commute_stats = df_tr.groupby(group_cols)["Daily_Commute_km"].agg(["mean"]).reset_index()
    group_commute_stats.columns = group_cols + ["peer_commute_mean"]

    df_tr = df_tr.merge(group_income_stats, on=group_cols, how="left")
    df_tr = df_tr.merge(group_commute_stats, on=group_cols, how="left")
    df_te = df_te.merge(group_income_stats, on=group_cols, how="left")
    df_te = df_te.merge(group_commute_stats, on=group_cols, how="left")

    for df in [df_tr, df_te]:
        df["feat_income_peer_diff"] = df["Annual_Income_USD"] - df["peer_income_mean"]
        df["feat_commute_peer_diff"] = df["Daily_Commute_km"] - df["peer_commute_mean"]

    new_cols.extend(["feat_income_peer_diff", "feat_commute_peer_diff"])

    drop_cols = ["peer_income_mean", "peer_income_std", "peer_commute_mean"]
    df_tr.drop(columns=drop_cols, inplace=True, errors="ignore")
    df_te.drop(columns=drop_cols, inplace=True, errors="ignore")

    return df_tr, df_te, new_cols
