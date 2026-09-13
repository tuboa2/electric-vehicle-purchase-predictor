"""
Domain Feature Generator for playground-series-s6e9 (EV Purchase Prediction).
Implements hypothesis-driven feature transformations:
1. Total Charging Availability: Home + Work stations
2. Commute-to-Charging Ratio: Daily distance normalized by local infrastructure
3. Economic Cushion per Vehicle: Annual income divided by owned fleet size
4. Behavioral Gating: Subsidy Available flag multiplying Environmental Concern
5. Range Anxiety Penalty: Distance amplified by psychological range anxiety severity
6. Peer Group Residuals: Deviations from City_Type x Current_Car_Type mean income and commute
"""

from typing import Tuple
import numpy as np
import pandas as pd


def generate_domain_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Applies deterministic domain interaction and aggregation features.
    Guaranteed zero target leakage: only unsupervised and row-wise features.
    Train-set peer statistics are strictly applied to test-set without re-estimation.
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

        # 4. Behavioral Gating: Subsidy x Environmental Concern
        # EDA proved Subsidy='No' drops adoption to 0.58%; environmental desire only acts when financially unlocked
        subsidy_flag = (df["Subsidy_Available"] == "Yes").astype(float)
        df["feat_subsidy_env_gate"] = subsidy_flag * df["Environmental_Concern_Level"].astype(float)

        # 5. Commute x Range Anxiety Penalty
        anxiety_score = df["Range_Anxiety_Level"].map(anxiety_map).fillna(1.0).astype(float)
        df["feat_commute_anxiety_penalty"] = df["Daily_Commute_km"].astype(float) * anxiety_score

        # 6. Compound Categorical Tuples for tree splitting
        df["feat_city_car"] = df["City_Type"].astype(str) + "_" + df["Current_Car_Type"].astype(str)
        df["feat_car_charging"] = df["Current_Car_Type"].astype(str) + "_" + df["Home_Charging_Possible"].astype(str)

    new_cols.extend([
        "feat_total_charging",
        "feat_commute_charging_ratio",
        "feat_income_per_car",
        "feat_subsidy_env_gate",
        "feat_commute_anxiety_penalty",
        "feat_city_car",
        "feat_car_charging",
    ])

    # 7. Groupby Peer Aggregations (Residuals relative to City_Type x Current_Car_Type)
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
