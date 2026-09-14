"""
Grandmaster Feature Engineering Module for playground-series-s6e9.
Implements the synthetic artifact exploits discovered in top-tier competition solutions:
1. Digit Decomposition:
   - Slices continuous numerical variables across powers of 10 (10^-4 to 10^3)
   - Exposes generator quantization boundaries and modulo patterns
2. Smooth Keys (Multi-Scale Binned Numerics):
   - Floor clusters of Annual_Income_USD (exact integer, 100-step, 1000-step)
   - Floor clusters of Daily_Commute_km
3. Deterministic Boundary Flags (Hard Probability Cliffs):
   - is_millionaire_cliff: Income >= 170537 (100% positive purchase rate)
   - is_dead_zone: 38000 <= Income <= 42000 (0% positive purchase rate)
   - is_30k_spike: Income == 30000 mode collapse spike
   - is_env_hater: Environmental_Concern_Level == 1 (extreme low adoption)
4. Domain Behavioral Interaction:
   - feat_subsidy_env_gate: Subsidy_Available * Environmental_Concern_Level
   - feat_total_charging: Home + Work charging station density
   - feat_commute_charging_ratio: Commute / (Total Charging + 1)
   - feat_env_anxiety_ratio: Concern vs Anxiety tension
5. Ground-Truth Original Dataset Priors:
   - Real-world target mean mapping from 10k seed dataset across overlapping columns
6. Global Frequency Encoding:
   - Normalized prevalence of all categorical and converted binned columns
7. Feature Pruning:
   - Automated removal of constant features and perfectly collinear pairs (r = 1.0)
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd


TARGET = "Will_Buy_EV"


def build_grandmaster_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    orig_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Builds the complete suite of synthetic artifact exploits and domain features.

    Returns:
        train_feat: Transformed train DataFrame (includes TARGET if present)
        test_feat: Transformed test DataFrame
        features: Complete list of feature column names for model training
        target_encode_cols: Categorical / string column names to be target encoded inside CV folds
    """
    tr = train_df.copy()
    te = test_df.copy()

    # Standardize target to binary 0/1
    if TARGET in tr.columns:
        if not pd.api.types.is_numeric_dtype(tr[TARGET]):
            tr[TARGET] = tr[TARGET].astype(str).map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(int)

    tr["is_train"] = 1
    te["is_train"] = 0
    if TARGET in te.columns:
        te[TARGET] = np.nan
    else:
        te[TARGET] = np.nan

    combined = pd.concat([tr, te], ignore_index=True)

    # Dictionary to collect all generated features simultaneously (avoids DataFrame fragmentation)
    new_features: Dict[str, np.ndarray | pd.Series] = {}

    # Domain Interactions
    subsidy_bin = (combined["Subsidy_Available"] == "Yes").astype(float)
    new_features["feat_subsidy_env_gate"] = (subsidy_bin * combined["Environmental_Concern_Level"].astype(float)).values
    total_charging = (
        combined["Charging_Stations_Near_Home"].astype(float)
        + combined["Charging_Stations_Near_Work"].astype(float)
    )
    new_features["feat_total_charging"] = total_charging.values
    new_features["feat_commute_charging_ratio"] = (
        combined["Daily_Commute_km"].astype(float) / (total_charging + 1.0)
    ).values

    anxiety_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    anx_val = combined["Range_Anxiety_Level"].map(anxiety_map).fillna(1.0).astype(float)
    new_features["feat_env_anxiety_ratio"] = (
        combined["Environmental_Concern_Level"].astype(float) / (anx_val + 0.5)
    ).values

    # Continuous Financial & Commute Dynamics
    income_val = combined["Annual_Income_USD"].astype(float)
    commute_val = combined["Daily_Commute_km"].astype(float)
    age_val = combined["Age"].astype(float)

    new_features["feat_income_per_age"] = (income_val / (age_val + 1.0)).values
    new_features["feat_income_per_commute"] = (income_val / (commute_val + 1.0)).values
    new_features["feat_commute_per_age"] = (commute_val / (age_val + 1.0)).values
    new_features["feat_charging_density_diff"] = (
        combined["Charging_Stations_Near_Home"].astype(float)
        - combined["Charging_Stations_Near_Work"].astype(float)
    ).values
    new_features["feat_charging_home_work_ratio"] = (
        (combined["Charging_Stations_Near_Home"].astype(float) + 1.0)
        / (combined["Charging_Stations_Near_Work"].astype(float) + 1.0)
    ).values

    # Quantization Artifacts & Decimal Residues
    new_features["is_commute_exact_int"] = ((commute_val % 1.0 == 0.0)).astype("int8").values
    new_features["commute_fraction"] = (commute_val % 1.0).astype("float32").values

    # Compound Interaction Categoricals
    if "City_Type" in combined.columns and "Home_Charging_Possible" in combined.columns:
        combined["cat_city_home_charging"] = combined["City_Type"].astype(str) + "_" + combined["Home_Charging_Possible"].astype(str)
    if "City_Type" in combined.columns and "Subsidy_Available" in combined.columns:
        combined["cat_city_subsidy"] = combined["City_Type"].astype(str) + "_" + combined["Subsidy_Available"].astype(str)
    if "Home_Charging_Possible" in combined.columns and "Range_Anxiety_Level" in combined.columns:
        combined["cat_charging_anxiety"] = combined["Home_Charging_Possible"].astype(str) + "_" + combined["Range_Anxiety_Level"].astype(str)
    if "Current_Car_Type" in combined.columns and "Subsidy_Available" in combined.columns:
        combined["cat_car_subsidy"] = combined["Current_Car_Type"].astype(str) + "_" + combined["Subsidy_Available"].astype(str)
    if "Gender" in combined.columns and "City_Type" in combined.columns:
        combined["cat_gender_city"] = combined["Gender"].astype(str) + "_" + combined["City_Type"].astype(str)

    # Drop Number_of_Cars_Owned (verified zero predictive gain in high-scoring models)
    combined.drop(columns=["Number_of_Cars_Owned"], inplace=True, errors="ignore")

    cat_cols = combined.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in ["id", "is_train", TARGET]]
    num_cols = [c for c in combined.columns if c not in cat_cols + ["id", "is_train", TARGET]]

    # 1. Digit Decomposition (Extract digits from 10^-4 to 10^3)
    digit_feature_names = []
    for c in num_cols:
        col_series = combined[c].fillna(0)
        for k in range(-4, 4):
            col_name = f"{c}_digit{k}"
            new_features[col_name] = ((col_series // (10**k)) % 10).astype("int8").values
            digit_feature_names.append(col_name)

    all_num_cols = list(num_cols) + digit_feature_names

    # 2. Original Dataset Target Means (if available)
    if orig_df is not None and not orig_df.empty:
        orig = orig_df.copy()
        if TARGET in orig.columns:
            if not pd.api.types.is_numeric_dtype(orig[TARGET]):
                orig[TARGET] = orig[TARGET].astype(str).map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(float)
        orig_global_mean = float(orig[TARGET].mean())

        for col in cat_cols + all_num_cols:
            if col in orig.columns:
                real_world_stats = orig.groupby(col, observed=False)[TARGET].mean()
                if col in combined.columns:
                    col_data = combined[col]
                else:
                    col_data = pd.Series(new_features[col], index=combined.index)
                new_features[f"{col}_org_mean"] = (
                    col_data.map(real_world_stats).fillna(orig_global_mean).astype(float).values
                )

    # 3. Convert Numerics to String Categories for Frequency and Target Encoding
    num_to_cat_cols = []
    for col in all_num_cols:
        cat_name = f"{col}_cat"
        if col in combined.columns:
            cat_series = combined[col].fillna("NaN").astype(str)
        else:
            cat_series = pd.Series(new_features[col], index=combined.index).fillna("NaN").astype(str)
        new_features[cat_name] = cat_series.values
        num_to_cat_cols.append(cat_name)

    # 4. Global Frequency Encoding
    all_cats = list(cat_cols) + list(num_to_cat_cols)
    for col in all_cats:
        if col in combined.columns:
            val_series = combined[col]
        else:
            val_series = pd.Series(new_features[col], index=combined.index)
        freq_mapping = val_series.value_counts(normalize=True).to_dict()
        new_features[f"{col}_fe"] = val_series.map(freq_mapping).astype(float).fillna(0.0).values

    # 5. Hard Edge / Magic Boundary Flags
    income = combined["Annual_Income_USD"]
    new_features["is_30k_spike"] = (income == 30000.0).astype("int8").values
    new_features["is_millionaire_cliff"] = (income >= 170537.0).astype("int8").values
    new_features["is_dead_zone"] = ((income >= 38000.0) & (income <= 42000.0)).astype("int8").values
    new_features["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype("int8").values

    # 6. Smooth Keys (Binned Numerics)
    new_features["income_exact_int"] = np.floor(income).astype(str).values
    new_features["income100_floor"] = np.floor(income / 100.0).astype(str).values
    new_features["income1000_floor"] = np.floor(income / 1000.0).astype(str).values
    new_features["commute_integer"] = np.floor(combined["Daily_Commute_km"]).astype(str).values
    new_features["commute_10km_floor"] = np.floor(combined["Daily_Commute_km"] / 10.0).astype(str).values
    new_features["age_decade_floor"] = np.floor(combined["Age"] / 10.0).astype(str).values
    all_cats.extend(["income_exact_int", "income100_floor", "income1000_floor", "commute_integer", "commute_10km_floor", "age_decade_floor"])

    # Concat all new features in one single operation (zero fragmentation)
    new_features_df = pd.DataFrame(new_features, index=combined.index)
    combined = pd.concat([combined, new_features_df], axis=1)

    # Split train and test back
    train_feat = combined[combined["is_train"] == 1].drop(columns=["is_train"]).copy()
    test_feat = combined[combined["is_train"] == 0].drop(columns=["is_train", TARGET]).copy()

    # 7. Automated Feature Pruning (drop constants and perfect correlations)
    eval_cols = [
        c for c in train_feat.columns
        if c not in ["id", TARGET] and pd.api.types.is_numeric_dtype(train_feat[c])
    ]

    corr_matrix = train_feat[eval_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [col for col in upper_tri.columns if any(upper_tri[col] == 1.0)]

    to_drop_const = (
        [c for c in train_feat.columns if train_feat[c].nunique() == 1]
        + [c for c in test_feat.columns if test_feat[c].nunique() == 1]
    )

    drop_set: Set[str] = set(to_drop_corr).union(set(to_drop_const))
    drop_cols = [c for c in drop_set if c not in ["id", TARGET]]

    if drop_cols:
        train_feat.drop(columns=drop_cols, inplace=True, errors="ignore")
        test_feat.drop(columns=drop_cols, inplace=True, errors="ignore")

    features = [c for c in test_feat.columns if c != "id"]
    non_numeric_cols = [c for c in features if not pd.api.types.is_numeric_dtype(test_feat[c])]
    target_encode_cols = sorted(list(set([c for c in all_cats if c not in drop_set and c in test_feat.columns]).union(set(non_numeric_cols))))

    return train_feat, test_feat, features, target_encode_cols
