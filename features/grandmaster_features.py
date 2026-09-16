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
from scipy.special import ndtr


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

    # Continuous Financial & Commute Dynamics
    income_val = combined["Annual_Income_USD"].astype(float)
    commute_val = combined["Daily_Commute_km"].astype(float)
    age_val = combined["Age"].astype(float)
    subsidy_bin = (combined["Subsidy_Available"] == "Yes").astype(float)
    env_concern = combined["Environmental_Concern_Level"].astype(float)

    # 0. Chris Deotte & Fable 5.1 Generator Recipe (0.93769 baseline signal)
    anx_med = (combined["Range_Anxiety_Level"] == "Medium").astype(float)
    anx_high = (combined["Range_Anxiety_Level"] == "High").astype(float)
    buy_recipe_score = (
        1.2 * (income_val / 100000.0)
        + 0.6 * env_concern
        + 2.0 * subsidy_bin
        - 1.0 * anx_med
        - 3.0 * anx_high
    )
    recipe_diff = (buy_recipe_score - 5.61235).values
    abs_recipe_diff = np.abs(recipe_diff)
    new_features["feat_buy_recipe_score"] = buy_recipe_score.values
    new_features["feat_recipe_dist_to_boundary"] = recipe_diff
    new_features["feat_recipe_abs_dist"] = abs_recipe_diff
    new_features["feat_recipe_z_sq"] = (recipe_diff ** 2).astype("float32")
    new_features["feat_recipe_z_cube"] = (recipe_diff ** 3).astype("float32")
    new_features["feat_recipe_sign_z"] = np.sign(recipe_diff).astype("int8")
    new_features["feat_recipe_is_above_boundary"] = (recipe_diff > 0.0).astype("int8")
    new_features["feat_recipe_prob_logit"] = (
        1.0 / (1.0 + np.exp(-np.clip(recipe_diff * 2.17464, -35.0, 35.0)))
    ).astype("float32")
    new_features["feat_recipe_prob_probit"] = (
        ndtr(np.clip(recipe_diff / 0.834476, -8.0, 8.0))
    ).astype("float32")
    new_features["feat_recipe_base_margin"] = (
        np.clip(recipe_diff * 2.17464, -15.0, 15.0)
    ).astype("float32")

    # Boundary High-Uncertainty Zones (where 95% of residual classification errors occur)
    new_features["feat_is_boundary_10"] = (abs_recipe_diff < 0.10).astype("int8")
    new_features["feat_is_boundary_20"] = (abs_recipe_diff < 0.20).astype("int8")
    new_features["feat_is_boundary_30"] = (abs_recipe_diff < 0.30).astype("int8")
    new_features["feat_is_boundary_50"] = (abs_recipe_diff < 0.50).astype("int8")
    new_features["feat_boundary_gaussian_weight"] = np.exp(-((recipe_diff / 0.30) ** 2)).astype("float32")

    # Generator Modulo & Floor Artifacts (exploding synthetic discrete quantization)
    new_features["feat_age_mod_2"] = (age_val % 2).astype("int8").values
    new_features["feat_age_mod_3"] = (age_val % 3).astype("int8").values
    new_features["feat_age_mod_5"] = (age_val % 5).astype("int8").values
    new_features["feat_age_mod_7"] = (age_val % 7).astype("int8").values
    new_features["feat_age_bin_5"] = (age_val // 5).astype("int8").values
    new_features["feat_income_dist_from_30k"] = (income_val - 30000.0).astype("float32").values
    new_features["feat_income_above_30k"] = (income_val >= 30000.0).astype("int8").values
    new_features["feat_income_x_env"] = ((income_val / 100000.0) * env_concern).astype("float32").values
    new_features["feat_income_x_subsidy"] = ((income_val / 100000.0) * subsidy_bin).astype("float32").values

    # Domain Interactions
    new_features["feat_subsidy_env_gate"] = (subsidy_bin * env_concern).values
    total_charging = (
        combined["Charging_Stations_Near_Home"].astype(float)
        + combined["Charging_Stations_Near_Work"].astype(float)
    )
    new_features["feat_total_charging"] = total_charging.values
    new_features["feat_commute_charging_ratio"] = (
        commute_val / (total_charging + 1.0)
    ).values

    # EV Charger "Staircase" & Simpson's Paradox Inversion (Strict Native City_Type Cohorts)
    home_charging = combined["Charging_Stations_Near_Home"].astype(float)
    work_charging = combined["Charging_Stations_Near_Work"].astype(float)
    city_home_mean = combined.groupby("City_Type", observed=False)["Charging_Stations_Near_Home"].transform("mean")
    city_home_std = combined.groupby("City_Type", observed=False)["Charging_Stations_Near_Home"].transform("std").fillna(1.0)
    city_work_mean = combined.groupby("City_Type", observed=False)["Charging_Stations_Near_Work"].transform("mean")
    city_work_std = combined.groupby("City_Type", observed=False)["Charging_Stations_Near_Work"].transform("std").fillna(1.0)

    total_charging_s = pd.Series(total_charging.values, index=combined.index)
    city_total_mean = total_charging_s.groupby(combined["City_Type"], observed=False).transform("mean")
    city_total_std = total_charging_s.groupby(combined["City_Type"], observed=False).transform("std").fillna(1.0)

    # Within-Cohort Z-Scores (strictly inverting Simpson's paradox: sign flips from negative to positive)
    new_features["feat_charging_home_city_diff"] = (home_charging - city_home_mean).values
    new_features["feat_charging_work_city_diff"] = (work_charging - city_work_mean).values
    new_features["feat_charging_home_z_city"] = ((home_charging - city_home_mean) / (city_home_std + 1e-6)).astype("float32").values
    new_features["feat_charging_work_z_city"] = ((work_charging - city_work_mean) / (city_work_std + 1e-6)).astype("float32").values
    new_features["feat_total_charging_z_city"] = ((total_charging - city_total_mean) / (city_total_std + 1e-6)).astype("float32").values

    # Explicit City_Type Cohort One-Hot Multiplicative Interactions
    is_urban = (combined["City_Type"] == "Urban").astype("float32")
    is_suburban = (combined["City_Type"] == "Suburban").astype("float32")
    is_rural = (combined["City_Type"] == "Rural").astype("float32")
    charging_per_km = (total_charging / (commute_val + 1.0)).astype("float32")

    new_features["feat_urban_x_charging_per_km"] = (is_urban * charging_per_km).values
    new_features["feat_suburban_x_charging_per_km"] = (is_suburban * charging_per_km).values
    new_features["feat_rural_x_charging_per_km"] = (is_rural * charging_per_km).values
    new_features["feat_urban_x_commute"] = (is_urban * commute_val).astype("float32").values
    new_features["feat_suburban_x_commute"] = (is_suburban * commute_val).astype("float32").values
    new_features["feat_rural_x_commute"] = (is_rural * commute_val).astype("float32").values
    new_features["feat_urban_x_home_charging"] = (is_urban * home_charging).astype("float32").values
    new_features["feat_suburban_x_home_charging"] = (is_suburban * home_charging).astype("float32").values
    new_features["feat_rural_x_home_charging"] = (is_rural * home_charging).astype("float32").values

    # Procedural Hard Saturation Flags (mirroring synthetic data generation infrastructure caps)
    new_features["feat_is_rural_home_capped"] = (is_rural * (home_charging >= 3.0)).astype("int8").values
    new_features["feat_is_suburban_home_capped"] = (is_suburban * (home_charging >= 9.0)).astype("int8").values
    new_features["feat_is_urban_home_capped"] = (is_urban * (home_charging >= 14.0)).astype("int8").values

    # Chris Deotte Simpson's Paradox Resolution (Home Charging Ability vs Station Density)
    mean_home_by_ability_city = combined.groupby(["Home_Charging_Possible", "City_Type"], observed=False)["Charging_Stations_Near_Home"].transform("mean")
    new_features["feat_charging_home_simpson_diff"] = (home_charging - mean_home_by_ability_city).astype("float32").values
    can_charge_bin = (combined["Home_Charging_Possible"] == "Yes").astype("float32")
    new_features["feat_home_charge_ability_x_stations"] = (can_charge_bin * home_charging).astype("float32").values
    new_features["feat_no_home_charge_x_stations"] = ((1.0 - can_charge_bin) * home_charging).astype("float32").values

    anxiety_map = {"Low": 1.0, "Medium": 2.0, "High": 3.0}
    anx_val = combined["Range_Anxiety_Level"].map(anxiety_map).fillna(1.0).astype(float)
    new_features["feat_env_anxiety_ratio"] = (
        env_concern / (anx_val + 0.5)
    ).values

    new_features["feat_income_per_age"] = (income_val / (age_val + 1.0)).values
    new_features["feat_income_per_commute"] = (income_val / (commute_val + 1.0)).values
    new_features["feat_commute_per_age"] = (commute_val / (age_val + 1.0)).values
    new_features["feat_charging_density_diff"] = (home_charging - work_charging).values
    new_features["feat_charging_home_work_ratio"] = ((home_charging + 1.0) / (work_charging + 1.0)).values
    new_features["feat_commute_x_anxiety"] = (commute_val * anx_val).astype("float32").values
    if "Number_of_Cars_Owned" in combined.columns:
        cars_owned = combined["Number_of_Cars_Owned"].astype(float)
        new_features["feat_income_per_car"] = (income_val / (cars_owned + 1.0)).astype("float32").values

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

    # 1. Exact Float-Safe Digit Decomposition (eliminates IEEE 754 precision noise)
    digit_feature_names = []
    continuous_to_decompose = [
        "Annual_Income_USD",
        "Daily_Commute_km",
        "Age",
        "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work",
    ]
    for c in continuous_to_decompose:
        if c in combined.columns:
            col_series = combined[c].fillna(0).astype(float)
            for k in range(0, 5):
                col_name = f"{c}_d10p{k}"
                new_features[col_name] = ((np.floor(col_series) // (10**k)) % 10).astype("int8").values
                digit_feature_names.append(col_name)
            for k in range(1, 4):
                col_name = f"{c}_d10m{k}"
                new_features[col_name] = (np.floor(np.round(col_series * (10**k), 4)) % 10).astype("int8").values
                digit_feature_names.append(col_name)

    all_num_cols = list(num_cols) + digit_feature_names

    # 2. Original Dataset Target Means (if available)
    if orig_df is not None and not orig_df.empty:
        orig = orig_df.copy()
        if TARGET in orig.columns:
            if not pd.api.types.is_numeric_dtype(orig[TARGET]):
                orig[TARGET] = orig[TARGET].astype(str).map({"Yes": 1, "No": 0, "1": 1, "0": 0}).fillna(0).astype(float)
        orig_global_mean = float(orig[TARGET].mean())

        for col in cat_cols + num_cols:
            if col in orig.columns:
                real_world_stats = orig.groupby(col, observed=False)[TARGET].mean()
                if col in combined.columns:
                    col_data = combined[col]
                elif col in new_features:
                    col_data = pd.Series(new_features[col], index=combined.index)
                else:
                    continue
                new_features[f"{col}_org_mean"] = (
                    col_data.map(real_world_stats).fillna(orig_global_mean).astype(float).values
                )

    # 3. Hard Edge / Magic Boundary Flags
    income = combined["Annual_Income_USD"]
    new_features["is_30k_spike"] = (income == 30000.0).astype("int8").values
    new_features["is_millionaire_cliff"] = (income >= 170537.0).astype("int8").values
    new_features["is_dead_zone"] = ((income >= 38000.0) & (income <= 42000.0)).astype("int8").values
    new_features["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype("int8").values

    # 4. Smooth Keys (Binned Numerics)
    new_features["income_exact_int"] = np.floor(income).astype(str).values
    new_features["income100_floor"] = np.floor(income / 100.0).astype(str).values
    new_features["income1000_floor"] = np.floor(income / 1000.0).astype(str).values
    new_features["commute_integer"] = np.floor(combined["Daily_Commute_km"]).astype(str).values
    new_features["commute_10km_floor"] = np.floor(combined["Daily_Commute_km"] / 10.0).astype(str).values
    new_features["age_decade_floor"] = np.floor(combined["Age"] / 10.0).astype(str).values

    # 5. Global Frequency Encoding on High-Signal Categoricals & Bins (excluding raw single digits)
    all_cats = list(cat_cols) + [
        "income_exact_int",
        "income100_floor",
        "income1000_floor",
        "commute_integer",
        "commute_10km_floor",
        "age_decade_floor",
    ]
    for col in all_cats:
        if col in combined.columns:
            val_series = combined[col]
        else:
            val_series = pd.Series(new_features[col], index=combined.index)
        freq_mapping = val_series.value_counts(normalize=True).to_dict()
        new_features[f"{col}_fe"] = val_series.map(freq_mapping).astype(float).fillna(0.0).values

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
