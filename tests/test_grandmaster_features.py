"""
Unit tests for Grandmaster Feature Engineering module and fold processing.
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import TargetEncoder
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


def test_te_fold_concatenation_no_warning():
    """Verify that Target Encoding fold concatenation generates zero PerformanceWarning."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C"], size=n),
        "cat2": np.random.choice(["X", "Y", "Z"], size=n),
        "num": np.random.randn(n),
    })
    y = np.random.choice([0, 1], size=n)
    te_cols = ["cat1", "cat2"]

    te = TargetEncoder(smooth=10.0, random_state=42)
    enc = te.fit_transform(df[te_cols], y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        te_data = {}
        for idx, col in enumerate(te_cols):
            te_data[f"{col}_TE"] = enc[:, idx].astype("float32")
        res = pd.concat([df.drop(columns=te_cols), pd.DataFrame(te_data, index=df.index)], axis=1)

    assert "cat1_TE" in res.columns
    assert "cat2_TE" in res.columns
    assert "cat1" not in res.columns
    assert len(res) == n


def test_grandmaster_features_new_interactions():
    """Verify newly added compound categoricals, ratios, and quantization flags exist."""
    np.random.seed(42)
    n = 100
    data = {
        "id": range(n),
        "Age": np.random.randint(20, 70, size=n),
        "Annual_Income_USD": np.random.uniform(30000, 180000, size=n),
        "Daily_Commute_km": np.random.choice([25.0, 30.5, 45.12, 50.0], size=n),
        "Charging_Stations_Near_Home": np.random.randint(0, 15, size=n),
        "Charging_Stations_Near_Work": np.random.randint(0, 20, size=n),
        "Environmental_Concern_Level": np.random.randint(1, 6, size=n),
        "Gender": np.random.choice(["Male", "Female"], size=n),
        "City_Type": np.random.choice(["Urban", "Suburban"], size=n),
        "Current_Car_Type": np.random.choice(["Sedan", "SUV"], size=n),
        "Home_Charging_Possible": np.random.choice(["Yes", "No"], size=n),
        "Subsidy_Available": np.random.choice(["Yes", "No"], size=n),
        "Range_Anxiety_Level": np.random.choice(["Low", "High"], size=n),
        TARGET: np.random.choice([0, 1], size=n),
    }
    train_df = pd.DataFrame(data)
    test_df = train_df.drop(columns=[TARGET]).copy()
    test_df["id"] = range(n, 2 * n)

    tr_feat, te_feat, features, te_cols = build_grandmaster_features(train_df, test_df)

    assert "feat_income_per_age" in features
    assert "feat_income_per_commute" in features
    assert "feat_commute_per_age" in features
    assert "is_commute_exact_int" in features
    assert "commute_fraction" in features
    assert any("cat_city_home_charging" in c for c in features)


def test_grandmaster_triple_blend_nelder_mead(tmp_path):
    """Verify Triple Optimized Nelder-Mead Blending (Prob + Rank + Logit)."""
    from scripts.kaggle_train_grandmaster import blend_grandmaster_models

    np.random.seed(42)
    n = 200
    y_true = np.random.choice([0, 1], size=n)
    
    # Generate mock predictions
    m1_oof = np.clip(y_true * 0.7 + np.random.normal(0, 0.2, n), 0.01, 0.99)
    m2_oof = np.clip(y_true * 0.75 + np.random.normal(0, 0.25, n), 0.01, 0.99)
    m1_test = np.clip(np.random.uniform(0.1, 0.9, n), 0.01, 0.99)
    m2_test = np.clip(np.random.uniform(0.1, 0.9, n), 0.01, 0.99)

    models_oof = {"lgbm": m1_oof, "catboost": m2_oof}
    models_test = {"lgbm": m1_test, "catboost": m2_test}
    test_ids = pd.Series(range(n))

    out_dir = tmp_path / "models"
    primary_sub = tmp_path / "submission.csv"

    blend_grandmaster_models(
        models_oof=models_oof,
        models_test=models_test,
        y_true=y_true,
        test_ids=test_ids,
        output_dir=out_dir,
        primary_sub_path=primary_sub,
    )

    assert primary_sub.exists()
    sub_df = pd.read_csv(primary_sub)
    assert len(sub_df) == n
    assert TARGET in sub_df.columns
    assert (tmp_path / "ensemble_grandmaster" / "metrics.json").exists()


def test_train_single_model_smoke(tmp_path):
    """Verify train_single_model runs end-to-end without NameError or missing variables."""
    from scripts.kaggle_train_grandmaster import train_single_model, prepare_seed_folds

    np.random.seed(42)
    n = 100
    data = {
        "id": range(n),
        "Age": np.random.randint(18, 70, size=n),
        "Annual_Income_USD": np.random.randint(20000, 150000, size=n),
        "Daily_Commute_km": np.random.uniform(5.0, 100.0, size=n),
        "Environmental_Concern_Level": np.random.randint(1, 5, size=n),
        "Subsidy_Available": np.random.choice(["Yes", "No"], size=n),
        "Range_Anxiety_Level": np.random.choice(["Low", "High"], size=n),
        "City_Type": np.random.choice(["Urban", "Rural"], size=n),
        "Current_Car_Type": np.random.choice(["SUV", "Sedan"], size=n),
        "Home_Charging_Possible": np.random.choice(["Yes", "No"], size=n),
        "Charging_Stations_Near_Home": np.random.randint(0, 10, size=n),
        "Charging_Stations_Near_Work": np.random.randint(0, 10, size=n),
        TARGET: np.random.choice([0, 1], size=n),
    }
    train_df = pd.DataFrame(data)
    test_df = train_df.drop(columns=[TARGET]).copy()
    test_df["id"] = range(n, 2 * n)

    from features.grandmaster_features import build_grandmaster_features
    tr_feat, te_feat, features, te_cols = build_grandmaster_features(train_df, test_df)

    out_dir = tmp_path / "models"
    primary_sub = tmp_path / "submission.csv"

    # Test with prepared_folds
    cached_folds = prepare_seed_folds(
        train_feat=tr_feat,
        test_feat=te_feat,
        features=features,
        te_cols=te_cols,
        n_splits=2,
        seed=42,
    )

    auc, oof, test_p = train_single_model(
        model_type="lgbm",
        train_feat=tr_feat,
        test_feat=te_feat,
        features=features,
        te_cols=te_cols,
        n_splits=2,
        seed=42,
        has_gpu=False,
        output_dir=out_dir,
        primary_sub_path=primary_sub,
        prepared_folds=cached_folds,
    )

    assert len(oof) == n
    assert len(test_p) == n
    assert (out_dir / "lgbm_grandmaster" / "oof_preds.parquet").exists()
    assert (out_dir / "lgbm_grandmaster" / "submission.csv").exists()

