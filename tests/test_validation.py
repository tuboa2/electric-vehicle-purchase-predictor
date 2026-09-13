import numpy as np
import pandas as pd
import pytest
from evaluation.cv import CrossValidationBuilder
from evaluation.adversarial import AdversarialValidator


def test_stratified_kfold_invariants():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "id": np.arange(n),
        "target": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        "feature": np.random.randn(n),
    })

    folds_df = CrossValidationBuilder.generate_folds(
        df=df,
        id_col="id",
        target_col="target",
        n_splits=5,
        is_classification=True,
    )

    assert len(folds_df) == n
    assert set(folds_df["fold"].unique()) == {0, 1, 2, 3, 4}
    assert (folds_df["fold"] >= 0).all()


def test_group_kfold_zero_overlap_invariant():
    np.random.seed(42)
    n = 300
    groups = [f"group_{i % 30}" for i in range(n)]
    df = pd.DataFrame({
        "id": np.arange(n),
        "target": np.random.choice([0, 1], n),
        "group_id": groups,
    })

    folds_df = CrossValidationBuilder.generate_folds(
        df=df,
        id_col="id",
        target_col="target",
        group_col="group_id",
        n_splits=5,
        is_classification=True,
    )

    # Verify that NO group spans across more than 1 fold
    combined = pd.concat([folds_df[["fold"]], df[["group_id"]]], axis=1)
    group_fold_counts = combined.groupby("group_id")["fold"].nunique()
    assert (group_fold_counts == 1).all(), "Detected group leaking across multiple folds!"


def test_adversarial_validation_detection():
    np.random.seed(42)
    n_train = 300
    n_test = 150

    # Case 1: Identical distribution -> AUC ~ 0.50
    train_df = pd.DataFrame({
        "id": np.arange(n_train),
        "f1": np.random.normal(0, 1, n_train),
        "f2": np.random.uniform(0, 10, n_train),
    })
    test_df = pd.DataFrame({
        "id": np.arange(n_train, n_train + n_test),
        "f1": np.random.normal(0, 1, n_test),
        "f2": np.random.uniform(0, 10, n_test),
    })

    res = AdversarialValidator.evaluate(train_df, test_df, drop_cols=["id"])
    assert res["adversarial_auc"] < 0.65
    assert res["severity"] in ["NEGLIGIBLE", "MILD"]

    # Case 2: Extreme covariate shift -> AUC > 0.85
    test_shifted = pd.DataFrame({
        "id": np.arange(n_train, n_train + n_test),
        "f1": np.random.normal(5.0, 1, n_test), # Massive shift
        "f2": np.random.uniform(20, 30, n_test),
    })
    res_shifted = AdversarialValidator.evaluate(train_df, test_shifted, drop_cols=["id"])
    assert res_shifted["adversarial_auc"] > 0.85
    assert res_shifted["severity"] == "CRITICAL"
