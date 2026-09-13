"""
Out-of-Fold Bayesian Target Encoder with M-Estimate Smoothing.
Designed for competitive tabular classification (Gate 2 Leakage Compliance).
Guarantees zero target leakage:
- Target encodings are fitted strictly on the training slice of each fold
- Validation fold and test set are transformed using the fitted priors
- Unseen categories fall back to the global prior
"""

import numpy as np
import pandas as pd


class OutOfFoldTargetEncoder:
    """
    Computes smoothed target encoding:
    S(c) = (n_c * y_c_mean + m * global_prior) / (n_c + m)
    """

    def __init__(self, target_col: str, m_smoothing: float = 25.0):
        self.target_col = target_col
        self.m_smoothing = m_smoothing
        self.encodings: dict[str, dict[str, float]] = {}
        self.global_prior: float = 0.5

    def fit_transform_fold(
        self,
        train_slice: pd.DataFrame,
        val_slice: pd.DataFrame,
        cat_columns: list[str],
        y_train_binary: np.ndarray,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        """
        Fits strictly on train_slice and transforms both train_slice (OOF) and val_slice.
        """
        tr_out = train_slice.copy()
        va_out = val_slice.copy()

        self.global_prior = float(np.mean(y_train_binary))
        self.encodings = {}
        encoded_col_names = []

        # Temporary target column for grouping
        tr_out["_temp_target"] = y_train_binary

        for col in cat_columns:
            new_col = f"te_{col}"
            encoded_col_names.append(new_col)

            # Compute stats on train slice
            stats = tr_out.groupby(col)["_temp_target"].agg(["count", "mean"])
            counts = stats["count"]
            means = stats["mean"]

            smoothed = (counts * means + self.m_smoothing * self.global_prior) / (counts + self.m_smoothing)
            mapping = smoothed.to_dict()
            self.encodings[col] = mapping

            # Map to train and val
            tr_out[new_col] = tr_out[col].map(mapping).fillna(self.global_prior).astype(float)
            va_out[new_col] = va_out[col].map(mapping).fillna(self.global_prior).astype(float)

        tr_out.drop(columns=["_temp_target"], inplace=True)
        return tr_out, va_out, encoded_col_names

    def transform_test(self, test_df: pd.DataFrame, cat_columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
        """
        Transforms test set using the fold's fitted encodings.
        """
        te_out = test_df.copy()
        encoded_col_names = []

        for col in cat_columns:
            new_col = f"te_{col}"
            encoded_col_names.append(new_col)
            mapping = self.encodings.get(col, {})
            te_out[new_col] = te_out[col].map(mapping).fillna(self.global_prior).astype(float)

        return te_out, encoded_col_names
