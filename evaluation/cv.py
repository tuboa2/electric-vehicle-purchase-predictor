import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    GroupKFold,
    StratifiedGroupKFold,
    TimeSeriesSplit,
)


class CrossValidationBuilder:
    """
    Validation scheme generator and invariant verification engine.
    Ensures leak-free fold assignment with zero cross-fold group contamination.
    """

    @staticmethod
    def generate_folds(
        df: pd.DataFrame,
        id_col: str,
        target_col: str,
        group_col: str | None = None,
        is_time_series: bool = False,
        is_classification: bool = True,
        n_splits: int = 5,
        seed: int = 42,
    ) -> pd.DataFrame:
        df_out = pd.DataFrame({id_col: df[id_col], "fold": -1})

        if is_time_series:
            # Strictly sequential split
            tscv = TimeSeriesSplit(n_splits=n_splits)
            for fold_idx, (_, val_idx) in enumerate(tscv.split(df)):
                df_out.iloc[val_idx, df_out.columns.get_loc("fold")] = fold_idx
            # Unassigned early samples set to fold 0
            df_out.loc[df_out["fold"] == -1, "fold"] = 0

        elif group_col and group_col in df.columns:
            if is_classification:
                sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
                splits = sgkf.split(df, df[target_col], groups=df[group_col])
            else:
                gkf = GroupKFold(n_splits=n_splits)
                splits = gkf.split(df, groups=df[group_col])

            for fold_idx, (_, val_idx) in enumerate(splits):
                df_out.iloc[val_idx, df_out.columns.get_loc("fold")] = fold_idx

        elif is_classification:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx, (_, val_idx) in enumerate(skf.split(df, df[target_col])):
                df_out.iloc[val_idx, df_out.columns.get_loc("fold")] = fold_idx

        else:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            for fold_idx, (_, val_idx) in enumerate(kf.split(df)):
                df_out.iloc[val_idx, df_out.columns.get_loc("fold")] = fold_idx

        CrossValidationBuilder.verify_fold_invariants(df_out, df, group_col, n_splits)
        return df_out

    @staticmethod
    def verify_fold_invariants(
        folds_df: pd.DataFrame,
        original_df: pd.DataFrame,
        group_col: str | None,
        n_splits: int,
    ) -> None:
        # Invariant 1: No unassigned rows
        if (folds_df["fold"] == -1).any():
            raise ValueError("Cross-validation invariant broken: Found unassigned rows (fold == -1)")

        # Invariant 2: Correct unique fold count
        unique_folds = set(folds_df["fold"].unique())
        expected_folds = set(range(n_splits))
        if unique_folds != expected_folds:
            raise ValueError(f"Fold counts mismatch: Expected {expected_folds}, got {unique_folds}")

        # Invariant 3: Zero group overlap across folds
        if group_col and group_col in original_df.columns:
            combined = pd.concat([folds_df[["fold"]], original_df[[group_col]]], axis=1)
            group_fold_counts = combined.groupby(group_col)["fold"].nunique()
            overlapping_groups = group_fold_counts[group_fold_counts > 1]
            if len(overlapping_groups) > 0:
                raise ValueError(
                    f"CRITICAL LEAKAGE: {len(overlapping_groups)} groups span multiple folds. "
                    f"Example violating groups: {list(overlapping_groups.index[:5])}"
                )
