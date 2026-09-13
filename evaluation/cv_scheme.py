"""
Validation Architect Cross-Validation Scheme for playground-series-s6e9.
Generates deterministic, verified 5-fold StratifiedKFold assignments.
"""

from pathlib import Path
import polars as pl
from evaluation.cv import CrossValidationBuilder

COMPETITION_ID = "playground-series-s6e9"
N_SPLITS = 5
RANDOM_SEED = 42


def get_folds(
    train_parquet_path: str | Path = f"data/processed/{COMPETITION_ID}/train.parquet",
    folds_parquet_path: str | Path = f"data/processed/{COMPETITION_ID}/folds.parquet",
) -> pl.DataFrame:
    """Returns certified fold assignments DataFrame with columns ['id', 'fold']."""
    f_path = Path(folds_parquet_path)
    if f_path.exists():
        return pl.read_parquet(f_path)

    train_df = pl.read_parquet(train_parquet_path).to_pandas()
    folds_df = CrossValidationBuilder.generate_folds(
        df=train_df,
        id_col="id",
        target_col="Will_Buy_EV",
        group_col=None,
        is_time_series=False,
        is_classification=True,
        n_splits=N_SPLITS,
        seed=RANDOM_SEED,
    )
    res = pl.from_pandas(folds_df)
    f_path.parent.mkdir(parents=True, exist_ok=True)
    res.write_parquet(f_path, compression="zstd")
    return res


if __name__ == "__main__":
    df_folds = get_folds()
    print(f"Loaded certified folds: {df_folds.shape}")
    print(df_folds.group_by("fold").len())
