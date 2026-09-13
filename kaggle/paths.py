"""
Kaggle and Local Environment Path Resolver.
Provides automatic detection of Kaggle notebook environments (/kaggle/input and /kaggle/working)
versus local repository execution paths.
"""

import os
from pathlib import Path

DEFAULT_COMPETITION_ID = "playground-series-s6e9"


def is_kaggle_environment() -> bool:
    """Detects whether code is executing inside a Kaggle Notebook container."""
    return Path("/kaggle").exists() and (Path("/kaggle/input").exists() or Path("/kaggle/working").exists())


def resolve_data_dir(competition_id: str = DEFAULT_COMPETITION_ID) -> Path:
    """
    Finds the directory containing train and test files across Kaggle and local environments.
    Checks candidate paths in order:
      1. /kaggle/input/{competition_id}
      2. data/processed/{competition_id}
      3. data/raw/{competition_id}
      4. ../data/processed/{competition_id}
      5. ../data/raw/{competition_id}
    """
    candidates = [
        Path(f"/kaggle/input/{competition_id}"),
        Path(f"data/processed/{competition_id}"),
        Path(f"data/raw/{competition_id}"),
        Path(f"../data/processed/{competition_id}"),
        Path(f"../data/raw/{competition_id}"),
        Path("data/processed"),
        Path("data/raw"),
    ]

    for cand in candidates:
        if cand.exists() and (
            (cand / "train.parquet").exists()
            or (cand / "train.csv").exists()
        ):
            return cand.resolve()

    # Fallback default
    return Path(f"data/processed/{competition_id}").resolve()


def resolve_output_dir(default_subfolder: str = "artifacts") -> Path:
    """
    Resolves writable output directory for predictions and submissions.
    Uses /kaggle/working in Kaggle environments or experiments/ locally.
    """
    if is_kaggle_environment():
        out = Path("/kaggle/working")
        out.mkdir(parents=True, exist_ok=True)
        return out

    out = Path(f"experiments/{default_subfolder}")
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()
