"""
Kaggle and Local Environment Path Resolver.
Provides automatic detection of Kaggle notebook environments (/kaggle/input and /kaggle/working)
versus local repository execution paths with deep recursive discovery and automated fallback.
"""

import os
from pathlib import Path
import shutil
import subprocess

DEFAULT_COMPETITION_ID = "playground-series-s6e9"


def is_kaggle_environment() -> bool:
    """Detects whether code is executing inside a Kaggle Notebook container."""
    return Path("/kaggle").exists() and (Path("/kaggle/input").exists() or Path("/kaggle/working").exists())


def resolve_data_dir(explicit_dir: str | Path | None = None, competition_id: str = DEFAULT_COMPETITION_ID) -> Path:
    """
    Intelligently discovers the directory containing competition data across Kaggle
    and local execution environments.

    Search progression:
      1. Explicit path if provided (checking direct and recursive subdirectories)
      2. Direct path /kaggle/input/{competition_id}
      3. Deep scan across all /kaggle/input/ subdirectories for train.csv or train.parquet
      4. Local paths: data/processed/{competition_id}, data/raw/{competition_id}, etc.
      5. Automatic Kaggle CLI download fallback into /kaggle/working/data if dataset is not attached
    """
    # 1. Check explicit directory if provided
    if explicit_dir is not None:
        exp_path = Path(explicit_dir).resolve()
        if exp_path.exists():
            if (exp_path / "train.parquet").exists() or (exp_path / "train.csv").exists():
                return exp_path
            # Check recursive subdirectories of explicit_dir
            for train_file in exp_path.rglob("train.*"):
                if train_file.name in ("train.csv", "train.parquet"):
                    print(f"[*] Located training data in subdirectory: {train_file.parent}")
                    return train_file.parent.resolve()

    # 2. Check standard candidate directories
    candidates = [
        Path(f"/kaggle/input/{competition_id}"),
        Path(f"/kaggle/input/{competition_id.replace('-', '_')}"),
        Path(f"/kaggle/input/{competition_id.replace('_', '-')}"),
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

    # 3. Deep scan across all mounted /kaggle/input folders
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        print(f"[*] Searching mounted /kaggle/input directories...")
        try:
            mounted = [d.name for d in kaggle_input.iterdir()]
            print(f"[*] Currently mounted in /kaggle/input: {mounted}")
        except Exception:
            mounted = []

        for train_file in kaggle_input.rglob("train.*"):
            if train_file.name in ("train.csv", "train.parquet"):
                print(f"[+] Located training data at: {train_file.parent}")
                return train_file.parent.resolve()

    # 4. Attempt automatic download via Kaggle CLI
    download_dir = Path(f"/kaggle/working/data/{competition_id}") if is_kaggle_environment() else Path(f"data/raw/{competition_id}")
    download_dir.mkdir(parents=True, exist_ok=True)
    kaggle_bin = shutil.which("kaggle") or ".venv/bin/kaggle"

    print(f"[*] Competition data not found in mounted inputs. Attempting automated download to {download_dir}...")
    try:
        res = subprocess.run(
            [kaggle_bin, "competitions", "download", "-c", competition_id, "-p", str(download_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            for zip_f in download_dir.glob("*.zip"):
                shutil.unpack_archive(zip_f, download_dir)
            if (download_dir / "train.csv").exists() or (download_dir / "train.parquet").exists():
                print(f"[+] Successfully downloaded and extracted {competition_id} data.")
                return download_dir.resolve()
        else:
            if res.stderr.strip():
                print(f"[-] Kaggle CLI download note: {res.stderr.strip()}")
    except Exception as e:
        print(f"[-] Kaggle CLI download exception: {e}")

    # 5. Diagnostic Error Message with exact Kaggle UI instructions
    error_msg = (
        f"\n{'='*70}\n"
        f"[!] CRITICAL: Could not locate competition dataset for '{competition_id}'.\n"
        f"{'='*70}\n"
        f"If running inside a Kaggle Notebook:\n"
        f"  1. Look at the right-hand sidebar panel under 'Input'.\n"
        f"  2. Click the '+ Add Input' button (or '+ Add Data' in older UI).\n"
        f"  3. Select the 'Competition Data' tab.\n"
        f"  4. Search for: {competition_id} ('Playground Series - Season 6, Episode 9').\n"
        f"  5. Click 'Add' next to the competition to mount the dataset.\n"
        f"{'='*70}\n"
    )
    raise FileNotFoundError(error_msg)


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
