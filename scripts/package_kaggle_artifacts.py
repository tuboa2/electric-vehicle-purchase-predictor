"""
Package all Kaggle run artifacts (submissions, OOFs, models, metrics) into a downloadable zip file.
"""

import os
from pathlib import Path
import zipfile
import sys

def package_artifacts():
    working_dir = Path("/kaggle/working")
    if not working_dir.exists():
        working_dir = Path(".")

    zip_filename = working_dir / "kaggle_run_artifacts.zip"
    print(f"[*] Packaging artifacts into: {zip_filename}")

    files_to_zip = []

    # 1. All submission CSVs in root
    for csv_file in working_dir.glob("*.csv"):
        files_to_zip.append((csv_file, csv_file.name))

    # 2. All model files (OOFs, metrics, test preds)
    models_dir = working_dir / "models"
    if models_dir.exists():
        for f in models_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(working_dir)
                files_to_zip.append((f, str(rel_path)))

    # 3. Ensemble artifacts
    ensemble_dir = working_dir / "ensemble_grandmaster"
    if ensemble_dir.exists():
        for f in ensemble_dir.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(working_dir)
                files_to_zip.append((f, str(rel_path)))

    print(f"[+] Found {len(files_to_zip)} artifact files to package.")
    with zipfile.ZipFile(zip_filename, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_zip:
            print(f"  -> Adding: {arc} ({src.stat().st_size / 1024:.1f} KB)")
            zf.write(src, arcname=arc)

    zip_size_mb = zip_filename.stat().st_size / (1024 * 1024)
    print(f"\n=================================================================")
    print(f"[+] Zip package ready: {zip_filename} ({zip_size_mb:.2f} MB)")
    print(f"=================================================================")

    # Display downloadable link in Kaggle notebook
    try:
        from IPython.display import FileLink, display
        display(FileLink(str(zip_filename.name)))
        print("[+] Download link rendered above!")
    except Exception:
        pass

if __name__ == "__main__":
    package_artifacts()
