import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class KaggleClient:
    """
    Kaggle CLI & API wrapper for competition data download, kernel push,
    status polling, and CSV submission tracking.
    """

    def __init__(self):
        self.kaggle_bin = shutil.which("kaggle") or ".venv/bin/kaggle"

    def check_auth(self) -> tuple[bool, str]:
        """Verifies presence of ~/.kaggle/kaggle.json and valid authentication."""
        kaggle_json = Path(os.path.expanduser("~/.kaggle/kaggle.json"))
        if not kaggle_json.exists():
            return False, f"Kaggle token not found at {kaggle_json}"

        try:
            res = subprocess.run(
                [self.kaggle_bin, "competitions", "list", "--page", "1"],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                return True, "Kaggle authentication successful"
            else:
                return False, f"Kaggle CLI authentication failed: {res.stderr.strip()}"
        except Exception as e:
            return False, f"Failed to execute kaggle CLI: {e}"

    def download_competition_data(self, competition_id: str, output_dir: str | Path) -> bool:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                [self.kaggle_bin, "competitions", "download", "-c", competition_id, "-p", str(out_p)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                # Check if a zip file needs unzipping
                for zip_file in out_p.glob("*.zip"):
                    shutil.unpack_archive(zip_file, out_p)
                logger.info("Successfully downloaded data for %s to %s", competition_id, out_p)
                return True
            else:
                logger.error("Download failed for %s: %s", competition_id, res.stderr)
                return False
        except Exception as e:
            logger.error("Exception during competition data download: %s", e)
            return False

    def push_kernel(self, bundle_dir: str | Path) -> tuple[bool, str]:
        bundle_p = Path(bundle_dir)
        try:
            res = subprocess.run(
                [self.kaggle_bin, "kernels", "push", "-p", str(bundle_p)],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                logger.info("Kernel pushed successfully from %s", bundle_p)
                return True, res.stdout.strip()
            else:
                logger.error("Kernel push failed: %s", res.stderr)
                return False, res.stderr.strip()
        except Exception as e:
            return False, str(e)

    def get_kernel_status(self, kernel_slug: str) -> str:
        try:
            res = subprocess.run(
                [self.kaggle_bin, "kernels", "status", kernel_slug],
                capture_output=True,
                text=True,
                check=False,
            )
            return res.stdout.strip() if res.returncode == 0 else "ERROR"
        except Exception:
            return "ERROR"

    def submit_csv(self, competition_id: str, file_path: str | Path, message: str) -> tuple[bool, str]:
        f_p = Path(file_path)
        if not f_p.exists():
            return False, f"Submission file {f_p} does not exist"

        try:
            res = subprocess.run(
                [self.kaggle_bin, "competitions", "submit", "-c", competition_id, "-f", str(f_p), "-m", message],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                logger.info("Submission uploaded: %s", res.stdout.strip())
                return True, res.stdout.strip()
            else:
                logger.error("Submission failed: %s", res.stderr)
                return False, res.stderr.strip()
        except Exception as e:
            return False, str(e)

    def get_submissions(self, competition_id: str) -> list[dict[str, Any]]:
        try:
            res = subprocess.run(
                [self.kaggle_bin, "competitions", "submissions", "-c", competition_id],
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode != 0:
                return []
            lines = [line for line in res.stdout.split("\n") if line.strip()]
            # Parse header and lines
            submissions = []
            if len(lines) > 1:
                for line in lines[1:]:
                    parts = line.split()
                    if parts:
                        submissions.append({"raw": line, "filename": parts[0]})
            return submissions
        except Exception:
            return []
