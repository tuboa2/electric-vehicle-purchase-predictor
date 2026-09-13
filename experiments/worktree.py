import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class GitWorktreeManager:
    """
    Manages isolated Git worktrees for sandboxed experimental trials.
    Protects the primary repository from untested code mutations.
    """

    def __init__(self, worktree_base_dir: str | Path = "experiments/worktrees"):
        self.base_dir = Path(worktree_base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def is_git_repo(self) -> bool:
        return Path(".git").exists()

    def create_worktree(self, run_id: str, branch_name: str | None = None) -> Path:
        target_path = self.base_dir / f"run_{run_id}"
        branch = branch_name or f"experiment/{run_id}"

        if not self.is_git_repo():
            # Fallback for environments without git repo initialized
            target_path.mkdir(parents=True, exist_ok=True)
            logger.warning("No Git repo found. Initialized standard directory sandbox at %s", target_path)
            return target_path

        try:
            cmd = ["git", "worktree", "add", "-b", branch, str(target_path), "HEAD"]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info("Created Git worktree at %s on branch %s", target_path, branch)
            return target_path
        except subprocess.CalledProcessError as e:
            logger.error("Failed to create worktree: %s. Using directory sandbox fallback.", e.stderr)
            target_path.mkdir(parents=True, exist_ok=True)
            return target_path

    def remove_worktree(self, run_id: str, delete_branch: bool = True) -> bool:
        target_path = self.base_dir / f"run_{run_id}"
        branch = f"experiment/{run_id}"

        if not target_path.exists():
            return True

        if self.is_git_repo():
            try:
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(target_path)],
                    check=False,
                    capture_output=True,
                )
                if delete_branch:
                    subprocess.run(["git", "branch", "-D", branch], check=False, capture_output=True)
                logger.info("Successfully pruned worktree %s", target_path)
            except Exception as e:
                logger.warning("Error running git worktree remove: %s", e)

        # Ensure directory is gone
        if target_path.exists():
            shutil.rmtree(target_path, ignore_errors=True)

        return True
