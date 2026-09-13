#!/usr/bin/env python3
"""
KAMAS Competition Initializer
Initializes directories, blackboard state, budget ledger, and SQLite project store.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from orchestration.blackboard import FilesystemBlackboard
from orchestration.budget import BudgetManager
from memory.sqlite_store import SQLiteExperimentStore
from kaggle.client import KaggleClient
from schemas.state import BlackboardState, SystemPhase


def init_competition(competition_id: str, download_data: bool = False) -> int:
    print("=" * 70)
    print(f"INITIALIZING KAMAS FOR COMPETITION: {competition_id}")
    print("=" * 70)

    # 1. Directory Structure
    dirs = [
        Path(f"data/raw/{competition_id}"),
        Path(f"data/processed/{competition_id}"),
        Path("experiments/blackboard/active_runs"),
        Path("experiments/blackboard/vetos"),
        Path("experiments/artifacts"),
        Path("experiments/submissions"),
        Path("experiments/worktrees"),
        Path("memory"),
        Path("knowledge"),
        Path("observability/logs"),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"[+] Directory verified: {d}")

    # 2. Initialize Blackboard
    bb = FilesystemBlackboard()
    current_state = bb.read_state()
    current_state.competition_id = competition_id
    current_state.phase = SystemPhase.INITIALIZED
    current_state.cleared_gates = []
    current_state.current_iteration = 0
    bb.write_state(current_state, updated_by="init_competition")
    print(f"[+] Blackboard state initialized for {competition_id}")

    # 3. Initialize Budget Manager
    budget_mgr = BudgetManager()
    print(f"[+] Budget Manager online. Remaining GPU hours: {budget_mgr.get_remaining_gpu_hours():.1f}h")

    # 4. Initialize SQLite Project Memory
    store = SQLiteExperimentStore("memory/experiments.db")
    print("[+] SQLite Project Memory online at memory/experiments.db")

    # 5. Optional Data Download
    if download_data:
        print("[*] Initiating Kaggle data download...")
        client = KaggleClient()
        success = client.download_competition_data(competition_id, f"data/raw/{competition_id}")
        if success:
            print("[+] Competition data downloaded and unpacked successfully.")
        else:
            print("[-] Data download failed. Place raw files manually in data/raw/{competition_id}")

    print("=" * 70)
    print(f"[+] COMPETITION {competition_id} READY FOR EXECUTION.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--competition-id", required=True, help="Kaggle competition slug")
    parser.add_argument("--download-data", action="store_true", help="Attempt download via Kaggle CLI")
    args = parser.parse_args()
    sys.exit(init_competition(args.competition_id, args.download_data))
