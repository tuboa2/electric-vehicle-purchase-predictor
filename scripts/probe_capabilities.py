#!/usr/bin/env python3
"""
KAMAS System Capability & Environment Probe
Verifies Python dependencies, Kaggle CLI auth, accelerator safety, and filesystem write access.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
from kaggle.hardware import HardwareManager
from kaggle.client import KaggleClient


def probe_system() -> int:
    print("=" * 70)
    print("KAMAS SYSTEM CAPABILITY & HEALTH PROBE")
    print("=" * 70)

    # 1. Python version check
    py_ver = sys.version_info
    print(f"[*] Python Version: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")
    if py_ver < (3, 11):
        print("[-] FAILED: Python 3.11+ is required.")
        return 1

    # 2. Required dependencies
    required_packages = [
        "polars",
        "pandas",
        "numpy",
        "scipy",
        "sklearn",
        "lightgbm",
        "pydantic",
        "yaml",
        "pytest",
    ]
    print("[*] Checking Python Dependencies:")
    missing = []
    for pkg in required_packages:
        try:
            importlib.import_module(pkg)
            print(f"    [+] {pkg}: OK")
        except ImportError:
            print(f"    [-] {pkg}: MISSING")
            missing.append(pkg)

    if missing:
        print(f"[-] FAILED: Missing packages: {missing}")
        return 1

    # 3. Hardware & Accelerator Safety
    print("[*] Checking Hardware Environment:")
    hw_valid, hw_msg = HardwareManager.validate_hardware_environment()
    if hw_valid:
        print(f"    [+] Hardware Status: {hw_msg}")
    else:
        print(f"    [-] Hardware Status: {hw_msg}")
        return 1

    # 4. Kaggle CLI Authentication
    print("[*] Checking Kaggle API Authentication:")
    client = KaggleClient()
    auth_ok, auth_msg = client.check_auth()
    if auth_ok:
        print(f"    [+] Kaggle Auth: {auth_msg}")
    else:
        print(f"    [!] Kaggle Auth Notice: {auth_msg}")
        print("        (Mock submission modes will be used if API key is not configured)")

    # 5. Filesystem Write Check
    print("[*] Checking Filesystem Blackboard Permissions:")
    test_path = Path("experiments/blackboard/.write_test")
    try:
        test_path.parent.mkdir(parents=True, exist_ok=True)
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink()
        print("    [+] Blackboard Write Access: OK")
    except Exception as e:
        print(f"    [-] Blackboard Write Access FAILED: {e}")
        return 1

    print("=" * 70)
    print("[+] ALL SYSTEM INTEGRITY CHECKS PASSED. SYSTEM READY FOR ORCHESTRATION.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(probe_system())
