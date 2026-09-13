import logging
import os
import shutil
import subprocess

logger = logging.getLogger(__name__)


class HardwareManager:
    """
    Hardware detection and accelerator compliance manager.
    Enforces avoidance of banned hardware (e.g. Pascal P100) and maps tasks
    to optimal accelerators (CPU vs Dual T4 vs L4).
    """

    BANNED_ACCELERATORS = ["Tesla P100", "Tesla K80"]
    PREFERRED_GPU = "Tesla T4"
    HIGH_VRAM_GPU = "NVIDIA L4"

    @classmethod
    def detect_gpu_info(cls) -> dict[str, str | int]:
        if not shutil.which("nvidia-smi"):
            return {"available": False, "device_name": "None", "count": 0}

        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()

            lines = [line.strip() for line in output.split("\n") if line.strip()]
            if not lines:
                return {"available": False, "device_name": "None", "count": 0}

            device_name = lines[0].split(",")[0].strip()
            return {
                "available": True,
                "device_name": device_name,
                "count": len(lines),
            }
        except Exception as e:
            logger.warning("Error querying nvidia-smi: %s", e)
            return {"available": False, "device_name": "None", "count": 0}

    @classmethod
    def validate_hardware_environment(cls) -> tuple[bool, str]:
        gpu_info = cls.detect_gpu_info()
        if not gpu_info["available"]:
            return True, "CPU-only environment detected. Proceeding with CPU optimizations."

        name = str(gpu_info["device_name"])
        for banned in cls.BANNED_ACCELERATORS:
            if banned.lower() in name.lower():
                return (
                    False,
                    f"CRITICAL HARDWARE ERROR: Accelerator '{name}' is banned. "
                    f"Pascal sm_60 architecture lacks modern PyTorch CUDA kernels on Kaggle base images.",
                )

        return True, f"Hardware verified: {gpu_info['count']}x {name}"

    @classmethod
    def get_optimal_accelerator(cls, task_type: str) -> str:
        """
        Routes task to optimal accelerator:
        - Tabular preprocessing / Polars / EDA -> CPU
        - GBDT training -> CPU (multi-core) or Dual T4
        - Large Neural Nets / Transformers -> Dual T4 or L4
        """
        if task_type in ["eda", "preprocessing", "feature_engineering"]:
            return "cpu"
        elif task_type in ["gbdt_training", "adversarial_validation"]:
            return "gpu_t4"
        elif task_type in ["deep_learning", "nlp_embeddings"]:
            return "gpu_l4"
        return "cpu"
