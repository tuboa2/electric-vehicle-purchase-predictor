from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR_CANDIDATES = [
    Path("/kaggle/input/competitions/playground-series-s6e9"),
    Path("/kaggle/input/playground-series-s6e9"),
    PROJECT_ROOT / "data" / "processed" / "playground-series-s6e9",
    PROJECT_ROOT / "data" / "raw" / "playground-series-s6e9",
    PROJECT_ROOT / "data" / "raw",
]

MODELS_DIR = PROJECT_ROOT / "models"
TARGET_COL = "Will_Buy_EV"

# Optimal blending weights discovered via Nelder-Mead
TRI_BLEND_WEIGHTS = {
    "backbone": 1.1173,
    "pseudo": 0.2013,
    "nn": 0.0420,
    "boundary": 0.0404
}
