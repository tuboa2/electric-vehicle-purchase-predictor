import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    root_mean_squared_error,
    log_loss,
    f1_score,
    cohen_kappa_score,
)
from typing import Callable


class MetricRegistry:
    """
    Standardized, deterministic evaluation metric registry.
    Ensures consistent mathematical formulation and clipping across all agents.
    """

    @staticmethod
    def roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Handle 2D predictions (e.g. from predict_proba with shape (N, 2))
        if y_pred.ndim == 2:
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            elif y_pred.shape[1] == 1:
                y_pred = y_pred.ravel()

        # Handle NaNs and Infs in predictions
        if np.isnan(y_pred).any() or np.isneginf(y_pred).any() or np.isposinf(y_pred).any():
            y_pred = np.nan_to_num(y_pred, nan=0.5, posinf=1.0 - 1e-15, neginf=1e-15)

        # Clip probabilities to prevent numerical degeneration
        y_pred = np.clip(y_pred, 1e-15, 1.0 - 1e-15)

        # Format y_true (handles string representations like 'Yes'/'No' or booleans)
        if y_true.dtype.kind in {"U", "O", "S"}:
            y_true_str = np.char.strip(y_true.astype(str))
            y_true = (
                (y_true_str == "Yes")
                | (y_true_str == "1")
                | (y_true_str == "True")
                | (y_true_str == "true")
            ).astype(int)
        elif y_true.dtype == bool:
            y_true = y_true.astype(int)

        # Guard against single-class edge case (e.g. invalid fold or degenerate batch)
        unique_classes = np.unique(y_true)
        if len(unique_classes) < 2:
            return 0.5

        return float(roc_auc_score(y_true, y_pred))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(root_mean_squared_error(y_true, y_pred))

    @staticmethod
    def logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return float(log_loss(y_true, y_pred))

    @staticmethod
    def f1_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if y_pred.ndim > 1 or (np.issubdtype(y_pred.dtype, np.floating) and (y_pred >= 0).all() and (y_pred <= 1).all()):
            y_pred_labels = (y_pred >= 0.5).astype(int)
        else:
            y_pred_labels = y_pred
        return float(f1_score(y_true, y_pred_labels, average="macro"))

    @staticmethod
    def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_labels = np.round(y_pred).astype(int)
        return float(cohen_kappa_score(y_true, y_pred_labels, weights="quadratic"))

    @classmethod
    def get_metric(cls, name: str) -> tuple[Callable[[np.ndarray, np.ndarray], float], str]:
        """Returns tuple of (metric_function, direction)."""
        metrics = {
            "roc_auc": (cls.roc_auc, "maximize"),
            "rmse": (cls.rmse, "minimize"),
            "log_loss": (cls.logloss, "minimize"),
            "f1_macro": (cls.f1_macro, "maximize"),
            "qwk": (cls.qwk, "maximize"),
        }
        name_lower = name.lower()
        if name_lower not in metrics:
            raise KeyError(f"Metric '{name}' not found in registry. Available: {list(metrics.keys())}")
        return metrics[name_lower]
