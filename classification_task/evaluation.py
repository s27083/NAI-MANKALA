"""
Moduł metryk: dokładność, precyzja, recall, F1, ROC-AUC i macierz pomyłek.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


@dataclass()
class Evaluator:
    """Oblicz podstawowe metryki jakości klasyfikacji."""

    average: str = "binary"  # dla wieloklasowych użyj "macro"

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average=self.average, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average=self.average, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average=self.average, zero_division=0)),
        }
        try:
            if y_proba is not None:
                proba = y_proba if y_proba.ndim == 1 else y_proba[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba))
        except Exception:
            pass
        cm = confusion_matrix(y_true, y_pred)
        metrics["confusion_tp"] = float(cm[1, 1]) if cm.shape == (2, 2) else np.nan
        metrics["confusion_tn"] = float(cm[0, 0]) if cm.shape == (2, 2) else np.nan
        metrics["confusion_fp"] = float(cm[0, 1]) if cm.shape == (2, 2) else np.nan
        metrics["confusion_fn"] = float(cm[1, 0]) if cm.shape == (2, 2) else np.nan
        return metrics

