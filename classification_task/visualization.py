"""
Proste wizualizacje: macierz korelacji i macierz pomyłek.
Zapisywane do katalogu output/.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


@dataclass()
class Visualizer:
    """Generator prostych wizualizacji zapisanych do plików PNG."""

    output_dir: Path

    def correlation_heatmap(self, X: pd.DataFrame, name: str) -> Path:
        corr = X.corr(numeric_only=True)
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, cmap="viridis", linewidths=0.5)
        plt.title(f"Macierz korelacji: {name}")
        out = self.output_dir / f"{name}_corr.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

    def confusion_heatmap(self, y_true: np.ndarray, y_pred: np.ndarray, name: str) -> Path:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
        plt.xlabel("Predykcja")
        plt.ylabel("Rzeczywistość")
        plt.title(f"Macierz pomyłek: {name}")
        out = self.output_dir / f"{name}_cm.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out)
        plt.close()
        return out

