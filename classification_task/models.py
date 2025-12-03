"""
Proste klasy wrapper dla klasyfikatorów: drzewo decyzyjne i SVM.
Dokumentacja po polsku, parametry zrozumiałe i czytelne.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


@dataclass()
class DecisionTreeModel:
    """Drzewo decyzyjne z czytelnymi parametrami (kryterium, maks. głębokość)."""

    criterion: str = "gini"  # lub "entropy"
    max_depth: Optional[int] = None
    random_state: int = 42

    def build(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )


@dataclass()
class SVMModel:
    """SVM z parametrami kernela i regularizacji.

    kernel: "linear" | "rbf" | "poly" | "sigmoid"
    C: siła regularizacji (większe C = mniej regularizacji)
    gamma: wpływ pojedynczej próbki (dla rbf/sigmoid/poly)
    degree: stopień wielomianu (dla poly)
    """

    kernel: str = "rbf"
    C: float = 1.0
    gamma: str | float = "scale"
    degree: int = 3
    probability: bool = True
    random_state: int = 42

    def build(self) -> SVC:
        return SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            random_state=self.random_state,
        )

