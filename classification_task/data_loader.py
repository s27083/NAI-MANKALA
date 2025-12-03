"""
Problem: Klasyfikacja danych (serce i drugi zbiór) prostymi klasyfikatorami.
Autorzy: Adrian Kemski s27444, Kamil Bogdański s27083
Użycie: patrz main.py
Referencje:
 - Analysis of Depth of Entropy and GINI Index Based Decision Trees for Predicting Diabetes
   https://jns.edu.al/wp-content/uploads/2024/01/M.UlqinakuA.Ktona-FINAL.pdf
 - Zbiór 1 (serce): Heart Failure Clinical Data (Kaggle)
   https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data
 - Zbiór 2 (pierś): Breast Cancer Wisconsin (Diagnostic) (UCI)
   https://archive.ics.uci.edu/dataset/33/breast+cancer+wisconsin+diagnostic
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer


@dataclass()
class HeartFailureLoader:
    """Loader dla zbioru Heart Failure Clinical Data z Kaggle.

    Oczekuje pliku CSV zawierającego kolumnę celu `DEATH_EVENT` oraz cechy numeryczne.
    Minimalny preprocessing: usuwanie braków, konwersja typów, separacja X/y.
    """

    csv_path: Path

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Nie znaleziono pliku danych: {self.csv_path}. Pobierz z Kaggle i umieść w tej ścieżce."
            )
        df = pd.read_csv(self.csv_path)
        if "DEATH_EVENT" not in df.columns:
            raise ValueError("Brak kolumny celu 'DEATH_EVENT' w dostarczonym CSV.")
        y = df["DEATH_EVENT"].astype(int)
        X = df.drop(columns=["DEATH_EVENT"]).copy()
        # Prosty cleanup: wypełnianie braków medianą
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(X.median(numeric_only=True))
        return X, y


@dataclass()
class BreastCancerLoader:
    """Loader dla zbioru Breast Cancer (sklearn), z linkiem do UCI w nagłówku.

    Zwraca ramkę danych i serię celu zgodną z pandas.
    """

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=list(data.feature_names))
        y = pd.Series(data.target, name="target")
        return X, y

