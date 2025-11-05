"""fuzzy_logic.py
Kontroler sterowania pojazdem oparty o logikę rozmytą.

Wejścia: odległości z 3 czujników (left, front, right).
Wyjścia: znormalizowana prędkość (mnożnik) oraz znormalizowany skręt [-1, 1]
który interpretujemy jako szybkość skrętu (prawo dodatnie, lewo ujemne).

Zawiera też funkcje do wizualizacji krzywych przynależności.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


def triangular(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Funkcja przynależności trójkątna."""
    y = np.zeros_like(x, dtype=float)
    # Rosnąca krawędź
    left = (a < x) & (x <= b)
    y[left] = (x[left] - a) / max(1e-9, (b - a))
    # Malejąca krawędź
    right = (b < x) & (x < c)
    y[right] = (c - x[right]) / max(1e-9, (c - b))
    # Wierzchołek
    y[x == b] = 1.0
    return y


def trapezoid(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Funkcja przynależności trapezowa."""
    y = np.zeros_like(x, dtype=float)
    # Rosnąca
    inc = (a < x) & (x <= b)
    y[inc] = (x[inc] - a) / max(1e-9, (b - a))
    # Płaskie maksimum
    mid = (b < x) & (x <= c)
    y[mid] = 1.0
    # Malejąca
    dec = (c < x) & (x < d)
    y[dec] = (d - x[dec]) / max(1e-9, (d - c))
    return y


@dataclass
class FuzzyController:
    max_range: float = 220.0

    def __post_init__(self):
        # Parametry zbiorów rozmytych dla odległości
        r = self.max_range
        self.near_params = (0.0, 0.0, 0.35 * r, 0.55 * r)  # trapez: a,b,c,d
        self.mid_params = (0.35 * r, 0.55 * r, 0.75 * r)   # triangle: a,b,c
        self.far_params = (0.60 * r, 0.85 * r, r, r)       # trapez: a,b,c,d

        # Wartości wyjściowe sterowania skrętem (normowane)
        self.turn_values = {
            "left_strong": -1.0,
            "left": -0.6,
            "straight": 0.0,
            "right": 0.6,
            "right_strong": 1.0,
        }

    # ====== Krzywe przynależności (do wizualizacji) ======
    def distance_membership_curves(self, n: int = 200) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        x = np.linspace(0.0, self.max_range, n)
        near = trapezoid(x, *self.near_params)
        mid = triangular(x, *self.mid_params)
        far = trapezoid(x, *self.far_params)
        return {"x": x, "near": near, "mid": mid, "far": far}

    # ====== Wartości przynależności dla konkretnych odległości ======
    def distance_memberships(self, d: float) -> Dict[str, float]:
        # Pojedynczy punkt
        x = np.array([d])
        near = float(trapezoid(x, *self.near_params)[0])
        mid = float(triangular(x, *self.mid_params)[0])
        far = float(trapezoid(x, *self.far_params)[0])
        return {"near": near, "mid": mid, "far": far}

    # ====== Reguły i defuzyfikacja ======
    def compute(self, d_left: float, d_front: float, d_right: float) -> Tuple[float, float, Dict[str, float]]:
        """Zwraca (turn_norm, speed_factor, debug_info).

        - turn_norm w [-1, 1]
        - speed_factor w [0.3, 1.0] (mnożnik prędkości bazowej)
        - debug_info zwraca m.in. stopnie przynależności dla front
        """
        ml = self.distance_memberships(d_left)
        mf = self.distance_memberships(d_front)
        mr = self.distance_memberships(d_right)

        # Proste reguły sterowania kierunkiem
        rules = {
            # Jeśli z przodu blisko, a po lewej daleko -> mocno w prawo
            "right_strong": min(mf["near"], ml["far"]),
            # Jeśli z przodu blisko, a po prawej daleko -> mocno w lewo
            "left_strong": min(mf["near"], mr["far"]),
            # Gdy po lewej blisko -> lekko w prawo
            "right": ml["near"],
            # Gdy po prawej blisko -> lekko w lewo
            "left": mr["near"],
            # Gdy z przodu daleko i boki nie są skrajne -> prosto
            "straight": min(mf["far"], max(ml["mid"], 0.4), max(mr["mid"], 0.4)),
        }

        # Agregacja i defuzyfikacja przez średnią ważoną
        num = 0.0
        den = 1e-6
        for name, degree in rules.items():
            num += degree * self.turn_values[name]
            den += degree
        turn_norm = num / den

        # Prędkość: zwalniaj gdy z przodu blisko, przyspiesz gdy daleko
        speed_factor = 0.3 * mf["near"] + 0.6 * mf["mid"] + 1.0 * mf["far"]
        speed_factor = float(np.clip(speed_factor, 0.3, 1.0))

        debug = {
            "front_near": mf["near"],
            "front_mid": mf["mid"],
            "front_far": mf["far"],
            "turn_norm": turn_norm,
            "rules": rules,
        }
        return turn_norm, speed_factor, debug