"""Narzędzia przetwarzania wstępnego do przekształcania ocen w macierz użytkownik–pozycja."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .data_models import Rating, UserProfile


@dataclass()
class UserItemMatrix:
    """Numeryczna reprezentacja preferencji użytkowników."""

    matrix: np.ndarray
    user_index: Dict[str, int]
    item_index: Dict[str, int]
    users: List[str]
    items: List[str]

    def as_dataframe(self) -> pd.DataFrame:
        """Zwróć macierz jako pandas DataFrame z odpowiednimi etykietami."""
        df = pd.DataFrame(self.matrix, index=self.users, columns=self.items)
        df.index.name = "user_name"
        df.columns.name = "item_title"
        return df

    def get_user_vector(self, user_name: str) -> np.ndarray:
        """Pobierz wektor ocen dla wskazanego użytkownika."""
        try:
            return self.matrix[self.user_index[user_name]]
        except KeyError as error:  # pragma: no cover - defensive branch
            raise ValueError(f"Unknown user: {user_name}") from error


def build_matrix_from_profiles(profiles: Iterable[UserProfile]) -> UserItemMatrix:
    """Utwórz gęstą macierz na podstawie przekazanych profili użytkowników."""
    profile_list: List[UserProfile] = list(profiles)
    if not profile_list:
        raise ValueError("No user profiles received.")

    all_items: Sequence[str] = sorted(
        {title for profile in profile_list for title in profile.ratings.keys()}
    )
    users: List[str] = [profile.name for profile in profile_list]
    items: List[str] = list(all_items)
    user_index: Dict[str, int] = {name: idx for idx, name in enumerate(users)}
    item_index: Dict[str, int] = {title: idx for idx, title in enumerate(items)}

    matrix = np.full((len(profile_list), len(items)), np.nan, dtype=float)
    for profile in profile_list:
        for title, score in profile.ratings.items():
            row = user_index[profile.name]
            col = item_index[title]
            matrix[row, col] = score

    return UserItemMatrix(
        matrix=matrix,
        user_index=user_index,
        item_index=item_index,
        users=users,
        items=items,
    )


def build_matrix_from_ratings(ratings: Iterable[Rating]) -> UserItemMatrix:
    """Wygodny wrapper grupujący oceny per użytkownik przed budową macierzy."""
    profiles: Dict[str, UserProfile] = {}
    for rating in ratings:
        profile = profiles.setdefault(rating.user_name, UserProfile(name=rating.user_name))
        profile.add_rating(rating.item_title, rating.rating)
    return build_matrix_from_profiles(profiles.values())

