"""Narzędzia do wczytywania utrwalonych ocen do modeli domenowych."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .data_models import Rating, UserProfile


@dataclass()
class CSVRatingRepository:
    """Wczytuje oceny z pliku CSV zgodnego ze schematem danych."""

    csv_path: Path
    encoding: str = "utf-8"

    def load_ratings(self) -> List[Rating]:
        """Zparsuj plik CSV do silnie typowanych obiektów ocen."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {self.csv_path}")

        ratings: List[Rating] = []
        with self.csv_path.open(mode="r", encoding=self.encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            required_columns = {"user_name", "item_title", "rating"}
            if not required_columns.issubset(reader.fieldnames or []):
                missing = required_columns - set(reader.fieldnames or [])
                raise ValueError(
                    f"Invalid ratings schema. Missing columns: {', '.join(sorted(missing))}"
                )

            for row in reader:
                ratings.append(
                    Rating(
                        user_name=row["user_name"].strip(),
                        item_title=row["item_title"].strip(),
                        rating=float(row["rating"]),
                    )
                )

        return ratings

    def load_user_profiles(self) -> Dict[str, UserProfile]:
        """Zagreguj wszystkie oceny per użytkownik dla wygody."""
        profiles: Dict[str, UserProfile] = {}
        for rating in self.load_ratings():
            profile = profiles.setdefault(rating.user_name, UserProfile(name=rating.user_name))
            profile.add_rating(rating.item_title, rating.rating)
        return profiles

