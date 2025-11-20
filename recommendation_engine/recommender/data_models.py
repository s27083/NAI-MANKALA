"""Podstawowe modele domenowe używane przez silnik rekomendacji."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Rating:
    """Reprezentuje pojedynczą ocenę wystawioną przez użytkownika dla tytułu."""

    user_name: str
    item_title: str
    rating: float


@dataclass()
class UserProfile:
    """Agreguje wszystkie dostępne oceny dla danego użytkownika."""

    name: str
    ratings: Dict[str, float] = field(default_factory=dict)

    def add_rating(self, title: str, score: float) -> None:
        """Zarejestruj lub nadpisz pojedynczą ocenę użytkownika."""
        self.ratings[title] = score


@dataclass()
class MovieMetadata:
    """Przechowuje opisowe metadane o filmie lub serialu."""

    title: str
    year: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    plot: Optional[str] = None
    poster_url: Optional[str] = None
    imdb_rating: Optional[float] = None
    runtime_minutes: Optional[int] = None
    source: Optional[str] = None


@dataclass()
class Recommendation:
    """Opisuje pojedynczą rekomendację z wynikiem i metadanymi."""

    title: str
    score: float
    metadata: Optional[MovieMetadata] = None


@dataclass()
class RecommendationResult:
    """Kontener na pozytywne i negatywne propozycje dla użytkownika."""

    user: str
    recommended: List[Recommendation] = field(default_factory=list)
    anti_recommended: List[Recommendation] = field(default_factory=list)

