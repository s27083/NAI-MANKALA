"""Pakiet najwyższego poziomu dla silnika rekomendacji."""

from .data_models import (
    Rating,
    UserProfile,
    MovieMetadata,
    Recommendation,
    RecommendationResult,
)
from .engine import RecommendationEngine

__all__ = [
    "Rating",
    "UserProfile",
    "MovieMetadata",
    "Recommendation",
    "RecommendationResult",
    "RecommendationEngine",
]

