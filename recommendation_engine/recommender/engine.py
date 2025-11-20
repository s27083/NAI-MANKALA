"""Silnik rekomendacji oparty o klastrowanie użytkowników."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .api_client import MovieMetadataClient
from .data_models import (
    MovieMetadata,
    Recommendation,
    RecommendationResult,
    UserProfile,
)
from .data_loader import CSVRatingRepository
from .preprocessing import UserItemMatrix, build_matrix_from_profiles


@dataclass()
class RecommendationEngine:
    """Wysokopoziomowy interfejs udostępniający funkcje rekomendacyjne."""

    metadata_client: MovieMetadataClient
    cluster_candidates: Sequence[int] = (3, 4, 5)
    random_state: int = 42
    min_votes: int = 2

    profiles: Dict[str, UserProfile] = field(init=False, default_factory=dict)
    matrix: UserItemMatrix = field(init=False)
    pipeline: Pipeline = field(init=False)
    processed_matrix: np.ndarray = field(init=False)
    cluster_labels: np.ndarray = field(init=False)
    global_item_mean: np.ndarray = field(init=False)
    global_item_votes: np.ndarray = field(init=False)

    def fit_from_csv(self, csv_path: str) -> None:
        """Wczytaj oceny z pliku CSV i wytrenuj model."""
        repository = CSVRatingRepository(csv_path=Path(csv_path))
        ratings = repository.load_ratings()
        profiles = repository.load_user_profiles()
        self.fit(profiles.values())

    def fit(self, profiles: Iterable[UserProfile]) -> None:
        """Wytrenuj silnik na podstawie przekazanych profili użytkowników."""
        self.profiles = {profile.name: profile for profile in profiles}
        self.matrix = build_matrix_from_profiles(self.profiles.values())
        self._configure_pipeline()
        self.processed_matrix = self.pipeline.fit_transform(self.matrix.matrix)
        self.processed_matrix = np.nan_to_num(self.processed_matrix, nan=0.0)

        n_clusters = self._determine_cluster_count(self.processed_matrix)
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        self.cluster_labels = kmeans.fit_predict(self.processed_matrix)

        self.global_item_mean = np.nanmean(self.matrix.matrix, axis=0)
        self.global_item_mean = np.nan_to_num(self.global_item_mean, nan=0.0)
        self.global_item_votes = np.sum(~np.isnan(self.matrix.matrix), axis=0)

    def recommend(
        self,
        user_name: str,
        top_n: int = 5,
        anti_n: int = 5,
    ) -> RecommendationResult:
        """Wygeneruj pozytywne i negatywne rekomendacje dla wskazanego użytkownika."""
        if not hasattr(self, "cluster_labels"):
            raise RuntimeError("Engine not fitted. Call fit() first.")

        if user_name not in self.matrix.user_index:
            raise ValueError(f"Unknown user: {user_name}")

        user_idx = self.matrix.user_index[user_name]
        cluster_id = self.cluster_labels[user_idx]

        peer_indices = self._get_peer_indices(cluster_id, exclude_index=user_idx)

        scores, vote_counts = self._compute_weighted_scores(peer_indices, user_idx)
        seen_mask = ~np.isnan(self.matrix.matrix[user_idx])

        recommended_titles = self._select_top_items(
            scores,
            vote_counts,
            seen_mask,
            top_n,
            reverse=True,
        )
        anti_titles = self._select_top_items(
            scores,
            vote_counts,
            seen_mask,
            anti_n,
            reverse=False,
        )

        return RecommendationResult(
            user=user_name,
            recommended=self._build_recommendations(recommended_titles),
            anti_recommended=self._build_recommendations(anti_titles),
        )

    def _configure_pipeline(self) -> None:
        """Przygotuj potok przetwarzania wstępnego."""
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler()),
            ]
        )

    def _determine_cluster_count(self, features: np.ndarray) -> int:
        """Wybierz optymalną liczbę klastrów na podstawie wskaźnika silhouette."""
        n_users = features.shape[0]
        valid_candidates = [k for k in self.cluster_candidates if 1 < k < n_users]
        if not valid_candidates:
            return 2 if n_users >= 2 else 1

        best_score = float("-inf")
        best_k = valid_candidates[0]
        for k in valid_candidates:
            model = KMeans(n_clusters=k, random_state=self.random_state, n_init="auto")
            labels = model.fit_predict(features)
            if len(set(labels)) == 1:
                continue
            score = silhouette_score(features, labels)
            if score > best_score:
                best_score = score
                best_k = k
        return best_k

    def _get_peer_indices(self, cluster_id: int, exclude_index: int) -> List[int]:
        """Zwróć indeksy użytkowników w tym samym klastrze z wyłączeniem celu."""
        peers = [
            idx for idx, label in enumerate(self.cluster_labels) if label == cluster_id and idx != exclude_index
        ]
        if not peers:
            peers = [idx for idx in range(len(self.cluster_labels)) if idx != exclude_index]
        return peers

    def _compute_weighted_scores(
        self,
        peer_indices: List[int],
        target_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Oblicz ważone średnie oceny wnoszone przez użytkowników w klastrze."""
        if not peer_indices:
            return self.global_item_mean, self.global_item_votes

        target_vector = self.processed_matrix[target_index]

        weights = []
        for peer_idx in peer_indices:
            peer_vector = self.processed_matrix[peer_idx]
            distance = np.linalg.norm(target_vector - peer_vector)
            weights.append(1.0 / (1.0 + distance))

        weights_array = np.array(weights).reshape(-1, 1)
        peer_ratings = self.matrix.matrix[peer_indices]

        valid_mask = ~np.isnan(peer_ratings)
        weighted_ratings = np.nan_to_num(peer_ratings) * weights_array

        weight_totals = (valid_mask * weights_array).sum(axis=0)
        summed_scores = weighted_ratings.sum(axis=0)

        scores = np.divide(
            summed_scores,
            weight_totals,
            out=np.copy(self.global_item_mean),
            where=weight_totals > 0,
        )

        vote_counts = valid_mask.sum(axis=0)
        vote_counts = np.where(vote_counts == 0, self.global_item_votes, vote_counts)
        return scores, vote_counts

    def _select_top_items(
        self,
        scores: np.ndarray,
        vote_counts: np.ndarray,
        seen_mask: np.ndarray,
        limit: int,
        reverse: bool,
    ) -> List[Tuple[str, float]]:
        """Wybierz najlepsze lub najgorsze pozycje na podstawie obliczonych ocen."""
        candidate_scores = np.copy(scores)
        candidate_scores[seen_mask] = np.nan

        if reverse:
            order = np.argsort(np.nan_to_num(candidate_scores, nan=float("-inf")))[::-1]
        else:
            order = np.argsort(np.nan_to_num(candidate_scores, nan=float("inf")))

        selections: List[Tuple[str, float]] = []
        for idx in order:
            if np.isnan(candidate_scores[idx]) or vote_counts[idx] < self.min_votes:
                continue
            title = self._item_from_index(idx)
            selections.append((title, float(candidate_scores[idx])))
            if len(selections) >= limit:
                break

        if len(selections) < limit:
            fallback_indices = np.where(~seen_mask)[0]
            additional = self._fallback_selection(fallback_indices, limit - len(selections), reverse)
            selections.extend(additional)

        return selections

    def _fallback_selection(
        self,
        candidate_indices: np.ndarray,
        remaining: int,
        reverse: bool,
    ) -> List[Tuple[str, float]]:
        """Zastosuj rekomendacje rezerwowe, używając średnich globalnych."""
        ordered = np.argsort(self.global_item_mean[candidate_indices])
        if reverse:
            ordered = ordered[::-1]

        selections: List[Tuple[str, float]] = []
        for pos in ordered:
            idx = candidate_indices[pos]
            title = self._item_from_index(idx)
            score = float(self.global_item_mean[idx])
            selections.append((title, score))
            if len(selections) >= remaining:
                break
        return selections

    def _build_recommendations(self, items: List[Tuple[str, float]]) -> List[Recommendation]:
        """Konwertuj pary (tytuł, ocena) na obiekty rekomendacji."""
        results: List[Recommendation] = []
        for title, score in items:
            metadata: Optional[MovieMetadata] = None
            if self.metadata_client:
                metadata = self.metadata_client.get_metadata(title)
            results.append(Recommendation(title=title, score=score, metadata=metadata))
        return results

    def _item_from_index(self, index: int) -> str:
        """Pobierz tytuł dla podanego indeksu kolumny w macierzy."""
        try:
            return self.matrix.items[index]
        except IndexError as error:  # pragma: no cover - defensive branch
            raise IndexError(f"No item found for index {index}") from error

