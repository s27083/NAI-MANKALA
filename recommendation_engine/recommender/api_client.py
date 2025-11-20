"""Klient odpowiedzialny za wzbogacanie tytułów metadanymi z zewnętrznego API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

import requests

from .data_models import MovieMetadata

OMDB_BASE_URL = "https://www.omdbapi.com/"
DEMO_API_KEY = "thewdb"
TITLE_ALIASES: Dict[str, str] = {
    "365 dni": "365 Days",
    "Kraina Lodu": "Frozen",
    "Kobiety Mafii": "Women of Mafia",
    "Wieloryb": "The Whale",
    "Jak sprzedawać dragi w sieci (szybko)": "How to Sell Drugs Online (Fast)",
    "Wszystko, wszędzie naraz": "Everything Everywhere All at Once",
    "Zaginiona dziewczyna": "Gone Girl",
    "Dom z papieru": "Money Heist",
    "Skazany na śmierć": "Prison Break",
    "Skazani na Shawshank (The Shawshank Redemption)": "The Shawshank Redemption",
    "Lot nad kukułczym gniazdem (One Flew Over the Cuckoo's Nest)": "One Flew Over the Cuckoo's Nest",
    "Źródło (The Fountain)": "The Fountain",
    "Święta góra (The Holy Mountain)": "The Holy Mountain",
    "Podziemny krąg (Fight Club)": "Fight Club",
    "W pogoni za szczęściem (The Pursuit of Happyness)": "The Pursuit of Happyness",
    "Efekt motyla (The Butterfly Effect)": "The Butterfly Effect",
    "Mechanik (El Maquinista)": "The Machinist",
    "Plan doskonały (Inside Man)": "Inside Man",
    "Odyseja kosmiczna (2001: A Space Odyssey)": "2001: A Space Odyssey",
    "Milczenie owiec (The Silence of the Lambs)": "The Silence of the Lambs",
    "Gra (The Game)": "The Game",
    "Dziewiąte wrota (The Ninth Gate)": "The Ninth Gate",
    "Tożsamość Bourne'a (The Bourne Identity)": "The Bourne Identity",
    "Jestem bogiem (Limitless)": "Limitless",
    "W sieci zła (Fallen)": "Fallen",
    "W objęciach węża (El abrazo de la serpiente)": "Embrace of the Serpent",
    "Mały Otik": "Little Otik",
    "Cicha Noc": "Silent Night",
    "Narodziny Gwiazdy": "A Star Is Born",
    "List do M.": "Letters to Santa",
    "List do M. 2": "Letters to Santa 2",
    "Planeta Singli": "Planet Single",
    "Nietykalni": "The Intouchables",
    "Millerowie": "We're the Millers",
    "Furia": "Fury",
    "Super tata": "Big Daddy",
    "Żona na niby": "Wife on Paper",
}


@dataclass()
class MovieMetadataClient:
    """Hermetyzuje logikę pobierania metadanych dla filmów/seriali."""

    api_key: Optional[str] = None
    session: requests.Session = field(default_factory=requests.Session)
    _memory_cache: Dict[str, MovieMetadata] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.getenv("OMDB_API_KEY") or DEMO_API_KEY

    def get_metadata(self, title: str) -> Optional[MovieMetadata]:
        """Pobierz metadane dla tytułu; cache’owane w trakcie działania procesu."""
        normalized_title = title.strip()
        if normalized_title in self._memory_cache:
            return self._memory_cache[normalized_title]

        for candidate in self._iter_query_variants(normalized_title):
            payload = self._fetch_from_api(candidate)
            if payload:
                metadata = self._to_movie_metadata(normalized_title, payload, source="omdb")
                self._memory_cache[normalized_title] = metadata
                return metadata

        return None

    def _iter_query_variants(self, original_title: str) -> Iterable[str]:
        """Generuj możliwe warianty zapytania dla podanego tytułu."""
        seen: set[str] = set()
        candidates: list[str] = [original_title]

        alias = TITLE_ALIASES.get(original_title)
        if alias:
            candidates.append(alias)

        if "(" in original_title and ")" in original_title:
            inside = original_title.split("(", 1)[1].rsplit(")", 1)[0]
            candidates.append(inside)

        if ":" in original_title:
            after_colon = original_title.split(":", 1)[1]
            candidates.append(after_colon)

        if " - " in original_title:
            after_dash = original_title.split(" - ", 1)[1]
            candidates.append(after_dash)

        for candidate in candidates:
            if not candidate:
                continue
            normalized = candidate.strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            yield normalized

    def _fetch_from_api(self, title: str) -> Optional[dict]:
        """Spróbuj pobrać dane z OMDb."""
        if not self.api_key:
            return None

        params = {"t": title, "apikey": self.api_key, "plot": "short"}
        try:
            response = self.session.get(OMDB_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException:
            return None

        data = response.json()
        if data.get("Response") != "True":
            return None
        return data

    def _to_movie_metadata(
        self,
        title: str,
        payload: dict,
        source: Optional[str] = None,
    ) -> MovieMetadata:
        """Konwertuj surowe dane odpowiedzi API na obiekt metadanych."""
        genres_raw = payload.get("Genre", "")
        genres = [genre.strip() for genre in genres_raw.split(",") if genre.strip()]

        runtime_raw = payload.get("Runtime")
        runtime_minutes: Optional[int] = None
        if isinstance(runtime_raw, str) and runtime_raw.lower().endswith("min"):
            runtime_value = runtime_raw.lower().replace("min", "").strip()
            if runtime_value.isdigit():
                runtime_minutes = int(runtime_value)

        imdb_rating = payload.get("imdbRating")
        try:
            imdb_value = float(imdb_rating) if imdb_rating and imdb_rating != "N/A" else None
        except (TypeError, ValueError):
            imdb_value = None

        return MovieMetadata(
            title=title,
            year=payload.get("Year"),
            genres=genres,
            plot=payload.get("Plot") or payload.get("plot"),
            poster_url=payload.get("Poster") if payload.get("Poster") != "N/A" else None,
            imdb_rating=imdb_value,
            runtime_minutes=runtime_minutes,
            source=source or "omdb",
        )
