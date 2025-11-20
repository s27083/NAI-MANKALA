"""
Problem: Dostarczyć silnik rekomendacji filmów/seriali oparty o klastrowanie,
         antyrekomendacje, wzbogacanie metadanymi oraz powtarzalny CLI.
Autorzy: Adrian Kemski s27444 i Kamil Bogdański s27083
Użycie:
    export OMDB_API_KEY=<twój_klucz>  # opcjonalnie; domyślnie klucz demo OMDb
    python main.py --user "Adrian Kemski"
Referencje:
    - Ishika Chatterjee, "A Comparative Study of Clustering Algorithms", Analytics Vidhya, 2020.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from recommender.api_client import MovieMetadataClient
from recommender.engine import RecommendationEngine
from recommender.data_models import Recommendation, RecommendationResult

DEFAULT_DATA_PATH = Path(__file__).parent / "data" / "user_ratings.csv"
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parsuj argumenty linii poleceń."""
    parser = argparse.ArgumentParser(
        description="Movie/series recommendation engine CLI.",
        epilog="Metadata is fetched live from OMDb. Provide OMDB_API_KEY env var or the demo key is used.",
    )
    parser.add_argument("--user", required=True, help="Name of the user to generate recommendations for.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Path to the ratings CSV file.",
    )
    parser.add_argument("--top-n", type=int, default=5, help="Number of positive recommendations.")
    parser.add_argument("--anti-n", type=int, default=5, help="Number of anti-recommendations.")
    parser.add_argument(
        "--clusters",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Candidate cluster counts evaluated via silhouette score.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=2,
        help="Minimum number of peer votes required to consider a title.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the report.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the report output.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Punkt wejścia dla CLI."""
    args = parse_args(argv)

    metadata_client = MovieMetadataClient()

    engine = RecommendationEngine(
        metadata_client=metadata_client,
        cluster_candidates=tuple(args.clusters),
        min_votes=args.min_votes,
    )
    engine.fit_from_csv(args.data)

    result = engine.recommend(
        user_name=args.user,
        top_n=args.top_n,
        anti_n=args.anti_n,
    )

    if args.format == "json":
        output = _format_json(result)
    else:
        output = _format_text(result)

    print(output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")


def _format_text(result: RecommendationResult) -> str:
    """Generuj czytelny raport tekstowy."""
    lines = [
        f"Rekomendacje dla: {result.user}",
        "",
        "Top propozycje:",
    ]
    lines.extend(_format_recommendation_section(result.recommended))
    lines.append("")
    lines.append("Antyrekomendacje:")
    lines.extend(_format_recommendation_section(result.anti_recommended))
    return "\n".join(lines)


def _format_recommendation_section(entries: Iterable[Recommendation]) -> List[str]:
    """Wyrenderuj sekcję rekomendacji do wyjścia tekstowego."""
    formatted: List[str] = []
    for idx, recommendation in enumerate(entries, start=1):
        line = f"{idx}. {recommendation.title} (ocena: {recommendation.score:.2f})"
        if recommendation.metadata:
            meta = recommendation.metadata
            details = []
            if meta.year:
                details.append(meta.year)
            if meta.genres:
                details.append("/".join(meta.genres[:3]))
            if meta.imdb_rating is not None:
                details.append(f"IMDb {meta.imdb_rating}")
            if details:
                line += f" — {' | '.join(details)}"
            if meta.plot:
                line += f"\n   Opis: {meta.plot}"
        formatted.append(line)
    if not formatted:
        formatted.append("   (brak pozycji spełniających kryteria)")
    return formatted


def _format_json(result: RecommendationResult) -> str:
    """Zserializuj rekomendacje do formatu JSON."""
    payload = {
        "user": result.user,
        "recommended": [_serialize_entry(entry) for entry in result.recommended],
        "anti_recommended": [_serialize_entry(entry) for entry in result.anti_recommended],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _serialize_entry(entry: Recommendation) -> dict:
    """Konwertuj pozycję rekomendacji do słownika JSON-owalnego."""
    metadata = entry.metadata
    return {
        "title": entry.title,
        "score": entry.score,
        "metadata": {
            "year": metadata.year if metadata else None,
            "genres": metadata.genres if metadata else None,
            "plot": metadata.plot if metadata else None,
            "poster_url": metadata.poster_url if metadata else None,
            "imdb_rating": metadata.imdb_rating if metadata else None,
            "runtime_minutes": metadata.runtime_minutes if metadata else None,
            "source": metadata.source if metadata else None,
        },
    }


if __name__ == "__main__":  # pragma: no cover
    main()

