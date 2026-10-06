"""Supported-catalog loading and conservative metadata filtering."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trustworthy_recsys.serving.contracts import CatalogMovie, PreferenceState


@dataclass(frozen=True)
class Catalog:
    movies: dict[str, CatalogMovie]

    @property
    def supported_ids(self) -> frozenset[str]:
        return frozenset(self.movies)

    @property
    def genres(self) -> frozenset[str]:
        return frozenset(genre for movie in self.movies.values() for genre in movie.genres)

    def search(self, query: str, limit: int) -> list[CatalogMovie]:
        needle = query.casefold().strip()
        values = self.movies.values()
        matches = [movie for movie in values if not needle or needle in movie.title.casefold()]
        return sorted(matches, key=lambda movie: (movie.title.casefold(), movie.movie_id))[:limit]

    def allows(self, movie_id: str, preferences: PreferenceState) -> bool:
        movie = self.movies.get(movie_id)
        if movie is None:
            return False
        if movie_id in preferences.history_items or movie_id in preferences.excluded_movie_ids:
            return False
        if preferences.include_genres or preferences.exclude_genres:
            # Missing genre metadata cannot prove a hard constraint, so exclude it.
            if not movie.genres:
                return False
            genres = set(movie.genres)
            if preferences.include_genres and not genres.intersection(preferences.include_genres):
                return False
            if genres.intersection(preferences.exclude_genres):
                return False
        return True

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"movie_id": movie.movie_id, "genres": movie.genres} for movie in self.movies.values()]
        )


FIXTURE_MOVIES = (
    ("101", "Signal in the Fog", ["Mystery", "Thriller"]),
    ("102", "Orbit House", ["Science Fiction", "Drama"]),
    ("103", "The Last Matinee", ["Comedy", "Drama"]),
    ("104", "Juniper Road", ["Drama", "Romance"]),
    ("105", "Clockwork Harbor", ["Adventure", "Fantasy"]),
    ("106", "Paper Moons", ["Animation", "Family"]),
    ("107", "Night Bus North", ["Crime", "Thriller"]),
    ("108", "A Quiet Encore", ["Drama", "Music"]),
    ("109", "Kitchen Table", ["Comedy", "Romance"]),
    ("110", "Red Meridian", ["Action", "Science Fiction"]),
    ("111", "Small Hours", ["Documentary"]),
    ("112", "The Glass Orchard", ["Fantasy", "Mystery"]),
    ("113", "Second Summer", ["Comedy", "Family"]),
    ("114", "Ash & Snow", ["Drama", "War"]),
    ("115", "Untitled Reel", []),
)


def fixture_catalog() -> Catalog:
    return Catalog(
        {movie_id: CatalogMovie(movie_id=movie_id, title=title, genres=genres) for movie_id, title, genres in FIXTURE_MOVIES}
    )


def catalog_from_parquet(path) -> Catalog:
    frame = pd.read_parquet(path, columns=["movie_id", "title", "genres"])
    return Catalog(
        {
            str(row.movie_id): CatalogMovie(
                movie_id=str(row.movie_id), title=str(row.title), genres=list(row.genres)
            )
            for row in frame.itertuples(index=False)
        }
    )
