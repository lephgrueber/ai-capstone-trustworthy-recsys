"""Article-, partial- and typo-tolerant title search (M3 feedback §3)."""

from trustworthy_recsys.serving.catalog import Catalog
from trustworthy_recsys.serving.contracts import CatalogMovie
from trustworthy_recsys.serving.title_search import TitleIndex, normalize, osa_distance, substring_search

TITLES = {
    "1": "Matrix, The (1999)", "2": "Matrix Reloaded, The (2003)", "3": "Alien (1979)",
    "4": "Aliens (1986)", "5": "Vue prise d'une baleinière en marche (1901)", "6": "Amélie (2001)",
    "7": "Godfather, The (1972)", "8": "GodFather (1991)",
}
POPULARITY = {"1": 900, "2": 300, "3": 800, "4": 700, "5": 1, "6": 400, "7": 1000, "8": 2}


def test_normalization_moves_articles_and_drops_years():
    assert normalize("Matrix, The (1999)") == "the matrix"
    assert normalize("Amélie (2001)") == "amelie"


def test_osa_counts_adjacent_swap_as_one_edit():
    assert osa_distance("alein", "alien", 2) == 1
    assert osa_distance("abc", "xyz", 1) == 2  # capped at limit + 1


def test_same_film_with_or_without_article_and_exact_first():
    index = TitleIndex(TITLES, POPULARITY)
    assert index.search("The Matrix", 2) == ["1", "2"]
    assert index.search("Matrix", 1) == ["1"]
    assert index.search("amelie", 1) == ["6"]
    # The earlier substring search misses the display form of the title.
    assert "1" not in substring_search(TITLES, "The Matrix", 5)


def test_typos_and_popularity_order():
    index = TitleIndex(TITLES, POPULARITY)
    assert index.search("Alein", 2)[0] == "3"  # beats the obscure mid-word 'baleinière' match
    assert index.search("godfathr", 2) == ["7", "8"]
    assert index.search("xqzv", 5) == []


def test_catalog_search_uses_title_index():
    movies = {i: CatalogMovie(movie_id=i, title=t, genres=[]) for i, t in TITLES.items()}
    catalog = Catalog(movies, popularity=POPULARITY)
    assert [m.movie_id for m in catalog.search("the matrix", 1)] == ["1"]
    assert len(Catalog(movies).search("", 3)) == 3
