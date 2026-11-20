"""Title search that tolerates articles, partial titles and typos (feedback §3).

MovieLens stores "Matrix, The (1999)". Titles and queries are normalized (case-folded,
punctuation removed, release year dropped, a trailing ", The"/", A"/... moved to the
front) and a leading article is optional on both sides, so "The Matrix" and "Matrix"
find the same film.

Ranking tiers: 0 exact title, 1 title prefix, 2 whole-word match, 3 a substring inside a
word or a fuzzy match (a mid-word substring is no stronger evidence than a one-letter typo).
The fuzzy tier uses the Damerau–Levenshtein distance (optimal string alignment: a swap
of adjacent letters is one edit), so "Alein" finds "Alien (1979)". To stay fast on the
87k-title catalog, an inverted index over contiguous and one-skip character pairs
(which survive transpositions) shortlists candidates before computing distances
against both the full title and its prefix of the typed length. Within a tier, more
popular movies come first when popularity is supplied, then title and movie ID.

``python -m trustworthy_recsys.serving.title_search --run-dir <run>`` compares this
search with the earlier substring search on generated partial-title, omitted-article
and misspelled queries.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re
import unicodedata

import numpy as np

ARTICLES = ("the", "a", "an", "les", "la", "le", "l'", "el", "il", "der", "die", "das", "los", "las")
TRAILING_ARTICLE = re.compile(r"^(.*), (" + "|".join(re.escape(a) for a in ARTICLES) + r")$")
YEAR = re.compile(r"\s*\((\d{4})(?:[-–]\d{0,4})?\)\s*$")
NON_WORD = re.compile(r"[^\w\s']+")
SPACES = re.compile(r"\s+")
SHORTLIST = 300


def fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c)).casefold()


def normalize(title: str) -> str:
    """'Matrix, The (1999)' -> 'the matrix'."""
    text = YEAR.sub("", fold(title).strip())
    match = TRAILING_ARTICLE.match(text)
    if match:
        text = f"{match.group(2)} {match.group(1)}"
    return SPACES.sub(" ", NON_WORD.sub(" ", text)).strip()


def strip_article(text: str) -> str:
    first, _, rest = text.partition(" ")
    return rest if first in ARTICLES and rest else text


def osa_distance(a: str, b: str, limit: int) -> int:
    """Damerau–Levenshtein (optimal string alignment) distance, capped at limit + 1."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    previous2, previous = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i] + [0] * len(b)
        best = current[0]
        for j in range(1, len(b) + 1):
            cost = a[i - 1] != b[j - 1]
            value = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            if previous2 is not None and i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                value = min(value, previous2[j - 2] + 1)
            current[j] = value
            best = min(best, value)
        if best > limit:
            return limit + 1
        previous2, previous = previous, current
    return previous[-1]


def grams(text: str) -> set[str]:
    text = text.replace(" ", "")
    return {text[i:i + 2] for i in range(len(text) - 1)} | {text[i] + text[i + 2] for i in range(len(text) - 2)}


def allowed_edits(length: int) -> int:
    return 0 if length < 4 else 1 if length < 8 else 2


class TitleIndex:
    def __init__(self, titles: dict[str, str], popularity: dict[str, float] | None = None):
        self.ids = list(titles)
        self.titles = [titles[i] for i in self.ids]
        self.keys = [strip_article(normalize(t)) for t in self.titles]
        self.popularity = np.array([float((popularity or {}).get(i, 0.0)) for i in self.ids])
        postings = defaultdict(list)
        for row, key in enumerate(self.keys):
            for gram in grams(key[:40]):
                postings[gram].append(row)
        self.postings = {gram: np.array(rows, dtype=np.int64) for gram, rows in postings.items()}

    def _order(self, row, tier, distance=0):
        return (tier, -self.popularity[row], distance, self.titles[row].casefold(), self.ids[row])

    def search(self, query: str, limit: int) -> list[str]:
        needle = strip_article(normalize(query))
        if not needle:
            return [self.ids[r] for r in sorted(range(len(self.ids)), key=lambda r: self._order(r, 0))[:limit]]
        ranked = {}
        padded = f" {needle} "
        for row, key in enumerate(self.keys):
            if key == needle:
                ranked[row] = self._order(row, 0)
            elif key.startswith(needle):
                ranked[row] = self._order(row, 1)
            elif padded in f" {key} ":
                ranked[row] = self._order(row, 2)
            elif needle in key:
                ranked[row] = self._order(row, 3)
        limit_edits = allowed_edits(len(needle))
        strong = sum(1 for order in ranked.values() if order[0] < 3)
        if limit_edits and strong < limit:
            query_grams = [self.postings[g] for g in grams(needle) if g in self.postings]
            if query_grams:
                counts = np.bincount(np.concatenate(query_grams), minlength=len(self.keys))
                shortlist = np.argsort(-counts, kind="stable")[:SHORTLIST]
                for row in shortlist:
                    if row in ranked or not counts[row]:
                        continue
                    key = self.keys[row]
                    distance = min(osa_distance(needle, key, limit_edits),
                                   osa_distance(needle, key[:len(needle)], limit_edits))
                    if distance <= limit_edits:
                        ranked[row] = self._order(row, 3, distance)
        return [self.ids[row] for row in sorted(ranked, key=ranked.get)[:limit]]


def substring_search(titles: dict[str, str], query: str, limit: int) -> list[str]:
    """The earlier catalog search, kept for comparison."""
    needle = query.casefold().strip()
    matches = [i for i, t in titles.items() if not needle or needle in t.casefold()]

    def order(movie_id):
        title = titles[movie_id].casefold()
        bare = title.rsplit(" (", 1)[0] if title.endswith(")") else title
        return (0 if bare == needle else 1 if bare.startswith(needle) else 2), title, movie_id

    return sorted(matches, key=order)[:limit]


def query_cases(titles, popularity, n=300, seed=0):
    """Partial titles, omitted/added articles and misspellings for popular movies."""
    rng = np.random.default_rng(seed)
    top = sorted(titles, key=lambda i: -popularity.get(i, 0))[:n]
    cases = []
    for movie_id in top:
        full = normalize(titles[movie_id])
        bare = strip_article(full)
        words = bare.split()
        if len(bare) < 5:
            continue
        cases.append(("display_title", " ".join(w.capitalize() for w in full.split()), movie_id))
        if bare != full:
            cases.append(("omitted_article", bare, movie_id))
        if len(words) > 1:
            cases.append(("partial_title", " ".join(words[: max(1, len(words) // 2)]), movie_id))
        cases.append(("partial_word", bare[: max(4, int(len(bare) * .6))], movie_id))
        position = int(rng.integers(1, len(bare) - 2))
        if bare[position] != " " and bare[position + 1] != " ":
            swapped = bare[:position] + bare[position + 1] + bare[position] + bare[position + 2:]
            cases.append(("swapped_letters", swapped, movie_id))
        drop = int(rng.integers(1, len(bare)))
        if bare[drop] != " ":
            cases.append(("missing_letter", bare[:drop] + bare[drop + 1:], movie_id))
    return cases


def compare(titles, popularity, k=5):
    """Share of cases whose intended movie is in the first k results, per case type."""
    index = TitleIndex(titles, popularity)
    results = defaultdict(lambda: {"cases": 0, "substring": 0, "fuzzy": 0})
    for kind, query, movie_id in query_cases(titles, popularity):
        row = results[kind]
        row["cases"] += 1
        row["substring"] += movie_id in substring_search(titles, query, k)
        row["fuzzy"] += movie_id in index.search(query, k)
    return {kind: {"cases": r["cases"], f"substring_hit_at_{k}": r["substring"] / r["cases"],
                   f"fuzzy_hit_at_{k}": r["fuzzy"] / r["cases"]} for kind, r in sorted(results.items())}


def main():
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--scenario", default="ratio_80_10_10")
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    movies = pq.read_table(f"{a.run_dir}/movies.parquet", columns=["movie_id", "title"])
    titles = dict(zip(movies["movie_id"].to_pylist(), movies["title"].to_pylist()))
    counts = pc.value_counts(pq.read_table(f"{a.run_dir}/{a.scenario}/train.parquet", columns=["movie_id"])["movie_id"])
    popularity = {row["values"]: row["counts"] for row in counts.to_pylist()}
    print(json.dumps(compare(titles, popularity, a.k), indent=2))


if __name__ == "__main__":
    main()
