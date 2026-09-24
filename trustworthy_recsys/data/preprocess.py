"""Minimal documented transformations; raw files are never edited."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import schema


def load_metadata(raw):
    movies = pd.read_csv(Path(raw)/"movies.csv", dtype={"movieId": "int64"})
    links = pd.read_csv(Path(raw)/"links.csv", dtype={"movieId": "int64", "imdbId": "string", "tmdbId": "string"})
    for frame, columns in [(movies, ["movieId", "title", "genres"]), (links, ["movieId", "imdbId", "tmdbId"])]:
        if list(frame.columns) != columns or frame.movieId.isna().any() or not frame.movieId.gt(0).all() or frame.movieId.duplicated().any():
            raise ValueError("Metadata has invalid columns or movie IDs")
    if movies.title.isna().any() or movies.title.str.strip().eq("").any():
        raise ValueError("Movie titles must be nonempty")
    if set(movies.movieId) != set(links.movieId):
        raise ValueError("Movie and links catalog IDs differ")
    return movies, links


def interaction_table(frame):
    return pa.Table.from_arrays([
        pa.array(frame.userId).cast(pa.string()), pa.array(frame.movieId).cast(pa.string()),
        pa.array(frame.rating, type=pa.float32()),
        pa.array(frame.timestamp.to_numpy()*1000, type=schema.UTC),
    ], schema=schema.INTERACTIONS)


def write_metadata(movies, links, output):
    output = Path(output)
    genres = movies.genres.fillna("(no genres listed)")
    genre_lists = [value.split("|") if value != "(no genres listed)" else [] for value in genres]
    pq.write_table(pa.Table.from_arrays([
        pa.array(movies.movieId).cast(pa.string()), pa.array(movies.title),
        pa.array(genre_lists, type=pa.list_(pa.string()))], schema=schema.MOVIES), output/"movies.parquet")
    pq.write_table(pa.Table.from_arrays([
        pa.array(links.movieId).cast(pa.string()), pa.array(links.imdbId, type=pa.string()),
        pa.array(links.tmdbId, type=pa.string())], schema=schema.LINKS), output/"links.parquet")
    return {"catalog_rows": len(movies), "movies_without_listed_genres": int(genres.eq("(no genres listed)").sum()),
            "missing_imdb_ids": int(links.imdbId.isna().sum()), "missing_tmdb_ids": int(links.tmdbId.isna().sum()),
            "duplicate_title_strings": int(movies.title.duplicated().sum())}


def write_tags(raw, output, movie_ids, user_ids, cutoff_map, chunk_size):
    """Write all cleaned tags plus strictly historical tag tables per split run."""
    summary = {"input_rows": 0, "output_rows": 0, "blank_tags_removed": 0,
               "whitespace_trimmed_rows": 0, "users_without_ratings_rows": 0,
               "training_tag_rows": {name: 0 for name in cutoff_map}}
    tagged_movies, tagging_users = set(), set()
    writers = {"all": pq.ParquetWriter(Path(output)/"tags.parquet", schema.TAGS)}
    writers.update({name: pq.ParquetWriter(Path(output)/name/"train_tags.parquet", schema.TAGS) for name in cutoff_map})
    reader = pd.read_csv(Path(raw)/"tags.csv", chunksize=chunk_size, keep_default_na=False,
                         dtype={"userId": "int64", "movieId": "int64", "tag": str, "timestamp": "int64"})
    try:
        for frame in reader:
            if list(frame.columns) != ["userId", "movieId", "tag", "timestamp"]:
                raise ValueError("tags.csv: incorrect columns")
            if not frame.userId.gt(0).all() or not frame.movieId.isin(movie_ids).all() or not frame.timestamp.ge(0).all():
                raise ValueError("Invalid tag IDs, movie reference, or timestamp")
            pd.to_datetime(frame.timestamp, unit="s", utc=True, errors="raise")
            summary["input_rows"] += len(frame)
            summary["users_without_ratings_rows"] += int((~frame.userId.isin(user_ids)).sum())
            trimmed = frame.tag.str.strip()
            summary["whitespace_trimmed_rows"] += int(trimmed.ne(frame.tag).sum())
            summary["blank_tags_removed"] += int(trimmed.eq("").sum())
            frame = frame.assign(tag=trimmed).loc[trimmed.ne("")]
            summary["output_rows"] += len(frame)
            tagged_movies.update(frame.movieId)
            tagging_users.update(frame.userId)
            for name, subset in [("all", frame), *[(name, frame.loc[frame.timestamp < cutoffs[0]]) for name, cutoffs in cutoff_map.items()]]:
                table = pa.Table.from_arrays([
                    pa.array(subset.userId).cast(pa.string()), pa.array(subset.movieId).cast(pa.string()),
                    pa.array(subset.tag), pa.array(subset.timestamp.to_numpy()*1000, type=schema.UTC)], schema=schema.TAGS)
                if len(subset):
                    writers[name].write_table(table)
                if name != "all":
                    summary["training_tag_rows"][name] += len(subset)
    finally:
        reader.close()
        for writer in writers.values():
            writer.close()
    summary.update({"tagging_users": len(tagging_users), "tagged_movies": len(tagged_movies),
                    "catalog_tag_coverage_pct": 100*len(tagged_movies)/len(movie_ids)})
    return summary
