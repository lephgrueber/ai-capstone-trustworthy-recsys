"""Versioned contracts for pipeline artifacts and the existing evaluator."""

import pyarrow as pa

SCHEMA_VERSION = "1.0"
UTC = pa.timestamp("ms", tz="UTC")
INTERACTIONS = pa.schema([
    pa.field("user_id", pa.string(), nullable=False),
    pa.field("movie_id", pa.string(), nullable=False),
    pa.field("rating", pa.float32(), nullable=False),
    pa.field("timestamp_utc", UTC, nullable=False),
])
MOVIES = pa.schema([
    pa.field("movie_id", pa.string(), nullable=False),
    pa.field("title", pa.string(), nullable=False),
    pa.field("genres", pa.list_(pa.string()), nullable=False),
])
LINKS = pa.schema([
    pa.field("movie_id", pa.string(), nullable=False),
    pa.field("imdb_id", pa.string()), pa.field("tmdb_id", pa.string()),
])
TAGS = pa.schema([
    pa.field("user_id", pa.string(), nullable=False),
    pa.field("movie_id", pa.string(), nullable=False),
    pa.field("tag", pa.string(), nullable=False),
    pa.field("timestamp_utc", UTC, nullable=False),
])


def describe():
    """Provide a machine-readable schema without requiring Arrow to inspect it."""
    return {
        "version": SCHEMA_VERSION,
        "tables": {name: [{"name": f.name, "type": str(f.type), "nullable": f.nullable}
                          for f in schema]
                   for name, schema in [("interactions", INTERACTIONS), ("movies", MOVIES),
                                        ("links", LINKS), ("tags", TAGS)]},
        "evaluation_jsonl": {
            "user_id": "nonempty string; unique within file",
            "history_items": "list[string]; positive training movies, oldest first, ties by numeric movie ID",
            "relevant_items": "nonempty list[string]; positive held-out movies, oldest first, ties by numeric movie ID",
        },
        "ids": "Original decimal IDs represented as strings; no fitted ID remapping.",
        "split_rule": "train < validation_start <= validation < test_start <= test; UTC",
    }
