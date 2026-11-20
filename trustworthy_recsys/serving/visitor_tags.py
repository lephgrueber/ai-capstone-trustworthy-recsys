"""Tags that visitors add while reviewing a movie (feedback §4), off by default.

With ``TRUSTWORTHY_RECSYS_VISITOR_TAGS=record`` the server keeps each new (movie, phrase)
with the time it was first used, like a MovieLens tag, optionally in SQLite
(``TRUSTWORTHY_RECSYS_VISITOR_TAG_DB``). A phrase reuses its stored vector when it is
already known; a new phrase is encoded with the same text encoder as the training tags.
The movie's tag summary used by the reranker's tag features is then updated in place.

Visitor tags influence every visitor's recommendations, which is why recording is opt-in.
Refreshing a movie's retrieval-index vector is not done here: only towers trained with
tag inputs would need it, and they must be republished (retrieval.publish) to pick up
new tags.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3
import threading

from trustworthy_recsys.data.tags import normalize
from trustworthy_recsys.serving.contracts import PreferenceState

SCHEMA = """CREATE TABLE IF NOT EXISTS visitor_tags (
    movie_id TEXT NOT NULL, phrase TEXT NOT NULL, first_used REAL NOT NULL,
    PRIMARY KEY (movie_id, phrase))"""


class VisitorTagStore:
    def __init__(self, path: Path | None = None, features=None, encoder=None):
        self.lock = threading.Lock()
        self.connection = None
        self.seen: dict[tuple[str, str], float] = {}
        self.features = features
        self.encoder = encoder
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(path), check_same_thread=False)
            self.connection.execute(SCHEMA)
            self.connection.commit()
            for movie_id, phrase, first in self.connection.execute("SELECT movie_id, phrase, first_used FROM visitor_tags"):
                self.seen[(movie_id, phrase)] = first
            self._apply(list(self.seen))

    def record(self, preferences: PreferenceState, supported_ids=frozenset()) -> list[tuple[str, str]]:
        """Store new (movie, phrase) pairs from a request; returns the pairs that were new."""
        added = []
        for record in preferences.ratings or []:
            if not record.tags or (supported_ids and record.movie_id not in supported_ids):
                continue
            stamp = record.tagged_at.timestamp()
            for tag in record.tags:
                key = (record.movie_id, normalize(tag))
                if key[1] and key not in self.seen:
                    added.append((key, stamp))
        if not added:
            return []
        with self.lock:
            added = [(key, stamp) for key, stamp in added if key not in self.seen]
            for key, stamp in added:
                self.seen[key] = stamp
            if self.connection is not None and added:
                self.connection.executemany("INSERT OR IGNORE INTO visitor_tags VALUES (?, ?, ?)",
                                            [(m, p, s) for (m, p), s in added])
                self.connection.commit()
        self._apply([key for key, _ in added])
        return [key for key, _ in added]

    def _apply(self, pairs):
        features = self.features
        if features is None or getattr(features, "tag_vectors", None) is None or not pairs:
            return
        for movie_id, phrase in pairs:
            index = features.item_map.get(movie_id)
            if index is not None:
                features.add_movie_tags(index, [phrase], self.encoder)

    def close(self):
        if self.connection is not None:
            self.connection.close()
