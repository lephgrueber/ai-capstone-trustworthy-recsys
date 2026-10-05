"""Exact cosine reference for checkpoint selection and Weaviate comparisons."""
import numpy as np


def validate_vectors(vectors):
    vectors=np.ascontiguousarray(vectors,dtype='float32')
    if vectors.ndim!=2 or not len(vectors) or not np.isfinite(vectors).all():
        raise ValueError('Expected a finite, nonempty item matrix')
    if not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=1e-4):
        raise ValueError('ANN item embeddings must be unit-normalized')
    return vectors


class ExactIndex:
    """Brute-force reference, never a silent fallback for a database outage."""

    def __init__(self, vectors):
        self.vectors = validate_vectors(vectors)
        self.ntotal = len(self.vectors)

    def search_filtered(self, query, k, excluded):
        if k <= 0:
            raise ValueError('k must be positive')
        scores = self.vectors @ np.asarray(query, dtype='float32')
        eligible = np.ones(self.ntotal, dtype=bool)
        if excluded:
            eligible[list(excluded)] = False
        ids = np.flatnonzero(eligible)
        return ids[np.lexsort((ids, -scores[ids]))[:k]].tolist()


def filtered_search(index,query,k,excluded):
    return index.search_filtered(query,k,excluded)
