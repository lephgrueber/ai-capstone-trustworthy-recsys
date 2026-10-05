"""Weaviate storage for self-provided two-tower vectors and filtered retrieval."""

import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import weaviate
from weaviate.classes.config import Configure, DataType, Property, Tokenization, VectorDistances
from weaviate.util import generate_uuid5

from .index import validate_vectors


INDEX_CONFIG = {'kind': 'Weaviate HNSW', 'metric': 'cosine', 'max_connections': 32,
                'ef_construction': 100, 'ef': 128, 'quantization': 'none',
                'vectorizer': 'self_provided', 'filtering': 'adaptive overfetch, client-side seen exclusion'}


def connect():
    """Connect to the local/self-hosted service; credentials are never artifacts."""
    return weaviate.connect_to_local(
        host=os.environ.get('WEAVIATE_HOST', 'localhost'),
        port=int(os.environ.get('WEAVIATE_PORT', '8080')),
        grpc_port=int(os.environ.get('WEAVIATE_GRPC_PORT', '50051')),
    )


def fingerprint(item_ids, vectors):
    digest = hashlib.sha256(json.dumps(item_ids, separators=(',', ':')).encode())
    digest.update(np.asarray(vectors, dtype='<f4').tobytes())
    return digest.hexdigest()


class WeaviateIndex:
    def __init__(self, client, collection, item_ids, vectors, descriptor):
        self.client = client
        self.collection = collection
        self.item_ids = item_ids
        self.ntotal = len(item_ids)
        self.dimension = vectors.shape[1]
        self.descriptor = descriptor

    @classmethod
    def create(cls, name, item_ids, vectors):
        vectors = validate_vectors(vectors)
        if len(item_ids) != len(vectors) or len(set(item_ids)) != len(item_ids):
            raise ValueError('Movie IDs must be unique and aligned with vectors')
        if not re.fullmatch(r'[A-Z][A-Za-z0-9_]*', name):
            raise ValueError('Collection name must start with an uppercase letter and contain letters, digits or underscores')
        identity = fingerprint(item_ids, vectors)
        client = connect()
        try:
            if client.collections.exists(name):
                raise ValueError('Collection already exists; choose a new versioned collection name')
            collection = client.collections.create(
                name=name, description=f'two-tower:{identity}',
                vector_config=Configure.Vectors.self_provided(
                    vector_index_config=Configure.VectorIndex.hnsw(
                        distance_metric=VectorDistances.COSINE, max_connections=32,
                        ef_construction=100, ef=128, flat_search_cutoff=0,
                        quantizer=Configure.VectorIndex.Quantizer.none(),
                    )
                ),
                properties=[
                    Property(name='movie_id', data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                    Property(name='item_index', data_type=DataType.INT),
                    Property(name='embedding_fingerprint', data_type=DataType.TEXT, tokenization=Tokenization.FIELD),
                ],
            )
            with collection.batch.fixed_size(batch_size=256, concurrent_requests=2) as batch:
                for index, (movie_id, vector) in enumerate(zip(item_ids, vectors)):
                    batch.add_object(
                        uuid=generate_uuid5(f'{identity}:{movie_id}'),
                        properties={'movie_id': movie_id, 'item_index': index, 'embedding_fingerprint': identity},
                        vector=vector.tolist(),
                    )
            if collection.batch.failed_objects:
                raise RuntimeError(f'Weaviate import failed for {len(collection.batch.failed_objects)} objects; collection is incomplete')
            descriptor = {'schema_version': '1.0', 'collection': name, 'fingerprint': identity,
                          'items': len(item_ids), 'dimensions': vectors.shape[1],
                          'index': INDEX_CONFIG, 'server_version': client.get_meta()['version'],
                          'client_version': weaviate.__version__}
            result = cls(client, collection, item_ids, vectors, descriptor)
            result.verify(vectors)
            return result
        except BaseException:
            client.close()
            raise

    @classmethod
    def load(cls, descriptor_path, item_ids, vectors):
        path = Path(descriptor_path)
        if not path.exists():
            raise ValueError('Model is not published to Weaviate. Run python -m trustworthy_recsys.retrieval.publish first.')
        descriptor = json.loads(path.read_text(encoding='utf-8'))
        vectors = validate_vectors(vectors)
        if descriptor['fingerprint'] != fingerprint(item_ids, vectors):
            raise ValueError('Weaviate descriptor does not match model embeddings/catalog')
        if descriptor['items'] != len(item_ids) or descriptor['dimensions'] != vectors.shape[1]:
            raise ValueError('Weaviate descriptor shape mismatch')
        client = connect()
        try:
            if not client.collections.exists(descriptor['collection']):
                raise ValueError('Published Weaviate collection is missing')
            result = cls(client, client.collections.use(descriptor['collection']), item_ids, vectors, descriptor)
            result.verify(vectors)
            return result
        except BaseException:
            client.close()
            raise

    def verify(self, vectors):
        """Audit every remote ID and vector, not merely the database object count."""
        config = self.collection.config.get()
        if config.description != f"two-tower:{self.descriptor['fingerprint']}":
            raise ValueError('Weaviate collection fingerprint mismatch')
        vector_config = config.vector_config['default']
        index_config = vector_config.vector_index_config
        if (vector_config.vectorizer.vectorizer.value != 'none'
                or index_config.distance_metric != VectorDistances.COSINE
                or index_config.max_connections != 32 or index_config.ef_construction != 100
                or index_config.ef != 128 or index_config.quantizer is not None
                or index_config.flat_search_cutoff != 0):
            raise ValueError('Weaviate collection index configuration changed')
        if self.collection.aggregate.over_all(total_count=True).total_count != self.ntotal:
            raise ValueError('Weaviate collection size mismatch')
        seen = set()
        for obj in self.collection.iterator(include_vector=True):
            index = obj.properties['item_index']
            if not isinstance(index, int) or not 0 <= index < self.ntotal or index in seen:
                raise ValueError('Duplicate or invalid Weaviate item index')
            if obj.properties['movie_id'] != self.item_ids[index] or obj.properties['embedding_fingerprint'] != self.descriptor['fingerprint']:
                raise ValueError('Weaviate movie mapping/fingerprint mismatch')
            stored = np.asarray(obj.vector['default'], dtype='float32')
            if stored.shape != vectors[index].shape or not np.array_equal(stored, vectors[index]):
                raise ValueError('Weaviate vector differs from saved model embedding')
            seen.add(index)
        if len(seen) != self.ntotal:
            raise ValueError('Incomplete Weaviate catalog iteration')

    def search_filtered(self, query, k, excluded):
        if k <= 0:
            raise ValueError('k must be positive')
        query = np.asarray(query, dtype='float32')
        if query.shape != (self.dimension,) or not np.isfinite(query).all():
            raise ValueError('Query has invalid dimension or nonfinite values')
        if len(excluded) >= self.ntotal:
            return []
        requested = min(self.ntotal, max(k * 3, k + min(len(excluded), 500)))
        while True:
            response = self.collection.query.near_vector(
                near_vector=query.tolist(), limit=requested,
                return_properties=['movie_id', 'item_index', 'embedding_fingerprint'],
            )
            ranked, seen = [], set()
            for obj in response.objects:
                index = obj.properties['item_index']
                if not isinstance(index, int) or not 0 <= index < self.ntotal or index in seen:
                    raise ValueError('Invalid or duplicate Weaviate search result')
                if obj.properties['movie_id'] != self.item_ids[index] or obj.properties['embedding_fingerprint'] != self.descriptor['fingerprint']:
                    raise ValueError('Search returned a mismatched model/movie')
                seen.add(index)
                if index not in excluded:
                    ranked.append(index)
            if len(ranked) >= k or requested == self.ntotal:
                return ranked[:k]
            requested = min(self.ntotal, requested * 2)

    def close(self):
        self.client.close()
