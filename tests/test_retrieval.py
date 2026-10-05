"""Temporal leakage, artifact tampering and real train/index/harness integration."""

import json
import os
from uuid import uuid4
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip('torch')
pytest.importorskip('weaviate')

from eval.harness import RecommendationRequest, SavedPredictionsRecommender, load_evaluation_examples, evaluate as harness_evaluate
from trustworthy_recsys.data.pipeline import run_pipeline
from trustworthy_recsys.data.retrieval_inputs import prepare_inputs, ModelInputs
from trustworthy_recsys.retrieval.train import train, negative_mask
from trustworthy_recsys.retrieval.recommender import RetrievalRecommender
from trustworthy_recsys.retrieval.evaluate import evaluate
from trustworthy_recsys.retrieval.model import TwoTower
from trustworthy_recsys.retrieval.publish import publish
from trustworthy_recsys.retrieval.weaviate_store import connect, WeaviateIndex
from trustworthy_recsys.retrieval.index import ExactIndex
import test_data_pipeline as fixtures


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('WEAVIATE_INTEGRATION') != '1', reason='Set WEAVIATE_INTEGRATION=1 with local Weaviate running')
def test_end_to_end_temporal_training_and_roundtrip(tmp_path):
    fixture=fixtures.PipelineTests()
    fixture.setUp()
    collection='RetrievalTest'+uuid4().hex
    try:
        fixture.write_ratings([(1,1,5,1),(1,2,5,2),(1,3,5,2),(1,4,5,5),(1,5,5,6),
                               (2,2,5,5),(2,3,5,6),(2,4,5,1),(2,5,5,2)])
        fixture.refresh_manifest()
        source=tmp_path/'source'
        run_pipeline(fixture.raw,source,fixture.manifest,fixture.readme,
                     explicit=('2020-01-04','2020-01-06'),chunk_size=2,dataset_name='synthetic-test')
        inputs_dir=tmp_path/'inputs'
        prepare_inputs(source,'calendar',inputs_dir)
        inputs=ModelInputs(inputs_dir,verify=True)
        assert len(inputs.target_items)==3
        # Two equal-time targets must each see only the strictly earlier first rating.
        assert list(inputs.context_ends[:2])==[1,1]
        assert inputs.item_ids==['1','2','3','4','5']
        assert inputs.query_features.shape==(3,2)
        model_dir=tmp_path/'model'
        result=train(inputs_dir,model_dir,epochs=1,batch_size=3,validation_users=2,threads=1)
        assert np.isfinite(result['epochs_log'][0]['mean_training_loss'])
        published=tmp_path/'published'
        publish(inputs_dir,model_dir,published,collection)
        model_dir=published
        rec=RetrievalRecommender.load(inputs_dir,model_dir)
        assert set(rec(RecommendationRequest('1',['1','2','3']),100))=={'4','5'}
        rec.close()
        report_dir=tmp_path/'report'
        report=evaluate(inputs_dir,model_dir,report_dir,benchmark_users=2)
        assert all(report['robustness'].values())
        assert report['ann_validation_benchmark']['mean_exact_top100_overlap']==1
        rows=[json.loads(line) for line in (report_dir/'predictions.jsonl').read_text().splitlines()]
        scores,_=harness_evaluate(load_evaluation_examples(source/'calendar/test_all.jsonl'),
                                 SavedPredictionsRecommender({r['user_id']:r['ranked_items'] for r in rows}),100,10)
        assert scores['recall@100']==report['metrics']['two_tower/all']['recall_at_100']
        # The database is mutable: verification must detect remote mapping tampering too.
        with connect() as client:
            remote=client.collections.use(collection)
            obj=next(iter(remote.iterator()))
            remote.data.update(uuid=obj.uuid,properties={'movie_id':'tampered'})
        with pytest.raises(ValueError,match='mapping'):
            RetrievalRecommender.load(inputs_dir,model_dir)
        with (model_dir/'weaviate.json').open('ab') as handle:
            handle.write(b'tamper')
        with pytest.raises(ValueError,match='hash mismatch'):
            RetrievalRecommender.load(inputs_dir,model_dir)
        with (inputs_dir/'item_ids.json').open('a') as handle:
            handle.write(' ')
        with pytest.raises(ValueError,match='hash mismatch'):
            ModelInputs(inputs_dir,verify=True)
    finally:
        with connect() as client:
            if client.collections.exists(collection):
                client.collections.delete(collection)
        fixture.doCleanups()


def test_negative_mask_does_not_train_known_positives_as_negatives():
    inputs=SimpleNamespace(pair_users=np.array([0,1,2]),positive_offsets=np.array([0,1,2,3]),
                           context_ends=np.array([1,2,3]),positive_items=np.array([2,0,1]))
    result=negative_mask(inputs,np.array([0,1,2]),np.array([1,2,1]))
    assert not result.diagonal().any()
    assert result[0,1]  # Historical positive.
    assert result[0,2]  # Repeated target.
    assert not result[1,0]


def test_missing_genres_produce_finite_unit_vectors():
    model=TwoTower(np.zeros((3,2),dtype='float32'))
    vectors=model.all_item_vectors()
    assert np.isfinite(vectors).all()
    assert np.allclose(np.linalg.norm(vectors,axis=1),1)


def test_exact_search_exclusions_and_tie_breaks():
    index=ExactIndex(np.array([[1.,0.],[1.,0.],[0.,1.]],dtype='float32'))
    assert index.search_filtered([1.,0.],3,{0})==[1,2]
    assert index.search_filtered([1.,0.],3,{0,1,2})==[]


def test_weaviate_adaptive_overfetch_and_mapping_checks():
    vectors=np.tile(np.array([[1.,0.]],dtype='float32'),(800,1))
    ids=[str(i) for i in range(800)]
    calls=[]
    def near_vector(**kwargs):
        calls.append(kwargs['limit'])
        return SimpleNamespace(objects=[SimpleNamespace(properties={'item_index':i,'movie_id':str(i),'embedding_fingerprint':'test'}) for i in range(kwargs['limit'])])
    collection=SimpleNamespace(query=SimpleNamespace(near_vector=near_vector))
    index=WeaviateIndex(None,collection,ids,vectors,{'fingerprint':'test'})
    assert index.search_filtered([1.,0.],10,set(range(700)))==list(range(700,710))
    assert calls==[510,800]
    ids[0]='wrong-mapping'
    with pytest.raises(ValueError,match='mismatched'):
        index.search_filtered([1.,0.],10,set())
