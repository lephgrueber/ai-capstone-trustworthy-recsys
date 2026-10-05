"""Evaluate the frozen retrieval model with the existing harness metrics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import weaviate
import numpy as np
import torch

from eval.harness import RecommendationRequest
from trustworthy_recsys.evaluation.metrics import recall_at_k, ndcg_at_k
from trustworthy_recsys.data.reporting import hashes, write_json, code_provenance
from .recommender import RetrievalRecommender
from .index import filtered_search, ExactIndex
from .train import sampled_examples
from .model_card import render


def metric_row(labels, predictions):
    return [recall_at_k(labels,predictions,10),recall_at_k(labels,predictions,100),ndcg_at_k(labels,predictions,10)]


def percentiles(values):
    return dict(zip(['p50','p95','p99'],map(float,np.percentile(values,[50,95,99])))) if values else {}


def evaluate(inputs_dir,model_dir,output_dir,benchmark_users=200):
    out=Path(output_dir)
    if out.exists():
        raise ValueError('Choose a new report directory')
    if benchmark_users<=0:
        raise ValueError('benchmark_users must be positive')
    torch.set_num_threads(2)
    with RetrievalRecommender.load(inputs_dir,model_dir) as rec:
        return evaluate_loaded(rec,model_dir,out,benchmark_users)


def evaluate_loaded(rec,model_dir,out,benchmark_users):
    data=rec.inputs
    source=Path(data.meta['source_run'])
    if hashes(source/'manifest.json')['sha256']!=data.meta['source_manifest_sha256']:
        raise ValueError('Source manifest changed since model-input preparation')
    scenario=data.meta['scenario']
    manifest=json.loads((source/'manifest.json').read_text())
    paths={name:source/scenario/f'{name}.jsonl' for name in ('test_all','validation_warm')}
    for path in paths.values():
        if hashes(path)['sha256']!=manifest['outputs'][path.relative_to(source).as_posix()]['sha256']:
            raise ValueError(f'Evaluation source hash mismatch: {path.name}')
    out.mkdir(parents=True)
    (out/'INCOMPLETE').write_text('Evaluation in progress\n')
    totals=defaultdict(list)
    latency=defaultdict(list)
    misses=defaultdict(int)
    errors=[]
    catalog=set(data.item_ids)
    head=set(data.item_ids[i] for i in rec.popularity[:max(1,len(catalog)//10)])
    coverage=set()
    ceilings=[]
    users=set()
    with paths['test_all'].open(encoding='utf-8') as handle, (out/'predictions.jsonl').open('w',encoding='utf-8') as predictions:
        for line in handle:
            row=json.loads(line)
            if row['user_id'] in users:
                raise ValueError('Duplicate evaluation user')
            users.add(row['user_id'])
            request=RecommendationRequest(row['user_id'],row['history_items'])
            labels=set(row['relevant_items'])
            if not labels or labels.intersection(request.history_items):
                raise ValueError('Empty labels or history/label leakage')
            started=time.perf_counter()
            indices,fallback=rec.recommend_indices(request,100)
            latency['fallback' if fallback else 'learned'].append((time.perf_counter()-started)*1000)
            ranked=[data.item_ids[i] for i in indices]
            excluded=rec.excluded(request)
            if len(indices)!=len(set(indices)) or set(indices)&excluded or not set(ranked)<=catalog:
                raise ValueError('Invalid, duplicate, or seen predictions')
            popular=[data.item_ids[i] for i in rec.popularity_indices(excluded,100)]
            slices={'all':labels,'fallback' if fallback else 'learned':labels}
            warm=labels&catalog
            if request.user_id in data.user_map and warm:
                slices['warm']=warm
            size=len(rec.history_indices(request.history_items))
            slices['history_0' if size==0 else 'history_1_20' if size<=20 else 'history_21_plus']=labels
            for name, subset in [('head_labels',labels&head),('tail_labels',(labels&catalog)-head)]:
                if subset:
                    slices[name]=subset
            for name,subset in slices.items():
                totals[f'two_tower/{name}'].append(metric_row(subset,ranked))
                totals[f'popularity/{name}'].append(metric_row(subset,popular))
            coverage.update(ranked)
            ceilings.append(len(warm)/len(labels))
            missed=labels-set(ranked)
            misses['outside_training_catalog']+=len(missed-catalog)
            misses['in_catalog_not_retrieved']+=len(missed&catalog)
            misses['relevant_labels']+=len(labels)
            misses['hits_at_100']+=len(labels&set(ranked))
            error={'user_id':request.user_id,'history_length':size,'fallback':fallback,
                   'recall_at_100':len(labels&set(ranked))/len(labels),
                   'relevant_count':len(labels),'unretrievable_count':len(labels-catalog),
                   'missed_item_ids':sorted(missed)[:20],'recommended_item_ids':ranked[:10]}
            errors.append(error)
            if len(errors)>40:
                errors=sorted(errors,key=lambda x:(x['recall_at_100'],-x['relevant_count'],x['user_id']))[:20]
            predictions.write(json.dumps({'user_id':request.user_id,'ranked_items':ranked})+'\n')
    metrics={name:{'users':len(rows),**dict(zip(['recall_at_10','recall_at_100','ndcg_at_10'],map(float,np.mean(rows,axis=0))))} for name,rows in totals.items()}
    # Approximation quality is measured on validation, independently of held-out relevance.
    vectors=np.load(Path(model_dir)/'item_embeddings.npy')
    exact=ExactIndex(vectors)
    examples=sampled_examples(paths['validation_warm'],benchmark_users,42)
    overlaps=[]
    embedding_ms=[]
    ann_ms=[]
    exact_ms=[]
    perturbation=defaultdict(list)
    for row in examples[:10]:
        rec(RecommendationRequest(row['user_id'],row['history_items']),100)
    for row in examples:
        request=RecommendationRequest(row['user_id'],row['history_items'])
        started=time.perf_counter()
        query=rec.query_vector(request.history_items)
        embedding_ms.append((time.perf_counter()-started)*1000)
        if query is None:
            continue
        excluded=rec.excluded(request)
        started=time.perf_counter()
        approximate=filtered_search(rec.index,query,100,excluded)
        ann_ms.append((time.perf_counter()-started)*1000)
        started=time.perf_counter()
        reference=filtered_search(exact,query,100,excluded)
        exact_ms.append((time.perf_counter()-started)*1000)
        overlaps.append(len(set(approximate)&set(reference))/len(reference))
        perturbation['original_history_recall_at_100'].append(recall_at_k(row['relevant_items'],[data.item_ids[i] for i in approximate],100))
        shortened=RecommendationRequest(request.user_id,request.history_items[::2])
        perturbation['half_history_recall_at_100'].append(recall_at_k(row['relevant_items'],rec(shortened,100),100))
    # Synthetic perturbations exercise serving behavior, not predictive robustness claims.
    history=[data.item_ids[0],data.item_ids[-1]]
    probe=RecommendationRequest('__unknown_user__',history)
    base=rec(probe,10)
    robustness={
        'unknown_user_with_known_history':len(base)==min(10,len(catalog)-len(set(history))),
        'unknown_items_ignored':base==rec(RecommendationRequest('__unknown_user__',history+['__unknown_item__']),10),
        'duplicate_history_invariant':base==rec(RecommendationRequest('__unknown_user__',history*2),10),
        'history_order_invariant':np.allclose(rec.query_vector(history),rec.query_vector(history[::-1])),
        'empty_history_popularity_fallback':rec.recommend_indices(RecommendationRequest('__unknown_user__',[]),10)[1],
        'all_unknown_history_fallback':rec.recommend_indices(RecommendationRequest('__unknown_user__',['__unknown_item__']),10)[1],
        'long_duplicate_history_invariant':base==rec(RecommendationRequest('__unknown_user__',history*1000),10),
        'saved_embeddings_match_model':bool(np.allclose(vectors,rec.model.all_item_vectors(),atol=1e-6)),
    }
    # Audit every prepared pair independently of model training.
    for pair in range(len(data.target_items)):
        user=int(data.pair_users[pair])
        lo=int(data.positive_offsets[user]); end=int(data.context_ends[pair])
        if not lo<end<=int(data.positive_offsets[user+1]):
            raise ValueError('Invalid training context offsets')
        if data.positive_times[end-1]>=data.target_times[pair] or data.target_items[pair] in data.positive_items[lo:end]:
            raise ValueError('Training target/context leakage')
    training=json.loads((Path(model_dir)/'training.json').read_text())
    report={'metrics':metrics,'error_analysis':dict(misses),'code':code_provenance(),
            'validation_history_ablation':{'users':len(overlaps),**{name:float(np.mean(values)) for name,values in perturbation.items()}},
            'catalog_coverage_at_100':len(coverage)/len(catalog),'all_population_mean_catalog_recall_ceiling':float(np.mean(ceilings)),
            'latency_ms':{name:percentiles(values) for name,values in latency.items()},
            'ann_validation_benchmark':{'users':len(overlaps),'mean_exact_top100_overlap':float(np.mean(overlaps)) if overlaps else None,
                                        'embedding_ms':percentiles(embedding_ms),'ann_search_filter_ms':percentiles(ann_ms),'exact_search_filter_ms':percentiles(exact_ms)},
            'robustness':robustness,'integrity':{'input_and_model_hashes_verified':True,'evaluation_hashes_verified':True,
                'remote_movie_ids_and_vectors_verified':True,
                'training_pairs_audited':len(data.target_items),'strict_prefix_and_target_exclusion':True,
                'all_predictions_unique_in_catalog_and_unseen':True,'test_used_for_checkpoint_selection':False},
            'protocol':{'scenario':scenario,'evaluation':'full test_all; warm restricted exactly as pipeline warm definition',
                        'baseline':'positive training-count popularity with identical catalog and all-training-rated seen filtering',
                        'metric_implementation':'trustworthy_recsys.evaluation.metrics; macro user means',
                        'timing':'client CPU, 2 PyTorch threads; Weaviate RPC/network included, model loading and full remote audit excluded; validation benchmark after 10 warmups',
                        'platform':platform.platform(),'torch':torch.__version__,'weaviate_client':weaviate.__version__,
                        'weaviate':rec.index.descriptor,
                        'test_sha256':hashes(paths['test_all'])['sha256'],'model_manifest_sha256':hashes(Path(model_dir)/'manifest.json')['sha256']},
            'training':training}
    write_json(out/'report.json',report)
    write_json(out/'error_cases.json',sorted(errors,key=lambda x:(x['recall_at_100'],-x['relevant_count'],x['user_id']))[:20])
    lines=['# Retrieval evaluation report','','Metrics use the existing harness implementation. Every eligible test user is included.','',
           '| Method / population | Users | Recall@10 | Recall@100 | NDCG@10 |','|---|---:|---:|---:|---:|']
    for name,row in metrics.items():
        lines.append(f"| {name} | {row['users']} | {row['recall_at_10']:.4f} | {row['recall_at_100']:.4f} | {row['ndcg_at_10']:.4f} |")
    lines+=['','ANN agreement with exact filtered top-100 on validation: '+str(report['ann_validation_benchmark']['mean_exact_top100_overlap']),
            '', 'Serving latency in milliseconds (p50 / p95 / p99): '+json.dumps(report['latency_ms']),
            '', 'Missed-label counts: '+json.dumps(report['error_analysis']),
            '', 'Integrity: verified input/model/evaluation hashes, every training prefix, and every prediction. Test labels were used only for reporting.',
            '', 'Robustness checks: '+json.dumps(robustness),
            '', 'Validation history ablation (every second history item retained; full seen filtering preserved): '+json.dumps(report['validation_history_ablation']),
            '', 'The two-tower user representation averages genres and loses item-level order. Empty histories use popularity; out-of-catalog positives cannot be retrieved. These are offline explicit-rating results, not causal, online, or demographic-fairness evidence.',
            '', 'See report.json for timings, provenance, model selection, and catalog coverage; error_cases.json contains 20 lowest-recall cases (ties prefer more labels).']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'MODEL_CARD.md').write_text(render(report),encoding='utf-8')
    write_json(out/'manifest.json',{'outputs':{p.name:hashes(p)['sha256'] for p in out.iterdir() if p.is_file() and p.name!='INCOMPLETE'}})
    (out/'INCOMPLETE').unlink()
    print(json.dumps({'metrics':{k:v for k,v in metrics.items() if k.endswith(('/all','/warm','/learned'))},'ann':report['ann_validation_benchmark'],'robustness':robustness}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir',type=Path,required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--benchmark-users',type=int,default=200)
    args=parser.parse_args()
    evaluate(args.inputs_dir,args.model_dir,args.output_dir,args.benchmark_users)


if __name__=='__main__':
    main()
