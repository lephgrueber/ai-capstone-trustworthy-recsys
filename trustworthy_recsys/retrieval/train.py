"""Train on historical prefixes; select checkpoints using validation, never test."""

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from eval.harness import RecommendationRequest
from trustworthy_recsys.evaluation.metrics import recall_at_k
from trustworthy_recsys.data.retrieval_inputs import ModelInputs
from trustworthy_recsys.data.reporting import hashes, write_json, code_provenance
from .model import TwoTower, load_model
from .index import ExactIndex
from .recommender import RetrievalRecommender


def sampled_examples(path,limit,seed):
    """Deterministic uniform hash sample of users; labels never influence sampling."""
    heap=[]
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            record=json.loads(line)
            key=int.from_bytes(hashlib.sha256(f"{seed}:{record['user_id']}".encode()).digest()[:8],'big')
            entry=(-key,record['user_id'],record)
            if len(heap)<limit:
                heapq.heappush(heap,entry)
            elif key < -heap[0][0]:
                heapq.heapreplace(heap,entry)
    return [row[2] for row in sorted(heap,key=lambda row:(-row[0],row[1]))]


def negative_mask(inputs,pair_indices,targets):
    """Mask repeated targets and known historical positives from in-batch negatives."""
    mask=targets[:,None]==targets[None,:]
    for row,pair in enumerate(pair_indices):
        user=int(inputs.pair_users[pair])
        start=int(inputs.positive_offsets[user])
        end=int(inputs.context_ends[pair])
        mask[row]|=np.isin(targets,inputs.positive_items[start:end])
    np.fill_diagonal(mask,False)
    return mask


def train(inputs_dir,output_dir,epochs=4,batch_size=256,learning_rate=.002,seed=42,
          embedding_dim=32,max_pairs=0,validation_users=512,threads=2):
    if min(epochs,batch_size,validation_users,threads,embedding_dim)<=0 or batch_size<2 or learning_rate<=0 or max_pairs<0:
        raise ValueError('Invalid training configuration')
    root=Path(output_dir)
    if root.exists():
        raise ValueError('Model directory already exists')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    inputs=ModelInputs(inputs_dir,verify=True)
    source=Path(inputs.meta['source_run'])
    validation_path=source/inputs.meta['scenario']/'validation_warm.jsonl'
    source_manifest=json.loads((source/'manifest.json').read_text())
    relative=validation_path.relative_to(source).as_posix()
    if hashes(validation_path)['sha256']!=source_manifest['outputs'][relative]['sha256']:
        raise ValueError('Validation input hash mismatch')
    validation=sampled_examples(validation_path,validation_users,seed)
    if not validation:
        raise ValueError('No warm validation examples available for checkpoint selection')
    n=len(inputs.target_items)
    rng=np.random.default_rng(seed)
    selected=np.sort(rng.choice(n,size=min(max_pairs,n),replace=False)) if max_pairs else np.arange(n)
    if len(selected)<2:
        raise ValueError('Need at least two training pairs')
    frequencies=np.bincount(inputs.target_items[selected],minlength=len(inputs.item_ids)).astype('float32')
    log_q=torch.from_numpy(np.log(np.maximum(frequencies,1)/len(selected)))
    model=TwoTower(inputs.item_features,embedding_dim=embedding_dim)
    dense=torch.optim.AdamW([p for name,p in model.named_parameters() if name!='identity.weight'],lr=learning_rate,weight_decay=1e-4)
    sparse=torch.optim.SparseAdam([model.identity.weight],lr=learning_rate)
    root.mkdir(parents=True)
    (root/'INCOMPLETE').write_text('Training/embedding export in progress.\n')
    np.save(root/'training_pair_indices.npy',selected)
    write_json(root/'validation_user_ids.json',[record['user_id'] for record in validation])
    config={'architecture':model.config,'epochs':epochs,'batch_size':batch_size,'learning_rate':learning_rate,
            'seed':seed,'threads':threads,'device':'cpu','temperature':.1,'log_q_correction':True,
            'available_training_pairs':n,'selected_training_pairs':len(selected),'max_pairs':max_pairs,
            'validation_population':'validation_warm','validation_users':len(validation),
            'selection_metric':'macro Recall@100 on deterministic validation user sample',
            'input_manifest_sha256':hashes(Path(inputs_dir)/'manifest.json')['sha256'],
            'validation_sha256':hashes(validation_path)['sha256'],'torch_version':torch.__version__,
            'code':code_provenance(),'loss':'in-batch sampled softmax, empirical target-frequency logQ; historical positives and duplicate targets masked'}
    best=-1.0
    records=[]
    started=time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order=rng.permutation(selected)
        loss_sum=0.0
        trained=0
        epoch_started=time.perf_counter()
        for start in range(0,len(order),batch_size):
            batch=order[start:start+batch_size]
            if len(batch)<2:
                continue
            target=np.array(inputs.target_items[batch],dtype=np.int64)
            features=torch.from_numpy(np.array(inputs.query_features[batch]))
            target_tensor=torch.from_numpy(target)
            dense.zero_grad(set_to_none=True)
            sparse.zero_grad(set_to_none=True)
            logits=model.encode_user(features) @ model.encode_item(target_tensor).T / .1
            logits=logits-log_q[target_tensor][None,:]
            logits=logits.masked_fill(torch.from_numpy(negative_mask(inputs,batch,target)),float('-inf'))
            loss=F.cross_entropy(logits,torch.arange(len(batch)))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            dense.step()
            sparse.step()
            loss_sum+=float(loss.detach())*len(batch)
            trained+=len(batch)
        vectors=model.all_item_vectors()
        index=ExactIndex(vectors)
        recommender=RetrievalRecommender(inputs,model,index)
        score=float(np.mean([recall_at_k(record['relevant_items'],recommender(RecommendationRequest(record['user_id'],record['history_items']),100),100) for record in validation]))
        result={'epoch':epoch+1,'mean_training_loss':loss_sum/trained,'pairs_seen':trained,
                'validation_recall_at_100':score,'seconds':time.perf_counter()-epoch_started}
        records.append(result)
        print(json.dumps(result),flush=True)
        if score>best:
            best=score
            config['selected_epoch']=epoch+1
            torch.save({'config':model.config,'state_dict':model.state_dict()},root/'model.pt')
    model=load_model(root/'model.pt',inputs.item_features)
    vectors=model.all_item_vectors()
    np.save(root/'item_embeddings.npy',vectors)
    config.update({'training_seconds':time.perf_counter()-started,'best_validation_recall_at_100':best,
                   'epochs_log':records,'index':{'kind':'not yet published; run retrieval.publish'},
                   'checkpoint_search':'exact NumPy cosine; production retrieval uses Weaviate'})
    write_json(root/'training.json',config)
    write_json(root/'manifest.json',{'outputs':{p.name:hashes(p)['sha256'] for p in root.iterdir() if p.is_file() and p.name!='INCOMPLETE'}})
    (root/'INCOMPLETE').unlink()
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--learning-rate',type=float,default=.002)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--embedding-dim',type=int,default=32)
    p.add_argument('--max-pairs',type=int,default=0,help='0 trains on every prepared pair')
    p.add_argument('--validation-users',type=int,default=512)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args()
    train(a.inputs_dir,a.output_dir,a.epochs,a.batch_size,a.learning_rate,a.seed,a.embedding_dim,a.max_pairs,a.validation_users,a.threads)


if __name__=='__main__':
    main()
