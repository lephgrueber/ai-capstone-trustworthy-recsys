"""Train on historical prefixes; select checkpoints using dev, never test.

Selection (M3 feedback §1): macro Recall@500 over *every* viewer in a dev example file
(default: the window export dev.jsonl when given, else the run's validation_warm.jsonl).
With ``select=False`` (final refit / deployment fit) no labels are read: the towers train
for exactly ``epochs`` epochs and the last checkpoint is kept.
"""

import argparse
from dataclasses import asdict
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from eval.harness import RecommendationRequest
from trustworthy_recsys.evaluation.metrics import recall_at_k
from trustworthy_recsys.data.retrieval_inputs import ModelInputs
from trustworthy_recsys.data.windows import load_records, read_window_meta
from trustworthy_recsys.data.reporting import hashes, write_json, code_provenance
from .model import HistoryConfig, TwoTower, load_model
from .index import ExactIndex
from .recommender import RetrievalRecommender, request_from_record


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


class PairHistory:
    """Favorite/dislike tensors for training pairs, built only from strictly earlier ratings."""

    def __init__(self,inputs,config):
        if not inputs.has_preference_history:
            raise ValueError('History inputs need model inputs version 1.1+; re-run retrieval_inputs')
        self.inputs,self.config=inputs,config
        users=np.asarray(inputs.pair_users,dtype=np.int64)
        self.end=np.asarray(inputs.context_ends,dtype=np.int64)
        self.start=np.maximum(np.asarray(inputs.positive_offsets)[users],self.end-config.max_history)
        # Global key (user, time) locates each pair's dislikes strictly before its target.
        counts=np.diff(np.asarray(inputs.negative_offsets))
        owner=np.repeat(np.arange(len(counts),dtype=np.int64),counts)
        keys=(owner<<32)+np.asarray(inputs.negative_times,dtype=np.int64)
        self.neg_end=np.searchsorted(keys,(users<<32)+np.asarray(inputs.target_times,dtype=np.int64),side='left')
        self.neg_start=np.maximum(np.asarray(inputs.negative_offsets)[users],self.neg_end-config.max_dislikes)
        self.n_items=len(inputs.item_ids)

    def _gather(self,values,start,end,width,fill):
        idx=start[:,None]+np.arange(width)[None,:]
        mask=idx<end[:,None]
        out=np.where(mask,np.asarray(values)[np.minimum(idx,max(len(values)-1,0))] if len(values) else fill,fill)
        return out,mask

    def batch(self,pairs):
        c,inputs=self.config,self.inputs
        start,end=self.start[pairs],self.end[pairs]
        items,mask=self._gather(inputs.positive_items,start,end,c.max_history,self.n_items)
        stars,_=self._gather(inputs.positive_ratings,start,end,c.max_history,4.0)
        stamps,_=self._gather(inputs.positive_times,start,end,c.max_history,0)
        age=np.maximum(np.asarray(inputs.target_times)[pairs][:,None]-stamps,0)/86400*mask
        dis,dmask=self._gather(inputs.negative_items,self.neg_start[pairs],self.neg_end[pairs],c.max_dislikes,self.n_items)
        return {'items':torch.from_numpy(items.astype(np.int64)),'mask':torch.from_numpy(mask),
                'ratings':torch.from_numpy(stars.astype('float32')),'age_days':torch.from_numpy(age.astype('float32')),
                'dislikes':torch.from_numpy(dis.astype(np.int64)),'dislike_mask':torch.from_numpy(dmask)}

    def latest_dislike(self,pairs):
        """Most recent dislike before each target, or -1."""
        has=self.neg_end[pairs]>np.asarray(self.inputs.negative_offsets)[np.asarray(self.inputs.pair_users)[pairs]]
        return np.where(has,np.asarray(self.inputs.negative_items)[np.maximum(self.neg_end[pairs]-1,0)] if len(self.inputs.negative_items) else -1,-1)


def selection_score(recommender,records,k,as_of=None):
    return float(np.mean([recall_at_k(r['relevant_items'],recommender(request_from_record(r,as_of),k),k) for r in records]))


def train(inputs_dir,output_dir,epochs=4,batch_size=256,learning_rate=.002,seed=42,
          embedding_dim=32,max_pairs=0,validation_users=0,threads=2,selection_examples=None,
          selection_k=500,select=True,temperature=.1,history=None,dislike_penalty=0.0):
    """`validation_users=0` scores every selection viewer; a positive value hash-samples them."""
    if min(epochs,batch_size,threads,embedding_dim,selection_k)<=0 or validation_users<0 or batch_size<2 \
            or learning_rate<=0 or max_pairs<0 or temperature<=0:
        raise ValueError('Invalid training configuration')
    root=Path(output_dir)
    if root.exists():
        raise ValueError('Model directory already exists')
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    inputs=ModelInputs(inputs_dir,verify=True)
    history_config=HistoryConfig(**(history or {}))
    validation,validation_path,as_of=[],None,int(pd.Timestamp(inputs.meta['training_cutoff_utc']).timestamp())
    if select:
        validation_path=Path(selection_examples) if selection_examples else None
        if validation_path is None:
            source=Path(inputs.meta['source_run'])
            validation_path=source/inputs.meta['scenario']/'validation_warm.jsonl'
            source_manifest=json.loads((source/'manifest.json').read_text())
            relative=validation_path.relative_to(source).as_posix()
            if hashes(validation_path)['sha256']!=source_manifest['outputs'][relative]['sha256']:
                raise ValueError('Validation input hash mismatch')
        else:
            window=read_window_meta(validation_path)
            cutoff=int(pd.Timestamp(inputs.meta['training_cutoff_utc']).timestamp())
            if window is not None and window['label_start']<cutoff:
                raise ValueError('Selection labels overlap the towers\' fitting window')
            if window is not None:
                as_of=window['history_end']
        validation=(sampled_examples(validation_path,validation_users,seed) if validation_users
                    else list(load_records(validation_path)))
        if not validation:
            raise ValueError('No dev examples available for checkpoint selection')
    n=len(inputs.target_items)
    rng=np.random.default_rng(seed)
    selected=np.sort(rng.choice(n,size=min(max_pairs,n),replace=False)) if max_pairs else np.arange(n)
    if len(selected)<2:
        raise ValueError('Need at least two training pairs')
    frequencies=np.bincount(inputs.target_items[selected],minlength=len(inputs.item_ids)).astype('float32')
    log_q=torch.from_numpy(np.log(np.maximum(frequencies,1)/len(selected)))
    model=TwoTower(inputs.item_features,embedding_dim=embedding_dim,history=history_config)
    pair_history=PairHistory(inputs,history_config) if history_config.active or dislike_penalty else None
    dense=torch.optim.AdamW(model.dense_parameters(),lr=learning_rate,weight_decay=1e-4)
    sparse=torch.optim.SparseAdam(model.sparse_parameters(),lr=learning_rate)
    root.mkdir(parents=True)
    (root/'INCOMPLETE').write_text('Training/embedding export in progress.\n')
    np.save(root/'training_pair_indices.npy',selected)
    if validation_users:
        write_json(root/'validation_user_ids.json',[record['user_id'] for record in validation])
    config={'architecture':model.config,'epochs':epochs,'batch_size':batch_size,'learning_rate':learning_rate,
            'seed':seed,'threads':threads,'device':'cpu','temperature':temperature,'log_q_correction':True,
            'available_training_pairs':n,'selected_training_pairs':len(selected),'max_pairs':max_pairs,
            'validation_population':str(validation_path) if select else None,
            'validation_users':len(validation),'validation_sampled':bool(validation_users),
            'selection_metric':f'macro Recall@{selection_k} over every selection viewer' if select and not validation_users
                else f'macro Recall@{selection_k} on deterministic hash sample' if select
                else 'none: fixed epochs, last checkpoint (labels never read)',
            'input_manifest_sha256':hashes(Path(inputs_dir)/'manifest.json')['sha256'],
            'validation_sha256':hashes(validation_path)['sha256'] if select else None,'torch_version':torch.__version__,
            'code':code_provenance(),'loss':'in-batch sampled softmax, empirical target-frequency logQ; historical positives and duplicate targets masked'
                    +(f'; + {dislike_penalty} x softplus((s(latest earlier dislike) - s(target)) / T)' if dislike_penalty else ''),
            'history':asdict(history_config),'dislike_penalty':dislike_penalty}
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
            user_vectors=model.encode_user(features,pair_history.batch(batch) if history_config.active else None)
            target_vectors=model.encode_item(target_tensor)
            logits=user_vectors @ target_vectors.T / temperature
            logits=logits-log_q[target_tensor][None,:]
            logits=logits.masked_fill(torch.from_numpy(negative_mask(inputs,batch,target)),float('-inf'))
            loss=F.cross_entropy(logits,torch.arange(len(batch)))
            if dislike_penalty:
                # Penalize a liked target scoring below the viewer's latest earlier dislike.
                disliked=pair_history.latest_dislike(batch)
                rows=np.flatnonzero(disliked>=0)
                if len(rows):
                    rows_t=torch.from_numpy(rows)
                    bad=(user_vectors[rows_t]*model.encode_item(torch.from_numpy(disliked[rows].astype(np.int64)))).sum(-1)
                    good=(user_vectors[rows_t]*target_vectors[rows_t]).sum(-1)
                    loss=loss+dislike_penalty*F.softplus((bad-good)/temperature).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            dense.step()
            sparse.step()
            loss_sum+=float(loss.detach())*len(batch)
            trained+=len(batch)
        train_seconds=time.perf_counter()-epoch_started
        score=None
        if select:
            recommender=RetrievalRecommender(inputs,model,ExactIndex(model.all_item_vectors()))
            score=selection_score(recommender,validation,selection_k,as_of)
        result={'epoch':epoch+1,'mean_training_loss':loss_sum/trained,'pairs_seen':trained,
                f'validation_recall_at_{selection_k}':score,'seconds':time.perf_counter()-epoch_started,
                'training_seconds':train_seconds,'updates':-(-len(selected)//batch_size)*(epoch+1)}
        records.append(result)
        print(json.dumps(result),flush=True)
        if not select or score>best:
            best=score if select else best
            config['selected_epoch']=epoch+1
            torch.save({'config':model.config,'state_dict':model.state_dict()},root/'model.pt')
    model=load_model(root/'model.pt',inputs.item_features)
    vectors=model.all_item_vectors()
    np.save(root/'item_embeddings.npy',vectors)
    config.update({'training_seconds':time.perf_counter()-started,f'best_validation_recall_at_{selection_k}':best if select else None,
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
    p.add_argument('--validation-users',type=int,default=0,help='0 scores every selection viewer')
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--selection-examples',type=Path,help='Dev window export, e.g. <windows>/dev.jsonl')
    p.add_argument('--selection-k',type=int,default=500)
    p.add_argument('--no-selection',action='store_true',help='Final refit: train fixed epochs, keep the last')
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--history',type=json.loads,default=None,
                   help='Viewer-tower history JSON, e.g. \'{"favorites": true, "pooling": "recency", "dislikes": true}\'')
    p.add_argument('--dislike-penalty',type=float,default=0.0)
    a=p.parse_args()
    train(a.inputs_dir,a.output_dir,a.epochs,a.batch_size,a.learning_rate,a.seed,a.embedding_dim,a.max_pairs,
          a.validation_users,a.threads,a.selection_examples,a.selection_k,not a.no_selection,a.temperature,
          a.history,a.dislike_penalty)


if __name__=='__main__':
    main()
