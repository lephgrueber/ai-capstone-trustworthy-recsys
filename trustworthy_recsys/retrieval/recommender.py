"""Harness-compatible retrieval and deterministic cold-history fallback."""

from pathlib import Path

import math
import time

import numpy as np
import torch

from trustworthy_recsys.data.retrieval_inputs import ModelInputs, read_json
from trustworthy_recsys.data.reporting import hashes
from eval.harness import RecommendationRequest
from .model import history_batch, load_model
from .index import filtered_search
from .weaviate_store import WeaviateIndex


class RetrievalRecommender:
    def __init__(self,inputs,model,index,item_vectors=None):
        self.inputs,self.model,self.index=inputs,model,index
        # Saved, verified embeddings let callers recover scores; Weaviate search returns order only.
        self.item_vectors=item_vectors if item_vectors is not None else getattr(index,'vectors',None)
        self.model.eval()
        self.popularity=np.lexsort((np.arange(len(inputs.item_ids)),-inputs.positive_counts))

    @staticmethod
    def load_verified(inputs_dir,model_dir):
        """Return verified inputs, model and saved item vectors; no index is opened."""
        inputs=ModelInputs(inputs_dir,verify=True)
        root=Path(model_dir)
        config=read_json(root/'training.json')
        if config['input_manifest_sha256']!=hashes(Path(inputs_dir)/'manifest.json')['sha256']:
            raise ValueError('Model was trained with different inputs')
        if (root/'INCOMPLETE').exists():
            raise ValueError('Incomplete model artifacts')
        for name,digest in read_json(root/'manifest.json')['outputs'].items():
            if hashes(root/name)['sha256']!=digest:
                raise ValueError(f'Model artifact hash mismatch: {name}')
        return inputs,load_model(root/'model.pt',inputs.item_features,inputs.item_ids),np.load(root/'item_embeddings.npy')

    @classmethod
    def load(cls,inputs_dir,model_dir):
        inputs,model,vectors=cls.load_verified(inputs_dir,model_dir)
        index=WeaviateIndex.load(Path(model_dir)/'weaviate.json',inputs.item_ids,vectors)
        if index.ntotal!=len(inputs.item_ids):
            raise ValueError('Index/catalog size mismatch')
        return cls(inputs,model,index,vectors)

    def close(self):
        if hasattr(self.index, 'close'):
            self.index.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def history_indices(self,history):
        # Deduplicate while preserving order; unknown IDs cannot become fitted features.
        return list(dict.fromkeys(self.inputs.item_map[x] for x in history if x in self.inputs.item_map))

    def excluded(self,request):
        seen={self.inputs.item_map[x] for x in request.history_items if x in self.inputs.item_map}
        seen.update(self.inputs.item_map[x] for x in getattr(request,'exclude_items',()) if x in self.inputs.item_map)
        user=self.inputs.user_map.get(request.user_id)
        if user is not None:
            lo,hi=self.inputs.seen_offsets[user:user+2]
            seen.update(map(int,self.inputs.seen_items[lo:hi]))
        return seen

    def query_profile(self,history):
        indices=self.history_indices(history)
        if not indices:
            return None
        profile=np.concatenate([self.inputs.item_features[indices].mean(axis=0),[np.log1p(len(indices))/10]]).astype('float32')
        return profile

    def history_context(self,request):
        """Known favorites (deduplicated, chronological) with aligned rating/time, plus dislikes."""
        ratings=getattr(request,'history_ratings',()) or ()
        times=getattr(request,'history_times',()) or ()
        seen,indices,stars,stamps=set(),[],[],[]
        for position,item in enumerate(request.history_items):
            index=self.inputs.item_map.get(item)
            if index is None or index in seen:
                continue
            seen.add(index)
            indices.append(index)
            stars.append(_number(ratings,position))
            stamps.append(_number(times,position))
        dislikes=list(dict.fromkeys(self.inputs.item_map[x] for x in getattr(request,'disliked_items',()) or ()
                                    if x in self.inputs.item_map))
        as_of=getattr(request,'as_of',None)
        return indices,stars,stamps,dislikes,time.time() if as_of is None else as_of

    @torch.inference_mode()
    def query_vector(self,history,request=None):
        profile=self.query_profile(history)
        if profile is None:
            return None
        batch=None
        if self.model.history.active:
            request=request or RecommendationRequest('',list(history))
            indices,stars,stamps,dislikes,as_of=self.history_context(request)
            batch=history_batch(self.model.history,len(self.inputs.item_ids),[indices],[stars],[stamps],[as_of],[dislikes])
        return self.model.encode_user(torch.from_numpy(profile).unsqueeze(0),batch).numpy()[0]

    def recommend_indices(self,request,k):
        if k<=0:
            raise ValueError('k must be positive')
        excluded=self.excluded(request)
        query=self.query_vector(request.history_items,request)
        if query is None:
            return self.popularity_indices(excluded,k),True
        return filtered_search(self.index,query,k,excluded),False

    def recommend_scored(self,request,k):
        """Like recommend_indices, plus cosine scores; scores are None on popularity fallback."""
        if k<=0:
            raise ValueError('k must be positive')
        excluded=self.excluded(request)
        query=self.query_vector(request.history_items,request)
        if query is None:
            return self.popularity_indices(excluded,k),None,True
        if self.item_vectors is None:
            raise ValueError('Scores require the saved item embeddings')
        indices=filtered_search(self.index,query,k,excluded)
        return indices,(self.item_vectors[indices]@query).astype(float).tolist(),False

    def popularity_indices(self,excluded,k):
        result=[]
        for item in self.popularity:
            if int(item) not in excluded:
                result.append(int(item))
                if len(result)==k:
                    break
        return result

    def __call__(self,request,k):
        indices,_=self.recommend_indices(request,k)
        return [self.inputs.item_ids[i] for i in indices]


def _number(values,position):
    value=values[position] if position<len(values) else None
    return math.nan if value is None else float(value)


def request_from_record(record,as_of=None,dislike_threshold=2.0):
    """Request for a window-export record: inputs only, never its label fields."""
    other=record.get('other_rated_items',())
    other_ratings=record.get('other_rated_ratings',())
    return RecommendationRequest(
        record['user_id'],list(record['history_items']),tuple(other),
        tuple(record.get('history_ratings',())),tuple(record.get('history_times',())),
        tuple(item for item,rating in zip(other,other_ratings) if rating<=dislike_threshold),as_of)
