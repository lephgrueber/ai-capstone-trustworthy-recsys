"""Harness-compatible retrieval and deterministic cold-history fallback."""

from pathlib import Path

import numpy as np
import torch

from trustworthy_recsys.data.retrieval_inputs import ModelInputs, read_json
from trustworthy_recsys.data.reporting import hashes
from .model import load_model
from .index import filtered_search
from .weaviate_store import WeaviateIndex


class RetrievalRecommender:
    def __init__(self,inputs,model,index):
        self.inputs,self.model,self.index=inputs,model,index
        self.model.eval()
        self.popularity=np.lexsort((np.arange(len(inputs.item_ids)),-inputs.positive_counts))

    @classmethod
    def load(cls,inputs_dir,model_dir):
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
        model=load_model(root/'model.pt',inputs.item_features)
        index=WeaviateIndex.load(root/'weaviate.json',inputs.item_ids,np.load(root/'item_embeddings.npy'))
        if index.ntotal!=len(inputs.item_ids):
            raise ValueError('Index/catalog size mismatch')
        return cls(inputs,model,index)

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

    @torch.inference_mode()
    def query_vector(self,history):
        profile=self.query_profile(history)
        return None if profile is None else self.model.encode_user(torch.from_numpy(profile).unsqueeze(0)).numpy()[0]

    def recommend_indices(self,request,k):
        if k<=0:
            raise ValueError('k must be positive')
        excluded=self.excluded(request)
        query=self.query_vector(request.history_items)
        if query is None:
            return self.popularity_indices(excluded,k),True
        return filtered_search(self.index,query,k,excluded),False

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
