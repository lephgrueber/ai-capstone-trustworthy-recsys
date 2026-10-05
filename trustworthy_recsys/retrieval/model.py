"""Separate trainable user and item towers with normalized dot-product scores."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TwoTower(nn.Module):
    def __init__(self, item_features, embedding_dim=32, hidden_dim=64, identity_dim=16):
        super().__init__()
        features = torch.as_tensor(np.array(item_features,copy=True),dtype=torch.float32)
        self.register_buffer('item_features',features)
        self.identity = nn.Embedding(len(features),identity_dim,sparse=True)
        nn.init.normal_(self.identity.weight,std=.05)
        self.user_tower = nn.Sequential(nn.Linear(features.shape[1]+1,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,embedding_dim))
        self.item_tower = nn.Sequential(nn.Linear(features.shape[1]+identity_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,embedding_dim))
        self.config = dict(embedding_dim=embedding_dim,hidden_dim=hidden_dim,identity_dim=identity_dim)

    def encode_user(self, profiles):
        return F.normalize(self.user_tower(profiles),dim=-1)

    def encode_item(self, item_ids):
        return F.normalize(self.item_tower(torch.cat([self.identity(item_ids),self.item_features[item_ids]],dim=-1)),dim=-1)

    @torch.inference_mode()
    def all_item_vectors(self,batch_size=4096):
        self.eval()
        return np.concatenate([self.encode_item(torch.arange(start,min(start+batch_size,len(self.item_features)))).numpy()
                               for start in range(0,len(self.item_features),batch_size)]).astype('float32')


def load_model(path,features):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=TwoTower(features,**saved['config'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model
