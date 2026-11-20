"""Separate trainable user and item towers with normalized dot-product scores.

Viewer-tower inputs (M3 feedback §2). The M3 baseline sees only the averaged genre
weights of earlier favorites plus log1p(N)/10, so two favorites with identical genres
are indistinguishable. Optional history inputs add:

- ``favorites``: a pooled vector of *learned favorite-movie embeddings* over the most
  recent ``max_history`` favorites, weighted by ``pooling``:
    * ``mean``     w_i = 1
    * ``rating``   w_i = rating_i - 3.5          (4 -> 0.5, 4.5 -> 1, 5 -> 1.5)
    * ``recency``  w_i = 0.5 ** (age_days_i / half_life_days)
    * ``learned``  w_i = softmax_i(a * (rating_i - 4) + b * log1p(age_days_i)), a, b learned
  Weights are normalized to sum to one. Age is measured from the prediction time (target
  rating time in training, request time when serving). A favorite without a known date
  has age 0; without a known rating it counts as 4.0 (an explicit like).
- ``dislikes``: a separate mean of learned disliked-movie embeddings, the mean genre
  weights of those movies and a 0/1 flag for whether any exist.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SECONDS_PER_DAY = 86_400
POOLINGS = ("mean", "rating", "recency", "learned")


@dataclass(frozen=True)
class HistoryConfig:
    favorites: bool = False
    pooling: str = "mean"
    dislikes: bool = False
    max_history: int = 50
    max_dislikes: int = 50
    half_life_days: float = 365.0
    history_dim: int = 16

    def __post_init__(self):
        if self.pooling not in POOLINGS:
            raise ValueError(f"pooling must be one of {POOLINGS}")
        if min(self.max_history, self.max_dislikes, self.history_dim) <= 0 or self.half_life_days <= 0:
            raise ValueError("history sizes and half-life must be positive")

    @property
    def active(self):
        return self.favorites or self.dislikes


def pad_lists(lists, width, fill):
    """Right-aligned to the most recent `width` entries; returns (array, mask)."""
    out = np.full((len(lists), width), fill)
    mask = np.zeros((len(lists), width), dtype=bool)
    for row, values in enumerate(lists):
        values = np.asarray(values)[-width:]
        out[row, :len(values)] = values
        mask[row, :len(values)] = True
    return out, mask


def history_batch(config: HistoryConfig, n_items, favorites, ratings, times, as_of, dislikes):
    """Tensors for the viewer tower from per-viewer lists of catalog indices.

    favorites/ratings/times: chronological lists per viewer (ratings/times may hold NaN
    for unknown values); as_of: prediction time (Unix seconds) per viewer; dislikes:
    chronological disliked catalog indices before as_of.
    """
    items, mask = pad_lists(favorites, config.max_history, n_items)
    stars, _ = pad_lists(ratings, config.max_history, np.nan)
    stamps, _ = pad_lists(times, config.max_history, np.nan)
    stars = np.where(np.isfinite(stars), stars, 4.0)
    age = (np.asarray(as_of, dtype="float64")[:, None] - stamps) / SECONDS_PER_DAY
    age = np.where(np.isfinite(age), np.maximum(age, 0), 0.0)
    disliked, dis_mask = pad_lists(dislikes, config.max_dislikes, n_items)
    return {
        "items": torch.from_numpy(items.astype(np.int64)), "mask": torch.from_numpy(mask),
        "ratings": torch.from_numpy(stars.astype("float32")), "age_days": torch.from_numpy(age.astype("float32")),
        "dislikes": torch.from_numpy(disliked.astype(np.int64)), "dislike_mask": torch.from_numpy(dis_mask),
    }


class TwoTower(nn.Module):
    def __init__(self, item_features, embedding_dim=32, hidden_dim=64, identity_dim=16, history=None):
        super().__init__()
        features = torch.as_tensor(np.array(item_features,copy=True),dtype=torch.float32)
        self.register_buffer('item_features',features)
        self.identity = nn.Embedding(len(features),identity_dim,sparse=True)
        nn.init.normal_(self.identity.weight,std=.05)
        self.history = HistoryConfig(**history) if isinstance(history, dict) else (history or HistoryConfig())
        h = self.history
        extra = 0
        if h.active:
            # Row n_items is the padding row; it stays zero.
            self.history_identity = nn.Embedding(len(features) + 1, h.history_dim, sparse=True, padding_idx=len(features))
            nn.init.normal_(self.history_identity.weight, std=.05)
            with torch.no_grad():
                self.history_identity.weight[len(features)].zero_()
            self.register_buffer('padded_features', torch.cat([features, torch.zeros(1, features.shape[1])]))
        if h.favorites:
            extra += h.history_dim
            if h.pooling == 'learned':
                self.pool_weights = nn.Linear(2, 1)
                nn.init.zeros_(self.pool_weights.weight)
                nn.init.zeros_(self.pool_weights.bias)
        if h.dislikes:
            extra += h.history_dim + features.shape[1] + 1
        self.user_tower = nn.Sequential(nn.Linear(features.shape[1]+1+extra,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,embedding_dim))
        self.item_tower = nn.Sequential(nn.Linear(features.shape[1]+identity_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,embedding_dim))
        self.config = dict(embedding_dim=embedding_dim,hidden_dim=hidden_dim,identity_dim=identity_dim)
        if h.active:
            self.config['history'] = asdict(h)

    def favorite_weights(self, batch):
        mask = batch['mask'].float()
        pooling = self.history.pooling
        if pooling == 'mean':
            weights = mask
        elif pooling == 'rating':
            weights = (batch['ratings'] - 3.5).clamp(min=0.1) * mask
        elif pooling == 'recency':
            weights = torch.pow(0.5, batch['age_days'] / self.history.half_life_days) * mask
        else:
            logits = self.pool_weights(torch.stack([batch['ratings'] - 4, torch.log1p(batch['age_days'])], -1)).squeeze(-1)
            weights = torch.softmax(logits.masked_fill(~batch['mask'], -1e9), dim=-1) * mask
        return weights / weights.sum(-1, keepdim=True).clamp(min=1e-9)

    def history_inputs(self, batch):
        parts = []
        if self.history.favorites:
            weights = self.favorite_weights(batch)
            parts.append((self.history_identity(batch['items']) * weights.unsqueeze(-1)).sum(1))
        if self.history.dislikes:
            dmask = batch['dislike_mask'].float()
            count = dmask.sum(-1, keepdim=True)
            denom = count.clamp(min=1)
            parts.append((self.history_identity(batch['dislikes']) * dmask.unsqueeze(-1)).sum(1) / denom)
            parts.append((self.padded_features[batch['dislikes']] * dmask.unsqueeze(-1)).sum(1) / denom)
            parts.append((count > 0).float())
        return parts

    def encode_user(self, profiles, history=None):
        if self.history.active:
            if history is None:
                raise ValueError('This viewer tower needs favorite/dislike history inputs')
            profiles = torch.cat([profiles, *self.history_inputs(history)], dim=-1)
        return F.normalize(self.user_tower(profiles),dim=-1)

    def encode_item(self, item_ids):
        return F.normalize(self.item_tower(torch.cat([self.identity(item_ids),self.item_features[item_ids]],dim=-1)),dim=-1)

    def sparse_parameters(self):
        return [p for name, p in self.named_parameters() if name in ('identity.weight', 'history_identity.weight')]

    def dense_parameters(self):
        return [p for name, p in self.named_parameters() if name not in ('identity.weight', 'history_identity.weight')]

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
