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
- ``metadata``: the favorites' credit, runtime and tag summaries from the movie side,
  pooled with the same favorite weights (requires ``favorites`` and a movie side).

Movie-tower metadata (``movie_side``; see movie_side.py): per-role person vectors averaged
over the movie's directors / writers / cast with missing-role flags, scaled runtime with a
missing flag, and the projected tag summary with a has-tags flag.
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
    metadata: bool = False

    def __post_init__(self):
        if self.metadata and not self.favorites:
            raise ValueError("metadata pooling needs favorites=True")
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
    def __init__(self, item_features, embedding_dim=32, hidden_dim=64, identity_dim=16, history=None,
                 movie_side=None, side=None):
        """`movie_side` is the saved MovieSide.state(); `side` the rebuilt MovieSide object."""
        super().__init__()
        features = torch.as_tensor(np.array(item_features,copy=True),dtype=torch.float32)
        self.register_buffer('item_features',features)
        self.identity = nn.Embedding(len(features),identity_dim,sparse=True)
        nn.init.normal_(self.identity.weight,std=.05)
        self.history = HistoryConfig(**history) if isinstance(history, dict) else (history or HistoryConfig())
        h = self.history
        self.side = side
        self.side_state = movie_side
        side_width = 0
        if side is not None:
            from .movie_side import ROLES
            c = side.config
            if c.credits:
                self.people = nn.ModuleDict({role: nn.Embedding(n, c.person_dim, padding_idx=0, sparse=True)
                                             for role, n in side.vocab_sizes().items()})
                for table in self.people.values():
                    nn.init.normal_(table.weight, std=.05)
                    with torch.no_grad():
                        table.weight[0].zero_()
                for role in ROLES:
                    self.register_buffer(f'credit_{role}', torch.from_numpy(
                        np.vstack([side.credit_ids[role], np.zeros((1, side.credit_ids[role].shape[1]), dtype=np.int64)])),
                        persistent=False)
                side_width += len(ROLES) * (c.person_dim + 1)
            if c.runtime:
                self.register_buffer('runtime', torch.from_numpy(np.vstack([side.runtime, [[0, 1]]]).astype('float32')),
                                     persistent=False)
                side_width += 2
            if c.tags:
                side_width += side.tag_width
                self.tag_rows = None  # catalog tag matrix as of the current inference time
        self.side_width = side_width
        if h.metadata and side is None:
            raise ValueError('metadata pooling needs a movie side')
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
        if h.metadata:
            extra += side_width
        self.user_tower = nn.Sequential(nn.Linear(features.shape[1]+1+extra,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,embedding_dim))
        self.item_tower = nn.Sequential(nn.Linear(features.shape[1]+identity_dim+side_width,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,embedding_dim))
        self.config = dict(embedding_dim=embedding_dim,hidden_dim=hidden_dim,identity_dim=identity_dim)
        if h.active:
            self.config['history'] = asdict(h)
        if side is not None:
            self.config['movie_side'] = movie_side if movie_side is not None else side.state()

    def set_as_of(self, as_of):
        """Fix inference-time tag summaries to tags first used before `as_of`."""
        if self.side is not None and self.side.config.tags:
            rows = self.side.tag_matrix(as_of)
            self.tag_rows = torch.from_numpy(np.vstack([rows, np.zeros((1, rows.shape[1]), dtype='float32')]))
        self.as_of = as_of
        return self

    def item_side(self, item_ids, tag_rows=None):
        """Metadata features for item indices (any shape); index n_items is padding."""
        from .movie_side import ROLES, UNKNOWN
        parts = []
        c = self.side.config
        if c.credits:
            for role in ROLES:
                ids = getattr(self, f'credit_{role}')[item_ids]
                if self.training and c.unknown_mask_rate:
                    # Occasionally hide a known person so the shared UNKNOWN vector is trained.
                    hide = (torch.rand(ids.shape) < c.unknown_mask_rate) & (ids > UNKNOWN)
                    ids = ids.masked_fill(hide, UNKNOWN)
                present = (ids > 0).float()
                count = present.sum(-1, keepdim=True)
                pooled = (self.people[role](ids) * present.unsqueeze(-1)).sum(-2) / count.clamp(min=1)
                parts += [pooled, (count == 0).float()]
        if c.runtime:
            parts.append(self.runtime[item_ids])
        if c.tags:
            if tag_rows is None:
                if self.tag_rows is None:
                    raise ValueError('Call set_as_of() before encoding movies with tag inputs')
                tag_rows = self.tag_rows[item_ids]
            parts.append(tag_rows)
        return torch.cat(parts, dim=-1)

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
            if self.history.metadata:
                side = self.item_side(batch['items'], batch.get('favorite_tags'))
                parts.append((side * weights.unsqueeze(-1)).sum(1))
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

    def encode_item(self, item_ids, tag_rows=None):
        parts=[self.identity(item_ids),self.item_features[item_ids]]
        if self.side is not None:
            parts.append(self.item_side(item_ids, tag_rows))
        return F.normalize(self.item_tower(torch.cat(parts,dim=-1)),dim=-1)

    def _sparse(self, name):
        return name in ('identity.weight', 'history_identity.weight') or name.startswith('people.')

    def sparse_parameters(self):
        return [p for name, p in self.named_parameters() if self._sparse(name)]

    def dense_parameters(self):
        return [p for name, p in self.named_parameters() if not self._sparse(name)]

    @torch.inference_mode()
    def all_item_vectors(self,batch_size=4096):
        self.eval()
        return np.concatenate([self.encode_item(torch.arange(start,min(start+batch_size,len(self.item_features)))).numpy()
                               for start in range(0,len(self.item_features),batch_size)]).astype('float32')


def load_model(path,features,item_ids=None,as_of=None):
    """Load towers; models with movie metadata rebuild it for `item_ids` from verified sources."""
    saved=torch.load(path,map_location='cpu',weights_only=True)
    config=dict(saved['config'])
    side=None
    if config.get('movie_side'):
        from .movie_side import MovieSide, MovieSideConfig, verify_sources
        if item_ids is None:
            raise ValueError('This model uses movie metadata; pass the catalog item_ids')
        state=config['movie_side']
        side=MovieSide(MovieSideConfig(**state['config']),item_ids,state['vocab'],state['runtime_scale'])
        verify_sources(side,state['sources'])
        config['side']=side
    model=TwoTower(features,**config)
    model.load_state_dict(saved['state_dict'])
    model.eval()
    if side is not None:
        model.set_as_of(as_of if as_of is not None else saved.get('as_of'))
    return model
