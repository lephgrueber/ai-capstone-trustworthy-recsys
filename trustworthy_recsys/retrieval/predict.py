"""Load the published two-tower model and request recommendations from Weaviate."""

import argparse
import json
from pathlib import Path

import torch

from eval.harness import RecommendationRequest
from .recommender import RetrievalRecommender


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--user-id', default='demo-user')
    parser.add_argument('--history-items', nargs='*', default=[])
    parser.add_argument('--k', type=int, default=10)
    args = parser.parse_args()
    if args.k <= 0:
        parser.error('--k must be positive')
    torch.set_num_threads(2)
    with RetrievalRecommender.load(args.inputs_dir, args.model_dir) as model:
        request = RecommendationRequest(args.user_id, args.history_items)
        ranked = model(request, args.k)
        print(json.dumps({'user_id': args.user_id, 'ranked_items': ranked}, indent=2))


if __name__ == '__main__':
    main()
