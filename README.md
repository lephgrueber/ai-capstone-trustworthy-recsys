# Trustworthy Recommendation System

A recommendation-system capstone exploring multi-stage retrieval, reranking, and trustworthy offline evaluation.

Developed for CIS 5980: AI Capstone at the University of Pennsylvania under the AI Engineering track.

## Project

The project asks whether a new ranking policy can be evaluated reliably from biased historical interaction logs before it is tested live.

The planned system includes two-tower retrieval, approximate nearest-neighbor search, LambdaRank reranking, and offline policy evaluation.

The core system is not implemented yet. Milestone 1 focuses on project scoping and repository setup.

## Quick Start

Requires Python 3.11 or later.

```bash
git clone https://github.com/lephgrueber/ai-capstone-trustworthy-recsys
cd ai-capstone-trustworthy-recsys

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pytest

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE).
