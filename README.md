# Trustworthy Recommendation System

A recommendation-system capstone exploring multi-stage retrieval, reranking, and trustworthy offline evaluation.

Developed for CIS 5980: AI Capstone at the University of Pennsylvania under the AI Engineering track.

## Project

This project builds an end-to-end consumer-facing recommendation system around a two-stage retrieval and ranking architecture.

The system retrieves a candidate set using a two-tower model with approximate nearest-neighbor search, reranks those candidates with LambdaRank, and presents the resulting recommendations through a user-facing interface.

The project also includes offline evaluation using direct method (DM), inverse propensity scoring (IPS), self-normalized IPS (SNIPS), and doubly robust (DR) estimation, along with retrieval, ranking, latency, and estimator-reliability results.

## Project Structure

The repository separates reusable project code from data, evaluation, tests, documentation, and experiment outputs.

```text
ai-capstone-trustworthy-recsys/
├── data/                         # Local raw and processed datasets
├── docs/                         # Project documentation and milestone artifacts
├── eval/                         # Runnable evaluation scripts
├── results/                      # Saved evaluation and experiment results
├── scripts/                      # Utility and experiment scripts
├── tests/                        # Automated tests
├── trustworthy_recsys/           # Main Python package
│   ├── baselines/                # Recommendation baselines
│   ├── data/                     # Data loading and preprocessing logic
│   └── evaluation/               # Evaluation metrics and analysis
├── pyproject.toml                # Project configuration and dependencies
└── README.md

## Quick Start

Requires Python 3.11 or later.

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/lephgrueber/ai-capstone-trustworthy-recsys
cd ai-capstone-trustworthy-recsys

python -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).