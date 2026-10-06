"""Offline OpenAPI export command."""

from __future__ import annotations

import argparse
from pathlib import Path

from trustworthy_recsys.serving.app import export_openapi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("frontend/openapi.json"))
    args = parser.parse_args()
    export_openapi(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
