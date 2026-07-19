"""Run more than one million reproducible tracking experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from iscir.evaluation.massive_experiments import MassiveCampaignConfig, run_massive_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=1_050_000)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/experiments/generated/million_campaign"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_massive_campaign(
        args.output_dir,
        config=MassiveCampaignConfig(
            total_trials=args.trials,
            workers=args.workers,
            chunk_size=args.chunk_size,
            checkpoint_every=args.checkpoint_every,
            first_seed=args.first_seed,
        ),
    )
    print(f"Campaign complete: {args.trials:,} trials")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
