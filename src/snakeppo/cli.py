"""``snakeppo`` command line. Task 12 adds the train, eval, agents and compare commands."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="snakeppo", description="Train and evaluate PPO agents that play Snake."
    )
    parser.parse_args(argv)
    return 0
