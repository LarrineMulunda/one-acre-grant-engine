"""
close_period_cli.py — command-line entry point for close_period().

Usage:
    python close_period_cli.py --period-id 2021_2024 --closed-by "Larrine Mulunda"

Requires a reachable Prefect server (close_period internally uses
get_run_logger(), which needs an active flow run context — see the @flow
wrapper below):
    Terminal 1: prefect server start
    Terminal 2: prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
                python close_period_cli.py --period-id ... --closed-by ...
"""
import argparse
import os

from prefect import flow

from src.pipeline_flow import close_period


@flow
def run_close_period(period_id: str, closed_by: str) -> None:
    os.makedirs("exports", exist_ok=True)
    close_period.fn(period_id, closed_by=closed_by)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Close an accounting period: reruns the allocation one "
                     "final time, marks the period CLOSED, exports the "
                     "certified files, and sends a closure notification."
    )
    parser.add_argument(
        "--period-id", required=True,
        help="The period to close, e.g. 2021_2024 — must be a key in "
             "PERIOD_DEFINITIONS (core_allocator.py) and currently OPEN.",
    )
    parser.add_argument(
        "--closed-by", required=True,
        help="Name of the person closing the period, recorded in "
             "accounting_periods.ClosedBy for the audit trail.",
    )
    args = parser.parse_args()

    run_close_period(period_id=args.period_id, closed_by=args.closed_by)


if __name__ == "__main__":
    main()