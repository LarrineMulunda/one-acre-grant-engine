"""
reopen_cascade_cli.py — command-line entry point for reopen_and_cascade().

Usage:
    python reopen_cascade_cli.py --period-id 2021_2024 \
        --reason "Late-arriving transaction discovered post-close" \
        --corrected-by "Larrine Mulunda"

This is a separate file from close_period_cli.py deliberately — closing a
period is a routine operation; reopening and cascading a correction through
every later period is a higher-risk one that touches numbers already
signed off. Keeping them as distinct entry points makes it harder to run
the wrong one by accident, and gives a natural place to attach different
access control later if this ever needs it.

Requires a reachable Prefect server, same as close_period_cli.py:
"""
import argparse
import os

from prefect import flow

from src.pipeline_flow import reopen_and_cascade


@flow
def run_reopen_and_cascade(period_id: str, reason: str, corrected_by: str) -> None:
    os.makedirs("exports", exist_ok=True)
    os.makedirs("exports/snapshots", exist_ok=True)
    reopen_and_cascade(period_id, reason=reason, corrected_by=corrected_by)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reopen a CLOSED period to correct it, then cascade the "
                     "correction forward through every later period — "
                     "reopening and reclosing anything already closed, "
                     "recomputing anything still open. Snapshots the "
                     "before/after state of every affected period as "
                     "change-request evidence."
    )
    parser.add_argument(
        "--period-id", required=True,
        help="The CLOSED period to reopen and correct, e.g. 2021_2024 — "
             "must currently have Status='CLOSED' in accounting_periods.",
    )
    parser.add_argument(
        "--reason", required=True,
        help="Why this correction is happening — recorded in the run logs "
             "for audit purposes, e.g. 'Late-arriving transaction found'.",
    )
    parser.add_argument(
        "--corrected-by", required=True,
        help="Name of the person making the correction, recorded as "
             "ClosedBy when each affected period is reclosed.",
    )
    args = parser.parse_args()

    run_reopen_and_cascade(
        period_id=args.period_id,
        reason=args.reason,
        corrected_by=args.corrected_by,
    )


if __name__ == "__main__":
    main()