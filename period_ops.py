"""
period_ops.py — standalone entry points for close_period() and
reopen_and_cascade(), both of which need an active Prefect run context
(get_run_logger() requires one) — wrapping each in a minimal @flow provides
that, the same way end_to_end_pipeline() already does implicitly.

Requires a reachable Prefect server (same as everything else in this
project) — either:
    Terminal 1: prefect server start
    Terminal 2: prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
                python period_ops.py
"""
import os
from prefect import flow

from src.pipeline_flow import close_period, reopen_and_cascade


@flow
def run_close_period(period_id: str, closed_by: str):
    os.makedirs("exports", exist_ok=True)
    close_period.fn(period_id, closed_by=closed_by)


@flow
def run_reopen_and_cascade(period_id: str, reason: str, corrected_by: str):
    reopen_and_cascade(period_id, reason=reason, corrected_by=corrected_by)


if __name__ == "__main__":
    # Uncomment exactly one of these before running:

    # run_close_period("2021_2026", closed_by="Larrine Mulunda")

    # run_reopen_and_cascade(
    #     "2021_2026",
    #     reason="Late-arriving transaction discovered post-close",
    #     corrected_by="Larrine Mulunda",
    # )
    pass
