"""
core_allocator.py  —  Grant Allocation Engine (Production)
===========================================================
A pure, stateless function of (grants_df, expenses_df, period_id, tie_break,
opening_balances) that allocates every expense dollar to one or more grants
according to grants funding rules and returns three audit-ready
DataFrames plus a lineage RunId.

Design invariants
-----------------
  1. Deterministic   — same inputs always produce identical outputs.
  2. Idempotent      — safe to call N times; no side-effects, no DB writes.
  3. Reconciled      — Σ allocated + Σ unallocated == Σ expenses (verified
                       by the caller via _assert_reconciliation in pipeline.py).
  4. Auditable       — every row carries RunId + AllocationOrder so any dollar
                       can be traced back to the exact engine invocation.
  5. Period-scoped   — every call operates on exactly one accounting period.
                       Balances are seeded from the prior closed period, never
                       recomputed from all-time history. Whether a period is
                       OPEN or CLOSED, and whether a write is a full swap or a
                       reversal, is decided by the caller (pipeline.py) — this
                       module has no notion of "open" or "closed," only of
                       "which balance did I start from."



A note on zero-division handling
-----------------------------------
ConsumedPct in the balance snapshot is computed with a guard against
TotalAmount == 0 (np.where(total > 0, consumed / total, 0.0))

A note on null-restriction matching
-------------------------------------
When an expense has a NaN restriction value (e.g. no Country recorded), it
matches only wildcard grants — those with a null restriction on that column.

"""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Period definitions
# ─────────────────────────────────────────────────────────────────────────────

# Every transaction date must fall inside exactly one of these ranges.
# "2021_2024" is the initial backfill period — it has no prior closed period,

PERIOD_DEFINITIONS: dict[str, tuple[date, date]] = {
    "2021_2024": (date(2021, 1, 1), date(2024, 12, 31)),
    "2025_H1": (date(2025, 1, 1), date(2025, 6, 30)),
    "2025_H2": (date(2025, 7, 1), date(2025, 12, 31)),
    "2026_H1":      (date(2026, 1, 1), date(2026, 6, 30)),
    "2026_H2":      (date(2026, 7, 1), date(2026, 12, 31))
    
}


def get_period_id(txn_date) -> str:
    """
    Resolve a transaction date to its period_id.

    Raises ValueError if the date falls outside every defined period —
    this is a configuration gap (a new period needs to be added to
    PERIOD_DEFINITIONS), not a data-quality issue, so it should fail loudly
    rather than being silently classified as unallocated.
    """
    d = pd.Timestamp(txn_date).date()
    for period_id, (start, end) in PERIOD_DEFINITIONS.items():
        if start <= d <= end:
            return period_id
    raise ValueError(
        f"Date {d} falls outside every defined period in PERIOD_DEFINITIONS. "
        f"Add a new period entry before processing transactions in this range."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

GRANT_RESTRICTIONS = [
    "BusinessUnit", "Country", "Account", "ProjectName", "DepartmentName"
]

_ALLOC_COLS = (
    "RunId", "PeriodId", "AllocationOrder", "TransactionId", "TransactionDate",
    "GrantCode", "AllocatedAmount", "GrantPriority", "GrantEndDate",
)

_UNALLOC_COLS = (
    "RunId", "PeriodId", "TransactionId", "TransactionDate",
    "BusinessUnit", "Country", "Account", "ProjectName", "DepartmentName",
    "OriginalAmount", "UnallocatedAmount", "Reason",
)

_BALANCE_COLS = (
    "PeriodId", "GrantCode", "GrantName", "Priority", "EndDate",
    "OpeningAmount", "TotalAmount", "RemainingAmount",
    "ConsumedAmount", "ConsumedPct", "RunId",
)


# ─────────────────────────────────────────────────────────────────────────────
# Public helpers (used by pipeline.py and unit tests)
# ─────────────────────────────────────────────────────────────────────────────

def compute_run_id(
    grants_df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    period_id: str,
    tie_break: tuple,
) -> str:
    """
    Stable SHA-256 of canonicalised inputs + period_id + tie-break config —
    the lineage anchor. Including period_id means the same transactions
    processed under a different period boundary (e.g. a corrected
    PERIOD_DEFINITIONS entry) always produce a distinct RunId, never a
    silent collision with a prior run.
    """
    h = hashlib.sha256()
    for df, key in [(grants_df, "GrantCode"), (expenses_df, "TransactionId")]:
        canonical = df.sort_values(key).to_csv(index=False).encode()
        h.update(canonical)
    h.update(period_id.encode())
    h.update(json.dumps(list(tie_break)).encode())
    return h.hexdigest()[:12]


def build_closing_balances(grant_balances_df: pd.DataFrame) -> dict[str, float]:
    """
    Extract {GrantCode: RemainingAmount} from a period's grant_balances
    output. This is exactly what the caller passes as `opening_balances`
    when seeding the next period — the hand-off point between two calls
    to allocate(). Pure transformation; writes nothing.
    """
    return dict(zip(
        grant_balances_df["GrantCode"],
        grant_balances_df["RemainingAmount"],
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Private engine internals
# ─────────────────────────────────────────────────────────────────────────────

def _build_grant_arrays(grants_df: pd.DataFrame) -> dict:
    """Pre-compute numpy column views and null masks once per run."""
    return {
        "code":  grants_df["GrantCode"].to_numpy(),
        "start": grants_df["StartDate"].to_numpy(),
        "end":   grants_df["EndDate"].to_numpy(),
        "prio":  grants_df["Priority"].to_numpy(),
        "rest":  {c: grants_df[c].to_numpy()         for c in GRANT_RESTRICTIONS},
        "null":  {c: pd.isna(grants_df[c].to_numpy()) for c in GRANT_RESTRICTIONS},
    }


def _resolve_opening_balances(
    grants_df: pd.DataFrame,
    opening_balances: dict[str, float] | None,
    period_start: date,
) -> np.ndarray:
    """
    Build the starting balance array, aligned to grants_df row order.

    If opening_balances is None, every grant starts at its full TotalAmount
    — correct only for a period with no prior closed period (i.e. the very
    first period). Every later period must pass an explicit opening_balances
    dict (typically from build_closing_balances() on the prior period's
    output).

    A grant present in grants_df but absent from opening_balances is
    handled one of two ways, distinguished by the grant's own StartDate:

      - StartDate >= period_start: the grant genuinely did not exist as of
        the prior period's close — there was never a period in which it
        could have accrued a balance. It seeds at its own TotalAmount,
        same as a first-ever period. This is the normal, expected shape of
        "a new grant was approved mid-stream," not an error condition.

      - StartDate < period_start: the grant existed before this period
        started and should have a carried-forward balance, but doesn't.
        This is a real gap — a broken hand-off, a grant dropped from the
        closing snapshot, a bug — and must not be silently papered over by
        guessing a starting balance. Raises.
    """
    if opening_balances is None:
        return grants_df["TotalAmount"].to_numpy(dtype="float64").copy()

    missing_codes = set(grants_df["GrantCode"]) - set(opening_balances.keys())

    opening = grants_df["GrantCode"].map(opening_balances)

    if missing_codes:
        missing_mask = grants_df["GrantCode"].isin(missing_codes)
        period_start_ts = pd.Timestamp(period_start)
        genuinely_new = missing_mask & (grants_df["StartDate"] >= period_start_ts)
        suspicious = missing_mask & ~genuinely_new

        if suspicious.any():
            bad_codes = sorted(grants_df.loc[suspicious, "GrantCode"].tolist())
            raise ValueError(
                f"{len(bad_codes)} grant(s) are missing a carried-forward balance "
                f"despite predating this period ({period_start}) — this is a real "
                f"gap in the prior period's closing snapshot, not a new grant, "
                f"and will not be silently seeded: {bad_codes[:5]}"
                f"{'...' if len(bad_codes) > 5 else ''}"
            )

        # Genuinely new grants: start fresh at their own TotalAmount.
        
        opening = opening.where(~genuinely_new, grants_df["TotalAmount"])

    return opening.to_numpy(dtype="float64")


def _eligibility(tx, ga: dict, balances: np.ndarray) -> tuple:
    """
    Return the set of grant indices eligible for transaction `tx`.

    Diagnostic flags (in_window, restriction_match) are captured BEFORE the
    balance filter so _classify_reason can distinguish
    ELIGIBLE_GRANTS_EXHAUSTED (grants existed but ran out) from
    NO_RESTRICTION_MATCH (no grant ever qualified).
    """
    in_window_mask = (ga["start"] <= tx.TransactionDate) & (ga["end"] >= tx.TransactionDate)

    mask = in_window_mask.copy()
    for col in GRANT_RESTRICTIONS:
        mask &= ga["null"][col] | (ga["rest"][col] == getattr(tx, col))

    in_window         = bool(in_window_mask.any())
    restriction_match = bool(mask.any())

    mask &= balances > 0
    eligible_idx = np.where(mask)[0]

    return eligible_idx, in_window, restriction_match


def _draw_down(
    tx,
    eligible_idx: np.ndarray,
    ga: dict,
    balances: np.ndarray,
    tie_keys: list,
    run_id: str,
    period_id: str,
    alloc_order: int,
    alloc_cols: dict,
) -> tuple:
    """
    Draw `tx.Amount` from eligible grants in tie-break order.

    Mutates `balances` in-place and appends rows to `alloc_cols`.
    Returns (remaining_unallocated, new_alloc_order).
    """
    eligible_idx = sorted(eligible_idx, key=lambda i: tie_keys[i])

    remaining = float(tx.Amount)
    for i in eligible_idx:
        if remaining <= 0:
            break

        avail = balances[i]
        take  = avail if avail < remaining else remaining
        alloc_order += 1

        alloc_cols["RunId"].append(run_id)
        alloc_cols["PeriodId"].append(period_id)
        alloc_cols["AllocationOrder"].append(alloc_order)
        alloc_cols["TransactionId"].append(tx.TransactionId)
        alloc_cols["TransactionDate"].append(tx.TransactionDate)
        alloc_cols["GrantCode"].append(ga["code"][i])
        alloc_cols["AllocatedAmount"].append(take)
        alloc_cols["GrantPriority"].append(int(ga["prio"][i]))
        alloc_cols["GrantEndDate"].append(ga["end"][i])

        balances[i] = avail - take
        remaining  -= take

    return remaining, alloc_order


def _classify_reason(
    remaining: float,
    in_window: bool,
    restriction_match: bool,
) -> str | None:
    """
    Map the post-draw state to an unallocated reason code, or None if fully funded.

    Reason codes (surfaced to Finance in the unallocated table):
      OUT_OF_DATE_WINDOW        — transaction date falls outside every grant window.
      NO_RESTRICTION_MATCH      — no grant's restrictions match this expense's metadata.
      ELIGIBLE_GRANTS_EXHAUSTED — matching grants existed but all ran out of budget.
    """
    if remaining <= 0:
        return None
    if not in_window:
        return "OUT_OF_DATE_WINDOW"
    if not restriction_match:
        return "NO_RESTRICTION_MATCH"
    return "ELIGIBLE_GRANTS_EXHAUSTED"


def _record_unallocated(
    tx,
    remaining: float,
    reason: str,
    run_id: str,
    period_id: str,
    unalloc_cols: dict,
) -> None:
    """Append one unallocated row to the columnar accumulator."""
    unalloc_cols["RunId"].append(run_id)
    unalloc_cols["PeriodId"].append(period_id)
    unalloc_cols["TransactionId"].append(tx.TransactionId)
    unalloc_cols["TransactionDate"].append(tx.TransactionDate)
    unalloc_cols["BusinessUnit"].append(tx.BusinessUnit)
    unalloc_cols["Country"].append(tx.Country)
    unalloc_cols["Account"].append(tx.Account)
    unalloc_cols["ProjectName"].append(tx.ProjectName)
    unalloc_cols["DepartmentName"].append(tx.DepartmentName)
    unalloc_cols["OriginalAmount"].append(tx.Amount)
    unalloc_cols["UnallocatedAmount"].append(remaining)
    unalloc_cols["Reason"].append(reason)


def _build_balance_frame(
    grants_df: pd.DataFrame,
    opening: np.ndarray,
    balances: np.ndarray,
    run_id: str,
    period_id: str,
) -> pd.DataFrame:
    """
    Build the per-grant balance snapshot for this period.

    OpeningAmount is what the period started with (either TotalAmount for
    2021_2025, or the prior period's closing balance for every period
    after). TotalAmount is retained separately so ConsumedPct always reads
    against the grant's true lifetime total, not just this period's slice.
    """
    total    = grants_df["TotalAmount"].to_numpy(dtype="float64")
    consumed = total - balances
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = np.where(total > 0, consumed / total, 0.0)

    return pd.DataFrame({
        "PeriodId":        period_id,
        "GrantCode":       grants_df["GrantCode"].to_numpy(),
        "GrantName":       grants_df["GrantName"].to_numpy(),
        "Priority":        grants_df["Priority"].to_numpy(),
        "EndDate":         grants_df["EndDate"].to_numpy(),
        "OpeningAmount":   opening,
        "TotalAmount":     total,
        "RemainingAmount": balances,
        "ConsumedAmount":  consumed,
        "ConsumedPct":     pct,
        "RunId":           run_id,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def allocate(
    grants_df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    period_id: str,
    tie_break: tuple = ("EndDate", "Priority", "GrantCode"),
    opening_balances: dict[str, float] | None = None,
) -> dict:
    """
    Allocate every expense dollar to one or more grants per grant business rules,
    scoped to a single accounting period.

    Parameters
    ----------
    grants_df        : DataFrame containing the grant master columns.
    expenses_df      : DataFrame containing this period's expense/transaction
                        rows only. Every TransactionDate must fall inside
                        PERIOD_DEFINITIONS[period_id] — see get_period_id().
    period_id         : One of PERIOD_DEFINITIONS' keys. Stamped onto every
                        output row and folded into the RunId hash.
    tie_break         : Column-name tuple controlling grant priority when a
                        transaction qualifies for multiple grants. Default
                        ("EndDate", "Priority", "GrantCode") burns the
                        closest-to-expiry grant first.
    opening_balances  : {GrantCode: balance} carried forward from the prior
                        closed period's build_closing_balances() output.
                        None only for period_id == "2021_2025", the first
                        period, which has no prior period to inherit from.

    Returns
    -------
    dict with keys:
        "allocations"    pd.DataFrame  — one row per (transaction, grant) draw.
        "unallocated"    pd.DataFrame  — one row per transaction with leftover $.
        "grant_balances" pd.DataFrame  — per-grant opening/remaining $ and % consumed.
        "run_id"         str           — 12-char SHA-256 lineage anchor.

    Caller responsibilities
    ------------------------
    - Schema and business-rule validation run BEFORE calling allocate().
    - Every expenses_df row's date must already be scoped to period_id —
      this function does not filter by date itself, it trusts the caller's
      windowing (see pipeline._load_period_transactions()).
    - Reconciliation (Σ allocated + Σ unallocated == Σ expenses) is asserted
      AFTER calling allocate().
    - Whether this period's output is written via atomic swap (OPEN) or
      rejected/routed to a reversal path (CLOSED) is decided by pipeline.py,
      not by this module.
    """
    run_id = compute_run_id(grants_df, expenses_df, period_id, tie_break)

    ga      = _build_grant_arrays(grants_df)
    period_start, _period_end = PERIOD_DEFINITIONS[period_id]
    opening = _resolve_opening_balances(grants_df, opening_balances, period_start)
    balances = opening.copy()
    tie_keys = list(zip(*[grants_df[c].to_numpy() for c in tie_break]))

    # Process transactions in strict date order; TransactionId breaks ties
    # deterministically so RunId is stable regardless of input row order.
    txns = (
        expenses_df
        .sort_values(["TransactionDate", "TransactionId"])
        .reset_index(drop=True)
    )

    alloc_cols   = {k: [] for k in _ALLOC_COLS}
    unalloc_cols = {k: [] for k in _UNALLOC_COLS}
    alloc_order  = 0

    for tx in txns.itertuples(index=False):
        eligible_idx, in_window, restriction_match = _eligibility(tx, ga, balances)

        remaining, alloc_order = _draw_down(
            tx, eligible_idx, ga, balances, tie_keys,
            run_id, period_id, alloc_order, alloc_cols,
        )

        reason = _classify_reason(remaining, in_window, restriction_match)
        if reason is not None:
            _record_unallocated(tx, remaining, reason, run_id, period_id, unalloc_cols)

    return {
        "allocations":    pd.DataFrame(alloc_cols),
        "unallocated":    pd.DataFrame(unalloc_cols),
        "grant_balances": _build_balance_frame(grants_df, opening, balances, run_id, period_id),
        "run_id":         run_id,
    }