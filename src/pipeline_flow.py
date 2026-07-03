"""
pipeline.py  —  End-to-End Grant Allocation Pipeline (Production)
=================================================================
Author  : Larrine Mulunda
Context : One Acre Fund · Data Engineering · Final Round

Layers
------
  Bronze   — raw append / full-replace; nothing deleted, ever.  Audit origin.
  Silver   — cleaned, deduplicated, type-cast.
  Gold     — allocation outputs: allocations, unallocated, grant_balances.
             Scoped by PeriodId; immutable once accounting_periods marks
             that PeriodId CLOSED.
  Control  — etl_watermarks (raw ingestion cursor, one row per table),
             accounting_periods (OPEN/CLOSED per PeriodId — governs whether
             Gold can be atomically swapped for that period).

Watermarks vs. periods — two different mechanisms, don't conflate them
------------------------------------------------------------------------
etl_watermarks answers "how far into the raw source file have I already
copied into Bronze/Silver" — a Bronze/Silver ingestion optimization,
unrelated to allocation correctness.
accounting_periods answers "is this PeriodId still allowed to be
recomputed" — a Gold-layer governance gate. The allocator's window is
always PERIOD_DEFINITIONS[period_id], never "since the watermark."
"""

from __future__ import annotations

import logging
import smtplib
import traceback
from datetime import date, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import pandas as pd
from openai import OpenAI
from prefect import flow, get_run_logger, task
from prefect.blocks.system import Secret
from soda.scan import Scan
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.pool import NullPool

from src.core_allocator import allocate, PERIOD_DEFINITIONS, build_closing_balances

# ─────────────────────────────────────────────────────────────────────────────
# Engine / constants
# ─────────────────────────────────────────────────────────────────────────────

import os

# In a deployed environment (Prefect Cloud / a persistent Prefect server),
# the connection string comes from a Secret block, same as before. Locally,
# requiring a live Prefect server just to resolve a dev SQLite path is
# unnecessary fragility — one more process to remember to keep running,
# and its own separate point of failure (see: WinError 10061 the moment
# the local server isn't up). Falling back to an env var (or a hardcoded
# local default) keeps local runs self-contained.
try:
    DB_URI = Secret.load("warehouse-uri").get()
except Exception:
    DB_URI = os.environ.get("WAREHOUSE_URI", "sqlite:///data/my_local_database.db")
    logging.getLogger(__name__).warning(
        f"Could not reach Prefect server for the 'warehouse-uri' Secret block "
        f"— falling back to local DB_URI: {DB_URI}. This fallback should only "
        f"ever fire in local dev, never against a deployed flow."
    )

# NullPool: don't hold a pooled connection open between tasks — each Prefect
# task opens and closes its own connection, so a lingering pooled connection
# from an earlier task is a real way to self-collide on SQLite's file lock,
# separate from any external process (OneDrive, a stray python.exe, etc.).
# connect_args timeout + PRAGMA busy_timeout: wait for a lock instead of
# failing after SQLite's default 5s — that default is exactly why failures
# were landing ~5-6 seconds after each task started.
engine = create_engine(
    DB_URI,
    pool_pre_ping=True,
    poolclass=NullPool,
    connect_args={"timeout": 30},
)

with engine.begin() as _conn:
    _conn.execute(text("PRAGMA journal_mode=WAL"))
    _conn.execute(text("PRAGMA busy_timeout=30000"))

GRANT_RESTRICTIONS = [
    "BusinessUnit", "Country", "Account", "ProjectName", "DepartmentName"
]
GRANT_SCHEMA    = ["GrantCode", "GrantName", "Priority", "StartDate", "EndDate",
                   "TotalAmount"] + GRANT_RESTRICTIONS
EXPENSE_SCHEMA  = ["TransactionId", "TransactionDate", "Amount"] + GRANT_RESTRICTIONS

RETRY_POLICY = dict(retries=3, retry_delay_seconds=30)


# ─────────────────────────────────────────────────────────────────────────────
# Notifications — success/failure email, wired via Prefect flow-state hooks
# ─────────────────────────────────────────────────────────────────────────────
#
# Same pattern as DB_URI above: Secret blocks first (the real production
# path), env vars as a local-dev fallback. Credentials for a Gmail sender
# specifically need an App Password, not the account password — Gmail
# rejects plain SMTP auth for accounts with 2FA enabled, which is the
# default. Generate one at https://myaccount.google.com/apppasswords once
# 2-Step Verification is on, and use that 16-character value as
# SMTP_APP_PASSWORD, not the actual Google account password.

NOTIFY_TO = "mulslarry100@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def _get_smtp_credentials() -> tuple[str | None, str | None]:
    try:
        sender = Secret.load("smtp-sender-email").get()
        password = Secret.load("smtp-app-password").get()
        return sender, password
    except Exception:
        sender = os.environ.get("SMTP_SENDER_EMAIL")
        password = os.environ.get("SMTP_APP_PASSWORD")
        if not sender or not password:
            logging.getLogger(__name__).warning(
                "SMTP credentials not available (no Secret blocks reachable, "
                "and SMTP_SENDER_EMAIL/SMTP_APP_PASSWORD not set) — "
                "notification email skipped, not retried, pipeline result "
                "unaffected either way."
            )
        return sender, password


def _send_notification(subject: str, body: str) -> None:
    """
    Best-effort — a notification failure must never fail the pipeline run
    itself. Logs and returns rather than raising on any SMTP error.
    """
    sender, password = _get_smtp_credentials()
    if not sender or not password:
        return

    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = NOTIFY_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(sender, password)
            server.sendmail(sender, [NOTIFY_TO], msg.as_string())
        logging.getLogger(__name__).info(f"Notification email sent to {NOTIFY_TO}: {subject}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to send notification email: {e}")


def _get_ai_troubleshooting(error_message: str, traceback_text: str) -> str:
    """
    Best-effort, same fail-soft contract as _send_notification — an AI call
    failing (no key, network issue, rate limit) must never cascade into a
    second failure stacked on top of the pipeline's original one. Always
    returns a string; never raises.

    This is Use Case 3 from the Part 2 memo (Agentic DataOps &
    Observability) actually implemented, not just proposed: the agent
    drafts a remediation suggestion, a human engineer still has to read it,
    verify it against the real logs, and act — it never touches the
    pipeline or infrastructure itself.

    Model: gpt-5.4-mini — OpenAI's current cost-optimized tier for coding
    and professional-work tasks (verified against OpenAI's own model page
    at write time, not assumed from training data, since pricing/lineups
    shift). Swap for gpt-4.1-nano if this task's diagnostic quality is
    sufficient at an even lower cost — worth checking against a few real
    failures before deciding.
    """
    try:
        api_key = Secret.load("openai-api-key").get()
    except Exception:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return "(AI troubleshooting unavailable — no OPENAI_API_KEY configured.)"

    try:
        client = OpenAI(api_key=api_key)
        prompt = (
            "You are helping triage a production data pipeline failure. "
            "Context: a Prefect-orchestrated grant allocation pipeline — "
            "SQLite/Snowflake storage, a Bronze/Silver/Gold medallion "
            "layout, Soda Core data quality gates between Bronze and "
            "Silver, and a period-based accounting close process (open "
            "periods use atomic swap, closed periods are immutable).\n\n"
            "Given the exception and traceback below, provide a concise, "
            "numbered, step-by-step troubleshooting checklist a data "
            "engineer could act on immediately. Be specific to the actual "
            "error shown, not generic advice, and note which step is most "
            "likely to resolve it first based on the error itself.\n\n"
            f"Exception message:\n{error_message}\n\n"
            f"Traceback (most recent frames last):\n{traceback_text[-4000:]}\n"
        )
        response = client.chat.completions.create(
            model="gpt-5.4-mini",
            max_completion_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.getLogger(__name__).error(f"AI troubleshooting call failed: {e}")
        return f"(AI troubleshooting unavailable — {type(e).__name__}: {e})"


def _notify_success(flow_, flow_run, state) -> None:
    _send_notification(
        subject=f"[OK] Grant Allocation Pipeline succeeded — {flow_run.name}",
        body=(
            f"Flow run '{flow_run.name}' completed successfully.\n\n"
            f"State: {state.name}\n"
            f"Parameters: {flow_run.parameters}\n"
        ),
    )


def _notify_failure(flow_, flow_run, state) -> None:
    """
    state.message already contains the real exception text (e.g. "Flow run
    encountered an exception: ValueError: Reconciliation failed...") — not
    a generic placeholder, confirmed directly. state.result() additionally
    gives the raw exception object, which yields a full traceback showing
    exactly which task/line failed, not just the top-level message.
    """
    result = state.result(raise_on_failure=False)
    if isinstance(result, BaseException):
        full_traceback = "".join(
            traceback.format_exception(type(result), result, result.__traceback__)
        )
        error_message = str(result)
    else:
        full_traceback = "(no exception object available — see state.message below)"
        error_message = state.message or "(no message available)"

    ai_suggestions = _get_ai_troubleshooting(error_message, full_traceback)

    _send_notification(
        subject=f"[FAILED] Grant Allocation Pipeline failed — {flow_run.name}",
        body=(
            f"Flow run '{flow_run.name}' FAILED.\n\n"
            f"State: {state.name}\n"
            f"Message: {state.message}\n"
            f"Parameters: {flow_run.parameters}\n\n"
            f"--- Full traceback ---\n"
            f"{full_traceback}\n"
            f"--- End traceback ---\n\n"
            f"--- AI-suggested troubleshooting steps (unverified — read the "
            f"traceback yourself before acting on these) ---\n"
            f"{ai_suggestions}\n"
            f"--- End AI suggestions ---\n\n"
            f"Also check the Prefect UI for task-level logs."
        ),
    )


def _clean_restriction_column(series: pd.Series) -> pd.Series:
    """
    Stringify a restriction column with numeric-formatting normalization.

    Root-cause fix: grants and expenses are independent source files, and
    pandas infers numeric column dtype per-file — e.g. Account with no NaNs
    in the expenses file reads as int64 ("10002"), while the same column
    with any NaN in the grants file reads as float64 ("10002.0"). A plain
    astype(str) then produces two different strings for the same value,
    which silently fails the equality check in core_allocator._eligibility
    — not an exception, not a validation failure, just a grant that never
    matches for that specific value. Stripping a trailing ".0" after
    stringifying makes both sides converge on the same text regardless of
    which side's source file happened to trigger the float upcast.
    """
    s = series.astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Schema validation (data contract — Silver gate)
# ─────────────────────────────────────────────────────────────────────────────

def _validate_schema(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"[DATA CONTRACT VIOLATED] {label} is missing columns: {missing}")


def _validate_business_rules(grants: pd.DataFrame, expenses: pd.DataFrame) -> None:
    if (expenses["Amount"] <= 0).any():
        raise ValueError("Non-positive expense Amount detected — halting pipeline.")
    if (grants["TotalAmount"] <= 0).any():
        raise ValueError("Non-positive grant TotalAmount detected — halting pipeline.")
    if not grants["GrantCode"].is_unique:
        raise ValueError("Duplicate GrantCode — grants dimension is corrupt.")
    if not expenses["TransactionId"].is_unique:
        raise ValueError("Duplicate TransactionId in Silver expenses — deduplication failed.")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Watermark state management — Bronze/Silver ingestion only, unchanged
# ─────────────────────────────────────────────────────────────────────────────

@task(name="Initialize Watermark Table", **RETRY_POLICY)
def initialize_watermark_table() -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS etl_watermarks (
                TableName          TEXT PRIMARY KEY,
                LastProcessedDate  TEXT NOT NULL,
                UpdatedAt          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("""
            INSERT OR IGNORE INTO etl_watermarks (TableName, LastProcessedDate)
            VALUES ('raw_expenses', :baseline)
        """), {"baseline": "2023-01-01"})


@task(name="Get Current Watermark", **RETRY_POLICY)
def get_watermark(table_name: str) -> str:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT LastProcessedDate FROM etl_watermarks WHERE TableName = :t"),
            {"t": table_name},
        ).fetchone()
    return row[0] if row else "2023-01-01"


@task(name="Advance Watermark", **RETRY_POLICY)
def update_watermark(table_name: str, new_max_date: str) -> None:
    if not new_max_date:
        get_run_logger().warning("update_watermark called with empty date; skipping.")
        return
    with engine.begin() as conn:
        conn.execute(
            text("""
                UPDATE etl_watermarks
                SET    LastProcessedDate = :d,
                       UpdatedAt         = CURRENT_TIMESTAMP
                WHERE  TableName         = :t
            """),
            {"d": new_max_date, "t": table_name},
        )
    get_run_logger().info(f"Watermark [{table_name}] → {new_max_date}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Ingestion: Bronze & Silver — unchanged
# ─────────────────────────────────────────────────────────────────────────────

@task(name="Full Load Grants (Dimension)", **RETRY_POLICY)
def process_grants_full_load() -> None:
    logger = get_run_logger()
    logger.info("Executing FULL LOAD for Grants.")

    raw_grants = pd.read_csv("data/raw_grants.csv")
    _validate_schema(raw_grants, GRANT_SCHEMA, "raw_grants")

    with engine.begin() as conn:
        raw_grants.to_sql("bronze_grants", conn, if_exists="replace", index=False)

    silver = raw_grants.copy()
    silver["StartDate"] = pd.to_datetime(silver["StartDate"])
    silver["EndDate"]   = pd.to_datetime(silver["EndDate"])

    for col in GRANT_RESTRICTIONS:
        silver[col] = silver[col].where(silver[col].isna(), _clean_restriction_column(silver[col]))

    with engine.begin() as conn:
        silver.to_sql("silver_grants", conn, if_exists="replace", index=False)

    logger.info(f"Grants loaded: {len(silver):,} rows  ·  ${silver['TotalAmount'].sum():,.0f} total budget")


def _ensure_expense_tables() -> None:
    ddl_bronze = text("""
        CREATE TABLE IF NOT EXISTS bronze_expenses (
            TransactionId   TEXT NOT NULL,
            TransactionDate TEXT NOT NULL,
            Amount          REAL NOT NULL,
            BusinessUnit    TEXT,
            Country         TEXT,
            Account         TEXT,
            ProjectName     TEXT,
            DepartmentName  TEXT,
            UNIQUE (TransactionId)
        )
    """)
    ddl_silver = text("""
        CREATE TABLE IF NOT EXISTS silver_expenses (
            TransactionId   TEXT NOT NULL,
            TransactionDate TEXT NOT NULL,
            Amount          REAL NOT NULL,
            BusinessUnit    TEXT,
            Country         TEXT,
            Account         TEXT,
            ProjectName     TEXT,
            DepartmentName  TEXT,
            UNIQUE (TransactionId)
        )
    """)
    with engine.begin() as conn:
        conn.execute(ddl_bronze)
        conn.execute(ddl_silver)


def _insert_ignore(df: pd.DataFrame, table: str, conn, chunk_size: int = 100) -> int:
    if df.empty:
        return 0

    dialect  = engine.dialect.name
    columns  = list(df.columns)
    col_list = ", ".join(columns)
    total_written = 0

    for start_idx in range(0, len(df), chunk_size):
        chunk = df.iloc[start_idx:start_idx + chunk_size]
        placeholders, params = [], {}
        for r_idx, row in enumerate(chunk.itertuples(index=False)):
            row_params = {}
            for col in columns:
                val = getattr(row, col)
                if pd.isna(val):
                    val = None
                elif isinstance(val, pd.Timestamp):
                    val = str(val)
                row_params[f"{col}_{r_idx}"] = val
            params.update(row_params)
            row_ph = ", ".join(f":{col}_{r_idx}" for col in columns)
            placeholders.append(f"({row_ph})")

        values_clause = ", ".join(placeholders)

        if dialect == "sqlite":
            sql = text(f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES {values_clause}")
        else:
            sql = text(
                f"INSERT INTO {table} ({col_list}) VALUES {values_clause} "
                f"ON CONFLICT (TransactionId) DO NOTHING"
            )

        result = conn.execute(sql, params)
        total_written += result.rowcount if result.rowcount is not None else len(chunk)

    return total_written


@task(name="Incremental Load Expenses (Fact)", **RETRY_POLICY)
def process_expenses_incremental(watermark: str) -> str:
    logger = get_run_logger()
    logger.info(f"Executing INCREMENTAL LOAD for Expenses.  Watermark: {watermark}")

    _ensure_expense_tables()

    raw = pd.read_csv("data/raw_expenses.csv")
    _validate_schema(raw, EXPENSE_SCHEMA, "raw_expenses")

    raw["TransactionDate"] = pd.to_datetime(raw["TransactionDate"])
    incremental = raw[raw["TransactionDate"] > pd.Timestamp(watermark)].copy()

    if incremental.empty:
        logger.info("No new expenses since watermark.  Nothing to do.")
        return watermark

    with engine.begin() as conn:
        bronze_written = _insert_ignore(incremental, "bronze_expenses", conn)
    logger.info(f"Bronze: {bronze_written:,} rows inserted  ({len(incremental) - bronze_written:,} duplicates skipped)")

    valid = incremental[incremental["Amount"] > 0].copy()
    for col in GRANT_RESTRICTIONS:
        valid[col] = valid[col].where(valid[col].isna(), _clean_restriction_column(valid[col]))

    with engine.begin() as conn:
        silver_written = _insert_ignore(valid, "silver_expenses", conn)
    logger.info(f"Silver: {silver_written:,} rows inserted  ({len(valid) - silver_written:,} duplicates skipped)")

    new_max: str = incremental["TransactionDate"].max().strftime("%Y-%m-%d")
    logger.info(f"Incremental load complete — new watermark: {new_max}")
    return new_max


# ─────────────────────────────────────────────────────────────────────────────
# 3. Data Quality — Soda Core gate
# ─────────────────────────────────────────────────────────────────────────────
#
# Soda Core has no first-class SQLite driver. Since this exercise's
# persistence layer IS SQLite (not a local stand-in for something else),
# the bridge is explicit rather than hidden: mirror the exact tables each
# scan needs into a DuckDB file immediately beforehand, then point Soda at
# that. DuckDB reads pandas DataFrames natively and writes a real on-disk
# file Soda can connect to like any other data source — no separate
# warehouse, no network dependency, and the mirrored tables are a faithful
# byte-for-byte copy of what's actually in SQLite at scan time.
#
# In a Snowflake-backed deployment this whole mirror step disappears —
# soda-core-snowflake connects directly, same configuration.yml pattern,
# no bridge needed. It exists here specifically because the underlying
# store is SQLite.

import duckdb

SODA_MIRROR_PATH = "data/soda_mirror.duckdb"

# Columns that need to be real datetimes in the mirror, not raw strings —
# required for Soda's freshness() check specifically; every other check
# works fine on strings, freshness does not.
_DATE_COLUMNS_BY_TABLE = {
    "silver_grants":   ["StartDate", "EndDate"],
    "silver_expenses": ["TransactionDate"],
}


def _mirror_tables_to_duckdb(table_names: list[str]) -> None:
    """
    Copies each named table from the live SQLite DB into a fresh DuckDB
    file Soda Core can scan. Runs immediately before each scan so the
    mirror reflects the current state, not a stale snapshot.

    dtype_backend='numpy_nullable' is required, not optional: pandas 3.x
    defaults string columns to its new pyarrow-backed "str" dtype, which
    DuckDB 1.x's pandas scanner does not recognize (raises
    NotImplementedException: Data type 'str' not recognized). Forcing
    numpy_nullable keeps columns as numpy/nullable dtypes DuckDB can read.
    """
    dcon = duckdb.connect(SODA_MIRROR_PATH)
    for table in table_names:
        df = pd.read_sql(f"SELECT * FROM {table}", engine, dtype_backend="numpy_nullable")
        for col in _DATE_COLUMNS_BY_TABLE.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        dcon.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM df")
    dcon.close()


def _run_soda_scan(checks_yaml_path: str, scan_name: str) -> Scan:
    """
    Shared scan runner. Not a @task itself — the two callers below wrap it
    with different failure semantics (hard gate vs. logged canary).
    """
    scan = Scan()
    scan.set_scan_definition_name(scan_name)
    scan.set_data_source_name("oaf_warehouse")
    scan.add_configuration_yaml_file(file_path="soda/configuration.yml")
    scan.add_sodacl_yaml_file(file_path=checks_yaml_path)
    scan.execute()
    return scan


@task(name="Silver Data Quality Gate (Soda)", **RETRY_POLICY)
def run_silver_quality_gate() -> None:
    """
    Runs BEFORE the allocator ever reads silver_grants / silver_expenses.
    This is the circuit breaker: a failing check here halts the pipeline,
    same as _validate_business_rules, but declared in SodaCL rather than
    Python — reviewable and editable by Finance/analysts without touching
    the codebase, and runnable standalone in CI against fixture data.
    """
    logger = get_run_logger()
    _mirror_tables_to_duckdb(["silver_grants", "silver_expenses"])
    scan = _run_soda_scan("soda/checks/silver_checks.yml", scan_name="silver_gate")

    logger.info(scan.get_logs_text())

    if scan.has_check_fails():
        raise RuntimeError(
            f"[DATA CONTRACT VIOLATED] Silver quality gate failed:\n"
            f"{scan.get_checks_fail_text()}"
        )
    if scan.has_check_warns():
        logger.warning(f"Silver quality gate has warnings:\n{scan.get_checks_warn_or_fail_text()}")


@task(name="Gold Referential Integrity Canary (Soda)", **RETRY_POLICY)
def run_gold_quality_canary(period_id: str) -> None:
    """
    Runs AFTER load_gold_period, as a post-write check — not a gate that
    blocks anything (the swap already happened), but a canary that should
    never fire given core_allocator only draws from grants_df. If it does
    fire, that's a signal to investigate the swap/period-scoping logic
    itself, not routine data cleanup — so this logs loudly rather than
    raising, and the alert should page an engineer, not Finance.
    """
    logger = get_run_logger()
    _mirror_tables_to_duckdb(
        ["gold_allocations", "silver_grants", "accounting_periods", "gold_grant_balances"]
    )
    scan = _run_soda_scan("soda/checks/gold_checks.yml", scan_name=f"gold_canary_{period_id}")

    logger.info(scan.get_logs_text())

    if scan.has_check_fails():
        logger.error(
            f"[STRUCTURAL INVARIANT VIOLATED] Gold canary failed for "
            f"period {period_id} — this should be impossible:\n"
            f"{scan.get_checks_fail_text()}"
        )
        # Deliberately not raising: the write already committed, and halting
        # the flow here wouldn't undo it. This should page on-call directly
        # (wire to your alerting integration) rather than fail the run.


# ─────────────────────────────────────────────────────────────────────────────
# 4. Period control
# ─────────────────────────────────────────────────────────────────────────────

@task(name="Initialize Period Control Table", **RETRY_POLICY)
def initialize_period_table() -> None:
    """
    Seeds one row per PERIOD_DEFINITIONS entry as OPEN if it doesn't already
    exist. Idempotent — safe to call on every flow run, same pattern as
    initialize_watermark_table.
    """
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS accounting_periods (
                PeriodId   TEXT PRIMARY KEY,
                Status     TEXT NOT NULL DEFAULT 'OPEN',
                ClosedAt   TIMESTAMP,
                ClosedBy   TEXT
            )
        """))
        for period_id in PERIOD_DEFINITIONS:
            conn.execute(text("""
                INSERT OR IGNORE INTO accounting_periods (PeriodId, Status)
                VALUES (:p, 'OPEN')
            """), {"p": period_id})


@task(name="Get Period Status", **RETRY_POLICY)
def get_period_status(period_id: str) -> str:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT Status FROM accounting_periods WHERE PeriodId = :p"),
            {"p": period_id},
        ).fetchone()
    if row is None:
        raise ValueError(
            f"PeriodId '{period_id}' has no accounting_periods row — "
            f"run initialize_period_table() first."
        )
    return row[0]


@task(name="Get Opening Balances", **RETRY_POLICY)
def get_opening_balances(period_id: str) -> Optional[dict]:
    """
    None only for the first-ever period (no predecessor in PERIOD_DEFINITIONS).
    For every later period, reads the prior period's certified RemainingAmount
    from gold_grant_balances — and requires that prior period to be CLOSED
    before trusting its numbers as a seed.
    """
    period_ids = list(PERIOD_DEFINITIONS.keys())
    idx = period_ids.index(period_id)
    if idx == 0:
        return None

    prior_period_id = period_ids[idx - 1]
    prior_status = get_period_status.fn(prior_period_id)
    if prior_status != "CLOSED":
        raise RuntimeError(
            f"Cannot seed {period_id}: prior period {prior_period_id} is "
            f"'{prior_status}', not CLOSED. Close it first."
        )

    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT GrantCode, RemainingAmount FROM gold_grant_balances WHERE PeriodId = :p"),
            {"p": prior_period_id},
        ).fetchall()

    if not rows:
        raise RuntimeError(
            f"Prior period {prior_period_id} is CLOSED but has no "
            f"gold_grant_balances rows — cannot seed {period_id}."
        )
    return {r[0]: r[1] for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Business logic: Gold allocation
# ─────────────────────────────────────────────────────────────────────────────

@task(name="Run Period Allocation", **RETRY_POLICY)
def run_period_allocation(period_id: str, opening_balances: Optional[dict]) -> dict:
    """
    Pull the FULL bounded window for this period (not "since watermark" —
    the allocator's window is always PERIOD_DEFINITIONS[period_id]) and run
    the deterministic allocation engine, seeded from opening_balances.
    """
    logger = get_run_logger()
    start, end = PERIOD_DEFINITIONS[period_id]
    logger.info(f"Running allocation for period {period_id}  [{start} .. {end}]")

    expenses_df = pd.read_sql(
        text("SELECT * FROM silver_expenses WHERE TransactionDate BETWEEN :s AND :e"),
        engine,
        params={"s": str(start), "e": str(end)},
    )
    grants_df = pd.read_sql(text("SELECT * FROM silver_grants"), engine)

    if expenses_df.empty or grants_df.empty:
        logger.warning("Silver tables are empty for this period — skipping allocation.")
        return {}

    expenses_df["TransactionDate"] = pd.to_datetime(expenses_df["TransactionDate"])
    grants_df["StartDate"]         = pd.to_datetime(grants_df["StartDate"])
    grants_df["EndDate"]           = pd.to_datetime(grants_df["EndDate"])

    _validate_business_rules(grants_df, expenses_df)

    result = allocate(
        grants_df, expenses_df,
        period_id=period_id,
        opening_balances=opening_balances,
    )

    _assert_reconciliation(result, expenses_df)
    return result


def _assert_reconciliation(result: dict, expenses_df: pd.DataFrame) -> None:
    alloc   = result.get("allocations", pd.DataFrame())
    unalloc = result.get("unallocated",  pd.DataFrame())
    bal     = result.get("grant_balances", pd.DataFrame())

    total_in      = expenses_df["Amount"].sum()
    total_alloc   = alloc["AllocatedAmount"].sum()   if not alloc.empty   else 0.0
    total_unalloc = unalloc["UnallocatedAmount"].sum() if not unalloc.empty else 0.0
    delta         = abs(total_in - total_alloc - total_unalloc)

    if delta > 0.01:
        raise ValueError(
            f"[RECONCILIATION FAILED] Σ expenses={total_in:.2f}  "
            f"Σ allocated={total_alloc:.2f}  Σ unallocated={total_unalloc:.2f}  "
            f"Δ={delta:.4f}"
        )
    if not bal.empty and (bal["RemainingAmount"] < -1e-6).any():
        raise ValueError("[RECONCILIATION FAILED] Negative grant balance detected.")

    covered = set()
    if not alloc.empty:   covered |= set(alloc["TransactionId"])
    if not unalloc.empty: covered |= set(unalloc["TransactionId"])
    missing = set(expenses_df["TransactionId"]) - covered
    if missing:
        raise ValueError(f"[RECONCILIATION FAILED] Transactions not in any output: {missing}")

    get_run_logger().info(
        f"Reconciliation OK — allocated {total_alloc:,.2f}  "
        f"unallocated {total_unalloc:,.2f}  Δ={delta:.4f}"
    )


@task(name="Gold Atomic Swap (Period-Scoped)", **RETRY_POLICY)
def load_gold_period(results: dict, period_id: str) -> None:
    """
    Wipe THIS PERIOD's Gold rows only and atomically swap in the newly
    calculated outputs. Refuses to run against a CLOSED period — this is
    the guard that was structurally missing before: previously nothing
    stopped a scheduled run from overwriting a locked period's numbers.
    """
    logger = get_run_logger()

    status = get_period_status.fn(period_id)
    if status == "CLOSED":
        raise RuntimeError(
            f"PeriodId '{period_id}' is CLOSED — refusing atomic swap. "
            f"Post-close corrections must go through the reversal path, "
            f"not this task."
        )

    if not results:
        logger.warning("No allocation results — Gold layer unchanged.")
        return

    allocations_df = results.get("allocations")
    unallocated_df = results.get("unallocated")
    balances_df    = results.get("grant_balances")

    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()

    with engine.begin() as conn:
        if "gold_allocations" in existing_tables:
            conn.execute(text("DELETE FROM gold_allocations WHERE PeriodId = :p"), {"p": period_id})
        if "gold_unallocated" in existing_tables:
            conn.execute(text("DELETE FROM gold_unallocated WHERE PeriodId = :p"), {"p": period_id})
        if "gold_grant_balances" in existing_tables:
            conn.execute(text("DELETE FROM gold_grant_balances WHERE PeriodId = :p"), {"p": period_id})

        if allocations_df is not None and not allocations_df.empty:
            allocations_df.to_sql("gold_allocations", conn, if_exists="append", index=False)
        if unallocated_df is not None and not unallocated_df.empty:
            unallocated_df.to_sql("gold_unallocated", conn, if_exists="append", index=False)
        if balances_df is not None and not balances_df.empty:
            balances_df.to_sql("gold_grant_balances", conn, if_exists="append", index=False)

    n_alloc   = len(allocations_df) if allocations_df is not None else 0
    n_unalloc = len(unallocated_df) if unallocated_df is not None else 0
    logger.info(f"Gold[{period_id}] updated — {n_alloc:,} allocation rows  ·  {n_unalloc:,} unallocated rows")


@task(name="Close Period", **RETRY_POLICY)
def close_period(period_id: str, closed_by: str) -> None:
    """
    Deliberate, Finance-triggered operation — not part of the scheduled
    flow. Flips status to CLOSED only after the atomic swap for this
    period's final recompute has already succeeded, and exports the three
    certified files.
    """
    logger = get_run_logger()
    status = get_period_status.fn(period_id)
    if status == "CLOSED":
        raise RuntimeError(f"PeriodId '{period_id}' is already CLOSED.")

    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE accounting_periods
            SET Status = 'CLOSED', ClosedAt = CURRENT_TIMESTAMP, ClosedBy = :who
            WHERE PeriodId = :p
        """), {"who": closed_by, "p": period_id})

    for table, filename in [
        ("gold_allocations", f"exports/allocation_{period_id}.csv"),
        ("gold_unallocated", f"exports/unallocated_{period_id}.csv"),
        ("gold_grant_balances", f"exports/grant_balance_{period_id}.csv"),
    ]:
        df = pd.read_sql(text(f"SELECT * FROM {table} WHERE PeriodId = :p"), engine, params={"p": period_id})
        df.to_csv(filename, index=False)

    logger.info(f"Period {period_id} CLOSED by {closed_by}. Exports written.")


def _snapshot_period_state(period_id: str, tag: str) -> None:
    """
    Exports the three Gold tables for one period to a tagged snapshot file
    — the before/after evidence for a change request. Not the exports/
    certified files (those are close_period's output); these live
    separately under exports/snapshots/ so a correction's audit trail
    never overwrites or gets confused with the official closed-period
    export.
    """
    os.makedirs("exports/snapshots", exist_ok=True)
    for table in ["gold_allocations", "gold_unallocated", "gold_grant_balances"]:
        df = pd.read_sql(
            text(f"SELECT * FROM {table} WHERE PeriodId = :p"),
            engine, params={"p": period_id},
        )
        df.to_csv(f"exports/snapshots/{tag}_{table}_{period_id}.csv", index=False)


def reopen_and_cascade(period_id: str, reason: str, corrected_by: str) -> None:
    """
    Corrects a CLOSED period and propagates the correction forward through
    every later period, reusing the existing open-period recompute and
    close-period functions rather than introducing new write logic.

    Sequence
    --------
    1. Snapshot BEFORE state for period_id and every period after it.
    2. Reopen period_id (status -> OPEN) and recompute it via the normal
       run_period_allocation + load_gold_period path — same code, same
       guards, as any other open-period run.
    3. Reclose period_id (this is what makes its new closing balance
       available to seed the next period).
    4. Walk forward through every later PERIOD_DEFINITIONS entry in order:
         - if it was OPEN, just recompute it (new opening balance flows
           in automatically via get_opening_balances).
         - if it was CLOSED, it must be reopened first — load_gold_period
           refuses to write to a closed period — then recomputed, then
           reclosed. Skipping this would leave it silently stale, exactly
           the risk this whole function exists to avoid.
    5. Snapshot AFTER state for period_id and every period after it. The
       diff between BEFORE and AFTER *is* the change-request evidence —
       no reversal-row ledger needed for that purpose.

    This does not restrict correction to only the most recently closed
    period — reopening ANY closed period is allowed, and every period
    after it cascades automatically, however many there are.
    """
    logger = get_run_logger()
    all_period_ids = list(PERIOD_DEFINITIONS.keys())
    if period_id not in all_period_ids:
        raise ValueError(f"Unknown period_id '{period_id}'.")

    idx = all_period_ids.index(period_id)
    affected_period_ids = all_period_ids[idx:]  # period_id and everything after it

    status = get_period_status.fn(period_id)
    if status != "CLOSED":
        raise RuntimeError(
            f"'{period_id}' is '{status}', not CLOSED — nothing to reopen. "
            f"Use the normal open-period run for corrections to an open period."
        )

    logger.info(
        f"Reopening '{period_id}' for correction — reason: {reason!r}, by: {corrected_by}. "
        f"Cascade will touch: {affected_period_ids}"
    )

    for p in affected_period_ids:
        _snapshot_period_state(p, tag="before")

    # Reopen and recompute the corrected period itself.
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE accounting_periods SET Status='OPEN', ClosedAt=NULL, ClosedBy=NULL WHERE PeriodId=:p"),
            {"p": period_id},
        )
    opening = get_opening_balances.fn(period_id)
    result = run_period_allocation.fn(period_id, opening)
    load_gold_period.fn(result, period_id)
    close_period.fn(period_id, closed_by=corrected_by)

    # Cascade forward: recompute every later period, reopening first if needed.
    for p in affected_period_ids[1:]:
        p_status = get_period_status.fn(p)
        was_closed = p_status == "CLOSED"

        if was_closed:
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE accounting_periods SET Status='OPEN', ClosedAt=NULL, ClosedBy=NULL WHERE PeriodId=:p"),
                    {"p": p},
                )

        opening_p = get_opening_balances.fn(p)
        result_p = run_period_allocation.fn(p, opening_p)
        load_gold_period.fn(result_p, p)

        if was_closed:
            close_period.fn(p, closed_by=corrected_by)
            logger.info(f"Cascaded correction through '{p}' (was CLOSED — reopened, recomputed, reclosed).")
        else:
            logger.info(f"Cascaded correction through '{p}' (was OPEN — recomputed only).")

    for p in affected_period_ids:
        _snapshot_period_state(p, tag="after")

    logger.info(
        f"Cascade complete. BEFORE/AFTER snapshots for {affected_period_ids} "
        f"written to exports/snapshots/ — diff them to see exactly what changed."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. Orchestration flow
# ─────────────────────────────────────────────────────────────────────────────

@flow(
    name="End-to-End Grant Allocation Pipeline",
    on_completion=[_notify_success],
    on_failure=[_notify_failure],
)
def end_to_end_pipeline(period_id: str = "2027"):
    """
    Orchestration order
    -------------------
    1. Grants full load           (dimension — always runs first)
    2. Expenses incremental       (fact, Bronze/Silver only — watermark-gated)
    3. Advance watermark          (only after Bronze/Silver write succeeds)
    4. Silver quality gate (Soda) (hard stop — allocator never runs on
                                    data that fails this)
    5. Period control init        (idempotent — creates accounting_periods rows)
    6. Period status check        (fails fast if period_id is CLOSED)
    7. Opening balances           (None for first period; seeded from prior
                                    CLOSED period's Gold otherwise)
    8. Run period allocation      (full recompute of the bounded window)
    9. Gold atomic swap           (period-scoped delete + insert, re-checks
                                    CLOSED status internally as a second gate)
    10. Gold quality canary (Soda) (logs/pages, does not halt — the write
                                    already committed by this point)

    close_period() is intentionally NOT called here — it's a separate,
    Finance-triggered flow, not something the scheduled run decides on its
    own.
    """
    logger = get_run_logger()
    logger.info(f"Pipeline started — period_id={period_id}")

    grants_loaded = process_grants_full_load()

    _wm_init = initialize_watermark_table()
    current_wm: str = get_watermark("raw_expenses", wait_for=[_wm_init])
    new_wm: str = process_expenses_incremental(current_wm)
    wm_advanced = update_watermark("raw_expenses", new_wm)

    dq_gate = run_silver_quality_gate(wait_for=[grants_loaded, wm_advanced])

    period_init = initialize_period_table()
    status = get_period_status(period_id, wait_for=[period_init])
    if status == "CLOSED":
        raise RuntimeError(
            f"Period '{period_id}' is CLOSED — this flow only performs "
            f"open-period runs. Use the reversal path for corrections."
        )

    opening = get_opening_balances(period_id)

    results: dict = run_period_allocation(
        period_id, opening,
        wait_for=[grants_loaded, wm_advanced, dq_gate],
    )

    gold_swap = load_gold_period(results, period_id)

    run_gold_quality_canary(period_id, wait_for=[gold_swap])

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    end_to_end_pipeline(period_id="2021_2026")