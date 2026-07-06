"""
check_db.py — inspect a SQLite file's tables and row counts directly,
avoiding PowerShell's multi-line python -c quoting fragility entirely.

Run with: python check_db.py
"""
import sqlite3
import os

CANDIDATES = [
    "my_local_database.db",
    os.path.join("data", "my_local_database.db"),
]

for path in CANDIDATES:
    print(f"=== {path} ===")
    if not os.path.exists(path):
        print("  (file does not exist)")
        print()
        continue

    size = os.path.getsize(path)
    print(f"  size: {size:,} bytes")

    conn = sqlite3.connect(path)
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    if not tables:
        print("  no tables found")
    for (table_name,) in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"  {table_name}: {count} rows")
    conn.close()
    print()

print("=== Gold-layer snapshot tables (in my_local_database.db) ===")
if os.path.exists("my_local_database.db"):
    conn = sqlite3.connect("my_local_database.db")
    for snap_table in ["gold_allocations_snapshot", "gold_unallocated_snapshot", "gold_grant_balances_snapshot"]:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (snap_table,)
        ).fetchone()
        if not exists:
            print(f"  {snap_table}: DOES NOT EXIST YET")
            continue
        rows = conn.execute(f"SELECT PeriodId, SnapshotTag, COUNT(*), MAX(SnapshotAt) FROM {snap_table} GROUP BY PeriodId, SnapshotTag").fetchall()
        print(f"  {snap_table}:")
        if not rows:
            print("    (table exists but has zero rows)")
        for period_id, tag, count, latest in rows:
            print(f"    PeriodId={period_id}  Tag={tag}  rows={count}  latest={latest}")
    conn.close()
else:
    print("  my_local_database.db not found in current directory")