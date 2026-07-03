"""
reset_pipeline_db.py — full reset for a clean re-ingest.
Drops Bronze/Silver/Gold and clears the expense watermark so the next
pipeline run rebuilds everything from the raw CSVs, picking up the
_clean_restriction_column fix.
Run with: python reset_pipeline_db.py
"""
import sqlite3

conn = sqlite3.connect("my_local_database.db")

tables_to_drop = [
    "bronze_grants", "silver_grants",
    "bronze_expenses", "silver_expenses",
    "gold_allocations", "gold_unallocated", "gold_grant_balances",
]
for t in tables_to_drop:
    conn.execute(f"DROP TABLE IF EXISTS {t}")

conn.execute("DELETE FROM etl_watermarks WHERE TableName = 'raw_expenses'")

conn.commit()
conn.close()
print("Full reset complete — Bronze/Silver/Gold will rebuild from raw CSVs on next run.")
