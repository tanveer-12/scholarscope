"""
fix_community_ids.py
--------------------
Writes community_id from Phase 10 parquet back to PostgreSQL.
The phase10_save_only.py script saved bridge scores but
missed the community column.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "phase7_ingestion.py"))

import pandas as pd
from sqlalchemy import text
from tqdm import tqdm
from db import get_session

def run():
    session = get_session()

    print("Loading community assignments from parquet...")
    df = pd.read_parquet(
        "data/processed/phase10/papers_with_bridge_scores.parquet"
    )[["paper_id", "community"]]

    # Rename to match the DB column name
    df = df.rename(columns={"community": "community_id"})
    df = df[df["community_id"].notna()]
    print(f"  Papers with community assignment: {len(df):,}")

    print("Updating PostgreSQL...")
    rows = df.to_dict("records")

    for i in tqdm(range(0, len(rows), 200), desc="Saving"):
        batch = rows[i:i+200]
        for row in batch:
            session.execute(text("""
                UPDATE papers
                SET community_id = :comm
                WHERE paper_id   = :pid
            """), {
                "comm": str(row["community_id"]),
                "pid":  row["paper_id"],
            })
        session.commit()

    # Verify
    result = session.execute(text("""
        SELECT
            COUNT(*)                                               AS total,
            SUM(CASE WHEN community_id IS NOT NULL THEN 1 END)    AS with_community,
            SUM(CASE WHEN community_id IS NULL     THEN 1 END)    AS without_community
        FROM papers
    """)).fetchone()

    print(f"\nVerification:")
    print(f"  Total:            {result[0]:,}")
    print(f"  With community:   {result[1]:,}")
    print(f"  Without community: {result[2] or 0:,}")

    session.close()
    print("\nDone.")

if __name__ == "__main__":
    run()