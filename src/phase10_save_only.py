"""
phase10_save_only.py
--------------------
Saves the already-computed Phase 10 bridge scores
from parquet into PostgreSQL.
Run this after manually adding the columns in pgAdmin.
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

    print("Loading bridge scores from parquet...")
    df = pd.read_parquet(
        "data/processed/phase10/papers_with_bridge_scores.parquet"
    )
    print(f"  Papers: {len(df):,}")
    print(f"  Bridge candidates (is_bridge_v2=True): "
          f"{df['is_bridge_v2'].sum():,}")

    rows = df[[
        "paper_id", "bridge_score_v2", "is_bridge_v2",
        "weighted_cds", "weighted_cd", "betweenness_v2",
        "community"
    ]].to_dict("records")

    print("\nUpdating PostgreSQL...")
    batch_size = 200

    for i in tqdm(range(0, len(rows), batch_size),
                  desc="Saving"):
        batch = rows[i:i + batch_size]
        for row in batch:
            session.execute(text("""
                UPDATE papers SET
                    bridge_score_v2 = :bs,
                    is_bridge_v2    = :ib,
                    weighted_cds    = :wcds,
                    weighted_cd     = :wcd,
                    betweenness_v2  = :bc,
                    community_id    = :comm
                WHERE paper_id = :pid
            """), {
                "bs":   float(row["bridge_score_v2"]),
                "ib":   bool(row["is_bridge_v2"]),
                "wcds": float(row["weighted_cds"]),
                "wcd":  float(row["weighted_cd"]),
                "bc":   float(row["betweenness_v2"]),
                "comm": str(row["community"]) if row["community"] else None,
                "pid":  row["paper_id"],
            })
        session.commit()

    # Verify
    result = session.execute(text("""
        SELECT
            COUNT(*)                                    AS total,
            SUM(CASE WHEN bridge_score_v2 > 0
                     THEN 1 ELSE 0 END)                AS with_score,
            SUM(CASE WHEN is_bridge_v2
                     THEN 1 ELSE 0 END)                AS bridge_candidates,
            ROUND(MAX(bridge_score_v2)::numeric, 6)    AS max_score
        FROM papers
    """))
    row = result.fetchone()
    print(f"\nPostgreSQL verification:")
    print(f"  Total papers:       {row[0]:,}")
    print(f"  With bridge score:  {row[1]:,}")
    print(f"  Bridge candidates:  {row[2]:,}")
    print(f"  Max score:          {row[3]}")

    session.close()
    print("\nDone. Phase 10 results saved to PostgreSQL.")
    print("Next: Phase 11 — regression model retraining")

if __name__ == "__main__":
    run()