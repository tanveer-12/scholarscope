"""
resolve_citations.py
--------------------
Resolves raw citation IDs stored during ingestion
to canonical paper_ids so cross-paper JOIN queries work.

The problem:
  During ingestion, citation edges were stored as:
    citing_paper_id = "doi:10.xxx"  (canonical)
    cited_paper_id  = "abc123def"   (raw SS internal ID)

  But papers table only has canonical IDs.
  So JOIN on cited_paper_id finds nothing.

The fix:
  Build a lookup table mapping every raw source ID
  to its canonical paper_id. Then update citations
  table to replace raw IDs with canonical ones.
  Citations pointing to papers not in our DB are deleted
  (they are external references we cannot resolve).
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import text
from db import get_session

def run():
    session = get_session()

    print("Building source ID → canonical ID lookup...")

    # Step 1: build a mapping of every known source_id
    # to its canonical paper_id from the papers table
    # source_id is the raw ID from each API
    # (SS paperId, PubMed PMID, arXiv ID, OpenAlex ID)
    session.execute(text("""
        INSERT INTO paper_id_map (source, source_id, canonical_id)
        SELECT source, source_id, paper_id
        FROM papers
        ON CONFLICT (source, source_id) DO NOTHING
    """))
    session.commit()

    # Also map DOIs since citations often reference by DOI
    session.execute(text("""
        INSERT INTO paper_id_map (source, source_id, canonical_id)
        SELECT source, doi, paper_id
        FROM papers
        WHERE doi IS NOT NULL
        ON CONFLICT (source, source_id) DO NOTHING
    """))
    session.commit()

    # Count what we have
    result = session.execute(text(
        "SELECT COUNT(*) FROM paper_id_map"
    ))
    map_count = result.fetchone()[0]
    print(f"ID mappings built: {map_count:,}")

    print("\nResolving citation edges...")

    # Step 2: update citations where cited_paper_id
    # matches a known source_id in our mapping
    result = session.execute(text("""
        UPDATE citations c
        SET cited_paper_id = m.canonical_id
        FROM paper_id_map m
        WHERE c.cited_paper_id = m.source_id
          AND c.cited_paper_id != m.canonical_id
    """))
    session.commit()
    resolved = result.rowcount
    print(f"Citation edges resolved: {resolved:,}")

    # Step 3: delete citations that still point to
    # IDs not in our papers table — these are external
    # references we cannot resolve
    result = session.execute(text("""
        DELETE FROM citations
        WHERE cited_paper_id NOT IN (
            SELECT paper_id FROM papers
        )
    """))
    session.commit()
    deleted = result.rowcount
    print(f"Unresolvable citations deleted: {deleted:,}")

    # Step 4: also clean up citing side
    result = session.execute(text("""
        DELETE FROM citations
        WHERE citing_paper_id NOT IN (
            SELECT paper_id FROM papers
        )
    """))
    session.commit()
    print(f"Orphaned citing edges deleted: {result.rowcount:,}")

    # Final count
    result = session.execute(text(
        "SELECT COUNT(*) FROM citations"
    ))
    final = result.fetchone()[0]
    print(f"\nFinal internal citation edges: {final:,}")

    if final == 0:
        print("\nNote: 0 internal edges is expected if all citations")
        print("point to papers outside your current corpus.")
        print("Phase 8 soft edges will connect papers semantically")
        print("regardless of explicit citation links.")
    else:
        print("\nRun the cross-source JOIN query again in pgAdmin")
        print("to see which sources cite each other.")

    session.close()

if __name__ == "__main__":
    run()