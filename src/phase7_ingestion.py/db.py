"""
db.py
----
Database connection and helper function
All four ingestion scripts import from here

Instead of every script writing its own SQL, they all
call these shared functions. If the database schema
changes, you fix it here once - not in four places.
"""

import hashlib
import json
import re
import os
import sys

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

# ── Engine and Session ──
# create_engine() sets up the connection pool to PostgreSQL.
# echo=False means SQLAlchemy will not print every SQL
# statement to the console (set to True to debug queries).
engine  = create_engine(os.getenv("POSTGRES_URL"), echo=False)

# sessionmaker() creates a factory for database sessions.
# A session is one "conversation" with the database —
# you do a bunch of reads and writes, then commit or rollback.
Session = sessionmaker(bind=engine)

def get_session():
    """Return a new database session."""
    return Session()

# ── Paper ID generation ──
# This is the most important function in the whole file.
# Every paper needs one stable ID regardless of which source it came from. 
# The same paper appears in OpenAlex, Semantic Scholar, AND PubMed simultaneously
# with different internal IDs. We resolve them to one
# canonical ID using this priority:
#
#   1. DOI   — globally unique, most reliable
#   2. arXiv ID — unique for arXiv papers
#   3. Hash  — fallback: hash of title + author + year
#
# Example:
#   OpenAlex sees paper as "W2741809807"
#   Semantic Scholar sees same paper as "abc123def456"
#   PubMed sees same paper as PMID 29355075
#   All three have DOI "10.1038/nature25175"
#   → All three become "doi:10.1038/nature25175" in our DB
#   → Deduplication happens automatically

def make_paper_id(doi=None, arxiv_id=None, title=None, first_author=None, year=None):
    """
    Generate a canonical paper ID.
    Same paper from different sources -> same ID
    """
    if doi:
        # Normalize DOI: lowercase, strip URL prefix
        clean = doi.lower().strip()
        clean = clean.replace("https://doi.org/", "")
        clean = clean.replace("https://doi.org/", "")
        return f"doi:{clean}"
    
    if arxiv_id:
        # arXiv IDs look like "2304.12345" or "cs/0301001"
        return f"arxiv:{arxiv_id.strip()}"
    
    # fallback : hash of normalized title + author + year
    key = f"{_normalize(title)}_{first_author or ''}_{year or ''}"
    return f"hash:{hashlib.md5(key.encode()).hexdigest()[:16]}"

def _normalize(text):
    """
    Normalize text for comparison. Lowercase, remove punctuation, collapse spaces.
    Used so "Machine Learning!" and "machine learning" produce the same hash
    """
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r'[^a-z0=9\s]', '', t)
    t = re.sub(r'\s+',' ',t)
    return t.strip()

def compute_completeness(paper_dict):
    """
    Score from 0.0 to 1.0 indicating how complete a paper record is. Used later in regression as
    a feature - more complete papers are more discoverable and tend to get more citations.

    Checks five optional fields:
    abstract, doi, authors, fields, year
    Each present field adds 0.2 to the score.
    """
    optional = ["abstract","doi","authors","fields","year"]
    filled = sum(1 for f in optional if paper_dict.get(f))
    return round(filled / len(optional), 2)

def upsert_paper(session, paper_dict):
    """
    Insert or update a paper. Uses cast() instead of
    inline ::jsonb cast to avoid parameter style conflicts
    between SQLAlchemy text() and psycopg2.
    """
    sql = text("""
        INSERT INTO papers (
            paper_id, source, source_id, doi, title, abstract,
            year, authors, fields, language, has_full_text,
            completeness_score, num_citations
        ) VALUES (
            :paper_id, :source, :source_id, :doi, :title, :abstract,
            :year,
            cast(:authors as jsonb),
            cast(:fields as jsonb),
            :language, :has_full_text, :completeness_score, :num_citations
        )
        ON CONFLICT (paper_id) DO UPDATE SET
            abstract           = COALESCE(EXCLUDED.abstract,
                                          papers.abstract),
            doi                = COALESCE(EXCLUDED.doi,
                                          papers.doi),
            num_citations      = GREATEST(EXCLUDED.num_citations,
                                          papers.num_citations),
            completeness_score = GREATEST(EXCLUDED.completeness_score,
                                          papers.completeness_score),
            updated_at         = NOW()
        RETURNING paper_id
    """)

    result = session.execute(sql, {
        **paper_dict,
        "authors": json.dumps(paper_dict.get("authors", [])),
        "fields":  json.dumps(paper_dict.get("fields",  [])),
    })
    session.commit()
    return result.fetchone()[0]

def insert_citation(session, citing_id, cited_id, edge_type="hard"):
    """
    Insert a citation edge between two papers.
    citing_id → cited_id means:
      "paper citing_id cites paper cited_id"

    ON CONFLICT DO NOTHING means if this edge already
    exists (from another source), silently skip it.
    No error, no duplicate.

    edge_type:
      "hard" = explicit citation link from the paper's
               reference list
      "soft" = inferred from semantic similarity
               (added in Phase 8)
    """
    sql = text("""
        INSERT INTO citations
            (citing_paper_id, cited_paper_id, edge_type)
        VALUES
            (:citing, :cited, :etype)
        ON CONFLICT DO NOTHING
    """)
    session.execute(sql, {
        "citing": citing_id,
        "cited": cited_id,
        "etype": edge_type
    })
    session.commit()



