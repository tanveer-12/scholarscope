"""
ingest_semantic_scholar.py
--------------------------
Uses the BULK search endpoint instead of the regular search endpoint.

Key differences from before:
  - URL: /paper/search/bulk  (not /paper/search)
  - Pagination: token-based  (not offset-based)
    The API returns a "token" with each response.
    You pass that token in the next request to get
    the next page. No offset limit problem.
  - Filter: query parameter with field-specific terms
    instead of a fieldsOfStudy parameter that caused 400s.
"""

import requests
import time
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tqdm import tqdm
from config import (TARGET_PER_SOURCE, SEMANTIC_SCHOLAR_API_KEY)
from db import (get_session, make_paper_id, upsert_paper,
                insert_citation, compute_completeness)

# Bulk search endpoint — different from regular search
BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"

FIELDS = ",".join([
    "paperId",
    "externalIds",
    "title",
    "abstract",
    "year",
    "authors",
    "fieldsOfStudy",
    "citationCount",
])

HEADERS = ({"x-api-key": SEMANTIC_SCHOLAR_API_KEY}
           if SEMANTIC_SCHOLAR_API_KEY else {})

DELAY = 0.5   # bulk endpoint is more generous — 0.5s is safe

# Search queries per domain — plain language works well
# with the bulk endpoint which searches title+abstract
SS_QUERIES = [
    ("computer science", "machine learning deep learning neural network"),
    ("biology",          "genomics protein structure gene expression"),
    ("medicine",         "clinical trial patient treatment disease"),
    ("economics",        "economic policy market labor income"),
    ("physics",          "quantum mechanics particle physics condensed matter"),
]


def fetch_papers(query, max_papers):
    """
    Generator yielding raw paper dicts using bulk search.

    Bulk search uses TOKEN pagination:
      - First request: no token, just query
      - Response includes a "token" field
      - Next request passes that token to get next page
      - When response has no token, you're at the end

    This completely avoids the offset limit problem.
    """
    token   = None
    fetched = 0

    while fetched < max_papers:
        params = {
            "query":  query,
            "fields": FIELDS,
            "limit":  min(100, max_papers - fetched),
            "year": "2018-",
            "sort": "publicationDate:desc",
        }

        # Add token for pagination after first request
        if token:
            params["token"] = token

        for attempt in range(3):
            resp = requests.get(BASE_URL, params=params,
                                headers=HEADERS, timeout=30)

            if resp.status_code == 429:
                wait = 30 * (attempt + 1)
                print(f"\n  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue

            if resp.status_code == 400:
                print(f"\n  400 error: {resp.text[:200]}")
                return   # stop this query entirely

            resp.raise_for_status()
            break
        else:
            print("\n  All retries failed. Stopping.")
            return

        data  = resp.json()
        items = data.get("data", [])

        if not items:
            break

        for item in items:
            if item.get("abstract"):
                yield item
                fetched += 1
                if fetched >= max_papers:
                    return

        # Get next page token
        # If no token in response, we've reached the end
        token = data.get("token")
        if not token:
            break

        time.sleep(DELAY)


def process_paper(raw):
    ext      = raw.get("externalIds") or {}
    doi      = (ext.get("DOI") or "").strip()
    arxiv_id = (ext.get("ArXiv") or "").strip()

    authors_raw  = raw.get("authors") or []
    first_author = ""
    if authors_raw:
        parts = authors_raw[0].get("name", "").split()
        first_author = parts[-1] if parts else ""

    paper_id = make_paper_id(
        doi=doi or None,
        arxiv_id=arxiv_id or None,
        title=raw.get("title"),
        first_author=first_author,
        year=raw.get("year"),
    )

    authors = [
        {"name": a.get("name", ""), "id": a.get("authorId", "")}
        for a in authors_raw
    ]

    fields = raw.get("fieldsOfStudy") or []

    paper = {
        "paper_id":          paper_id,
        "source":            "semantic_scholar",
        "source_id":         raw.get("paperId", ""),
        "doi":               doi or None,
        "title":             (raw.get("title") or "").strip(),
        "abstract":          raw.get("abstract"),
        "year":              raw.get("year"),
        "authors":           authors,
        "fields":            fields,
        "language":          "en",
        "has_full_text":     False,
        "num_citations":     raw.get("citationCount", 0),
    }
    paper["completeness_score"] = compute_completeness(paper)

    return paper, []


def run():
    session   = get_session()
    per_query = max(1, TARGET_PER_SOURCE // len(SS_QUERIES))

    total_inserted = 0

    for domain, query in SS_QUERIES:
        print(f"\nDomain '{domain}' — fetching {per_query} papers...")
        print(f"  Query: '{query}'")

        for raw in tqdm(fetch_papers(query, per_query),
                        total=per_query, desc=domain):

            paper, refs = process_paper(raw)

            if not paper["title"]:
                continue

            canon_id = upsert_paper(session, paper)

            for ref_id in refs:
                if ref_id:
                    insert_citation(session, canon_id,
                                    ref_id, edge_type="hard")
            total_inserted += 1

    session.close()
    print(f"\nSemantic Scholar done. Papers inserted/updated: {total_inserted}")


if __name__ == "__main__":
    run()