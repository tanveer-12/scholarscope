"""
ingest_openalex.py
------------------
Pulls papers from OpenAlex across multiple research domains.

HOW OPENALEX WORKS:
  OpenAlex is the open successor to Microsoft Academic Graph
  (the source of your original 36,823 paper dataset).
  It has 250 million papers organized into "concepts" —
  broad research areas tagged on each paper by ML classifiers.

  We filter by concept ID to get papers from specific domains
  (CS, Biology, Economics, etc.) so we get true cross-domain
  coverage instead of only biomedical papers.

HOW AUTHENTICATION WORKS:
  OpenAlex uses a Bearer token in the HTTP header:
    Authorization: Bearer YOUR_KEY_HERE
  This is different from URL parameters — the key travels
  in the request header, not visible in the URL.

PAGINATION:
  OpenAlex uses cursor-based pagination. Instead of
  "give me page 3", you say "give me the next page after
  this cursor token". The API returns a next_cursor value
  with each response that you pass back in the next request.
  This is more reliable for large datasets.

ABSTRACT FORMAT:
  OpenAlex does not return abstracts as plain text.
  Instead it returns an "inverted index" — a dictionary
  mapping each word to the list of positions it appears at.
  Example: {"The": [0], "paper": [1], "shows": [2]}
  We reconstruct the plain text from this in decode_abstract().
"""
import requests
import time
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (TARGET_PER_SOURCE, OPENALEX_CONCEPTS, OPENALEX_API_KEY)
from db import(get_session, make_paper_id, upsert_paper, insert_citation, compute_completeness)

# CONSTANTS
BASE_URL = "https://api.openalex.org/works"
PER_PAGE = 200  # max openalex allows per request

# Tell OpenAlex exactly which fields to return.
# Requesting only what we need is faster and uses
# less of our daily API quota.
FIELDS = ",".join([
    "id",                        # OpenAlex internal ID
    "doi",                       # DOI for cross-source matching
    "title",
    "abstract_inverted_index",   # abstract in inverted index format
    "publication_year",
    "authorships",               # list of author objects
    "concepts",                  # research area tags with scores
    "cited_by_count",            # total worldwide citation count
    "referenced_works",          # list of OpenAlex IDs this paper cites
])

# Auth header — Bearer token authentication
# If no key, send empty dict (no auth header)
HEADERS = ({"Authorization": f"Bearer {OPENALEX_API_KEY}"}
           if OPENALEX_API_KEY else {})


def decode_abstract(inverted_index):
    """
    Convert OpenAlex inverted index → plain text string.

    Input:  {"The": [0], "effect": [1], "of": [2, 8], ...}
    Output: "The effect of ... of ..."

    How it works:
      1. Flatten the dict into (position, word) pairs
      2. Sort by position
      3. Join words in order
    """
    if not inverted_index:
        return None
    # Flatten: for each word and its list of positions,
    # create one (position, word) tuple per position
    pairs = [
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    ]
    # Sort by position number, then join words
    return " ".join(word for _, word in sorted(pairs))


def fetch_concept_papers(concept_id, max_papers):
    """
    Generator that yields raw paper dicts from OpenAlex
    for one concept ID, up to max_papers total.

    Using a generator (yield) means papers are processed
    one at a time as they arrive — we do not wait for
    all 1000 papers to download before processing any.
    This keeps memory usage low.
    """
    cursor  = "*"    # "*" is OpenAlex's starting cursor
    fetched = 0

    while fetched < max_papers:
        # How many to request this page —
        # either full page or however many remain
        this_page = min(PER_PAGE, max_papers - fetched)

        params = {
            # Filter: must be in this concept AND have an abstract
            "filter":   f"concepts.id:{concept_id},has_abstract:true,"
                        f"publication_year:>2018",
            "per-page": this_page,
            "cursor":   cursor,
            "select":   FIELDS,
            "sort":     "publication_year:desc",   # newest first
        }

        resp = requests.get(BASE_URL, params=params,
                            headers=HEADERS, timeout=30)
        resp.raise_for_status()   # raises exception for 4xx/5xx
        data = resp.json()

        results = data.get("results", [])
        if not results:
            break   # no more papers for this concept

        for paper in results:
            yield paper
            fetched += 1
            if fetched >= max_papers:
                break

        # Get cursor for next page
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor:
            break   # no next page

        # Polite delay — 10 req/sec limit with API key
        time.sleep(0.1)


def process_paper(raw):
    """
    Transform one OpenAlex paper dict → unified schema dict.

    Returns:
      paper  — dict ready for upsert_paper()
      refs   — list of OpenAlex IDs this paper cites
               (used to insert citation edges)
    """
    # Extract and clean DOI
    doi = raw.get("doi", "") or ""
    doi = doi.replace("https://doi.org/", "").strip()

    # Build canonical paper ID
    paper_id = make_paper_id(
        doi=doi or None,
        title=raw.get("title"),
        year=raw.get("publication_year")
    )

    # Extract authors — each authorship object has
    # an "author" sub-object with display_name
    authors = [
        {
            "name": a.get("author", {}).get("display_name", ""),
            "id":   a.get("author", {}).get("id", "")
        }
        for a in (raw.get("authorships") or [])
    ]

    # Extract concept names where score > 0.3
    # Score is OpenAlex's confidence this paper
    # belongs to this concept (0.0 to 1.0)
    fields = [
        c.get("display_name", "")
        for c in (raw.get("concepts") or [])
        if c.get("score", 0) > 0.3
    ]

    # Decode abstract from inverted index
    abstract = decode_abstract(
        raw.get("abstract_inverted_index")
    )

    paper = {
        "paper_id":          paper_id,
        "source":            "openalex",
        "source_id":         raw.get("id", ""),
        "doi":               doi or None,
        "title":             (raw.get("title") or "").strip(),
        "abstract":          abstract,
        "year":              raw.get("publication_year"),
        "authors":           authors,
        "fields":            fields,
        "language":          "en",
        "has_full_text":     False,
        "num_citations":     raw.get("cited_by_count", 0),
    }
    paper["completeness_score"] = compute_completeness(paper)

    # referenced_works is a list of OpenAlex IDs
    # Example: ["https://openalex.org/W2741809807", ...]
    refs = raw.get("referenced_works") or []
    return paper, refs


def run():
    session = get_session()

    # Divide target evenly across all concepts
    per_concept = max(1, TARGET_PER_SOURCE // len(OPENALEX_CONCEPTS))

    total_inserted = 0

    for concept_id in OPENALEX_CONCEPTS:
        print(f"\nConcept {concept_id} — fetching {per_concept} papers...")

        for raw in tqdm(fetch_concept_papers(concept_id, per_concept),
                        total=per_concept, desc=concept_id):

            paper, refs = process_paper(raw)

            # Skip papers with no title
            if not paper["title"]:
                continue

            # Insert or update paper in database
            canon_id = upsert_paper(session, paper)

            # Insert citation edges
            # We store OpenAlex IDs as-is for now.
            # Cross-source resolution happens in dedup step.
            for ref_id in refs:
                if ref_id:  # skip empty strings
                    insert_citation(session, canon_id,
                                    ref_id, edge_type="hard")

            total_inserted += 1

    session.close()
    print(f"\nOpenAlex done. Papers inserted/updated: {total_inserted}")


if __name__ == "__main__":
    run()