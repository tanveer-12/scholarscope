"""
ingest_arxiv.py
---------------
Pulls papers from arXiv across multiple subject categories.

HOW ARXIV WORKS:
  arXiv is a preprint server — papers here are NOT
  peer reviewed. They are uploaded directly by authors.
  arXiv is crucial for CS and physics where preprints
  circulate widely before (or instead of) journal publication.

  The API returns XML in the Atom feed format.
  We parse this using Python's built-in xml.etree module.

NO CITATION DATA:
  arXiv does not provide reference lists.
  This is why arXiv papers currently appear disconnected
  in citation graphs — there are no hard edges for them.
  Phase 8 fixes this by adding SOFT EDGES based on
  semantic similarity of abstracts.

NO API KEY NEEDED:
  arXiv is fully open. The only requirement is a
  3-second delay between requests (politeness rule).

PAGINATION:
  arXiv uses start/offset pagination like Semantic Scholar.
  "Give me 100 papers starting at position 300"
"""

import requests
import time
import xml.etree.ElementTree as ET
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tqdm import tqdm
from config import TARGET_PER_SOURCE, ARXIV_CATEGORIES
from db import (get_session, make_paper_id, upsert_paper,
                compute_completeness)

BASE_URL = "http://export.arxiv.org/api/query"
DELAY    = 5.0   # arXiv explicitly asks for 3 seconds between requests

# XML namespace declarations in Atom feed.
# ElementTree needs these to find tags like <atom:title>
# and <arxiv:doi>. Without them, tag lookups fail.
NS = {
    "atom":  "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_category_papers(category, max_papers):
    """
    Generator that yields parsed arXiv XML entry elements.
    Handles 429 rate limits and read timeouts with
    exponential backoff — each retry waits longer than
    the previous one.
    """
    start   = 0
    fetched = 0
    batch   = min(50, max_papers)

    while fetched < max_papers:
        params = {
            "search_query": f"cat:{category}",
            "start":        start,
            "max_results":  min(batch, max_papers - fetched),
            "sortBy":       "submittedDate",
            "sortOrder":    "descending",
        }

        # Exponential backoff retry loop
        # Waits: 30s → 60s → 120s before giving up
        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                resp = requests.get(
                    BASE_URL,
                    params=params,
                    timeout=60        # increased from 30 to 60 seconds
                )

                if resp.status_code == 429:
                    wait = 30 * (2 ** attempt)  # 30, 60, 120 seconds
                    print(f"\n  Rate limited (attempt {attempt+1}). "
                          f"Waiting {wait}s...")
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                break   # success

            except requests.exceptions.Timeout:
                wait = 20 * (attempt + 1)   # 20, 40, 60 seconds
                print(f"\n  Timeout (attempt {attempt+1}). "
                      f"Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            except requests.exceptions.ConnectionError:
                wait = 30 * (attempt + 1)
                print(f"\n  Connection error (attempt {attempt+1}). "
                      f"Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

        else:
            # All attempts failed — skip this batch and move on
            print(f"\n  Skipping batch start={start} "
                  f"after {max_attempts} failed attempts.")
            start += batch
            time.sleep(10)
            continue

        try:
            root    = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", NS)
        except ET.ParseError:
            print(f"\n  XML parse error at start={start}. Skipping batch.")
            start += batch
            time.sleep(5)
            continue

        if not entries:
            break

        for entry in entries:
            yield entry
            fetched += 1
            if fetched >= max_papers:
                break

        start += len(entries)
        time.sleep(8)   # conservative — 8 seconds between batches

def _text(entry, tag, ns_key="atom"):
    """
    Helper: safely get text from an XML element.
    Returns None if the element does not exist.

    entry.find("atom:title", NS) finds the <title> tag
    inside this entry using the namespace mapping.
    """
    el = entry.find(f"{ns_key}:{tag}", NS)
    return el.text.strip() if el is not None and el.text else None


def parse_entry(entry):
    """
    Transform one arXiv XML entry → unified schema dict.

    arXiv paper ID looks like:
      https://arxiv.org/abs/2304.12345v1
    We extract just "2304.12345" as the arXiv ID.
    """
    arxiv_url = _text(entry, "id")
    arxiv_id  = None
    if arxiv_url and "/abs/" in arxiv_url:
        # Split on /abs/ and take the part after it
        # "2304.12345v1" → strip version → "2304.12345"
        arxiv_id = arxiv_url.split("/abs/")[-1]
        # Remove version suffix if present (v1, v2, etc.)
        arxiv_id = arxiv_id.split("v")[0]

    # DOI — arXiv papers sometimes have a DOI
    # when they were also published in a journal
    doi = None
    doi_el = entry.find("arxiv:doi", NS)
    if doi_el is not None and doi_el.text:
        doi = doi_el.text.strip()

    # Year from publication date "2023-04-24T00:00:00Z"
    published = _text(entry, "published")
    year = int(published[:4]) if published else None

    # Authors — each <author> has a <name> child
    authors = []
    for a in entry.findall("atom:author", NS):
        name_el = a.find("atom:name", NS)
        if name_el is not None and name_el.text:
            authors.append({"name": name_el.text.strip()})

    first_author = ""
    if authors:
        parts = authors[0]["name"].split()
        first_author = parts[-1] if parts else ""

    # Category tags — each <category> has a "term" attribute
    # like "cs.AI", "q-bio.NC"
    categories = [
        t.get("term", "")
        for t in entry.findall("atom:category", NS)
        if t.get("term")
    ]

    # Abstract — <summary> in Atom format
    # Replace newlines with spaces for clean storage
    abstract = _text(entry, "summary")
    if abstract:
        abstract = " ".join(abstract.split())

    title = _text(entry, "title")
    if title:
        title = " ".join(title.split())  # clean whitespace

    paper_id = make_paper_id(
        doi=doi,
        arxiv_id=arxiv_id,
        title=title,
        first_author=first_author,
        year=year,
    )

    paper = {
        "paper_id":          paper_id,
        "source":            "arxiv",
        "source_id":         arxiv_id or "",
        "doi":               doi,
        "title":             title or "",
        "abstract":          abstract,
        "year":              year,
        "authors":           authors,
        "fields":            categories,
        "language":          "en",
        "has_full_text":     False,
        "num_citations":     0,   # arXiv does not provide this
    }
    paper["completeness_score"] = compute_completeness(paper)
    return paper

def run():
    session = get_session()
    per_cat = max(1, TARGET_PER_SOURCE // len(ARXIV_CATEGORIES))

    total_inserted = 0

    for cat in ARXIV_CATEGORIES:
        print(f"\nCategory '{cat}' — fetching {per_cat} papers...")

        # Inner batch size reduced to 50 for arXiv stability
        for entry in tqdm(fetch_category_papers(cat, per_cat),
                          total=per_cat, desc=cat):

            paper = parse_entry(entry)
            if not paper["abstract"]:
                continue

            upsert_paper(session, paper)
            total_inserted += 1

    session.close()
    print(f"\narXiv done. Papers inserted/updated: {total_inserted}")


if __name__ == "__main__":
    run()