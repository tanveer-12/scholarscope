"""
ingest_pubmed.py
----------------
Pulls papers from PubMed via NCBI E-utilities.

HOW PUBMED WORKS:
  PubMed uses a two-step process — you cannot just
  search and get full paper details in one call.

  Step 1 — ESearch: search for matching paper IDs (PMIDs)
    "Find all papers matching 'machine learning AND clinical'"
    Returns: [42127322, 42127298, ...]

  Step 2 — EFetch: fetch full details for a list of PMIDs
    "Give me the full XML records for these IDs"
    Returns: XML with titles, abstracts, authors, MeSH terms

  This two-step design is intentional — ESearch is fast
  and runs on indexes. EFetch is slower but returns full data.
  By separating them you can search efficiently then only
  fetch what you need.

MESH TERMS:
  PubMed tags papers with MeSH (Medical Subject Headings)
  — a controlled vocabulary of medical concepts.
  We store these as the paper's "fields" column.
  They are more precise than arXiv categories or
  OpenAlex concept names for biomedical papers.

XML PARSING:
  PubMed returns XML with a complex nested structure.
  We use try/except around each paper to skip malformed
  records without crashing the whole ingestion run.
"""

import requests
import time
import xml.etree.ElementTree as ET
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from tqdm import tqdm
from config import (TARGET_PER_SOURCE, PUBMED_SEARCH_TERMS,
                    NCBI_API_KEY)
from db import (get_session, make_paper_id, upsert_paper,
                compute_completeness)

SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Without API key: max 3 requests/second → 0.34 sec delay
# With API key:    max 10 requests/second → 0.11 sec delay
DELAY = 0.11 if NCBI_API_KEY else 0.34


def search_pmids(query, max_results):
    """
    Step 1: Search PubMed and return a list of PMIDs.

    retmode=json means the response is JSON not XML.
    retmax controls how many IDs to return.
    sort=relevance puts the most relevant papers first.
    """
    params = {
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "retmode": "json",
        "sort":    "relevance",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def fetch_details_xml(pmids):
    """
    Step 2: Fetch full XML records for a batch of PMIDs.

    rettype=abstract gives us the abstract and metadata.
    retmode=xml gives structured XML (not plain text).
    We send up to 100 PMIDs at once as a comma-separated string.
    """
    params = {
        "db":      "pubmed",
        "id":      ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract",
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(FETCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text


def parse_pubmed_xml(xml_text):
    """
    Parse PubMed XML → list of unified schema dicts.

    PubMed XML structure (simplified):
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>...</ArticleTitle>
            <Abstract><AbstractText>...</AbstractText></Abstract>
            <AuthorList>
              <Author><LastName>Smith</LastName>...</Author>
            </AuthorList>
          </Article>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="doi">10.xxx</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>

    findall(".//Tag") uses XPath — the ".." means
    "search anywhere in the tree below this element"
    """
    root   = ET.fromstring(xml_text)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        try:
            medline = article.find("MedlineCitation")
            art     = medline.find("Article")

            # Title — itertext() handles mixed content
            # like "Effect of <i>in vitro</i> treatment"
            title_el = art.find("ArticleTitle")
            title    = ("".join(title_el.itertext()).strip()
                        if title_el is not None else "")

            # Abstract — may have multiple sections
            # (Background, Methods, Results, Conclusions)
            # We join them all into one string
            abstract_parts = art.findall(".//AbstractText")
            abstract = " ".join(
                "".join(p.itertext()) for p in abstract_parts
            ).strip() or None

            # Year — inside PubDate which may have
            # Year, Month, Day or just MedlineDate
            year    = None
            pub_date = art.find(".//PubDate")
            if pub_date is not None:
                yr_el = pub_date.find("Year")
                if yr_el is not None and yr_el.text:
                    year = int(yr_el.text)

            # Authors
            authors = []
            for a in art.findall(".//Author"):
                last  = getattr(a.find("LastName"),  "text", "") or ""
                first = getattr(a.find("ForeName"),  "text", "") or ""
                name  = f"{first} {last}".strip()
                if name:
                    authors.append({"name": name})

            # DOI — inside ArticleIdList in PubmedData
            doi = None
            for eid in article.findall(".//ArticleId"):
                if eid.get("IdType") == "doi":
                    doi = (eid.text or "").strip()
                    break

            # PMID
            pmid_el = medline.find("PMID")
            pmid    = pmid_el.text.strip() if pmid_el is not None else ""

            # MeSH terms — controlled vocabulary tags
            # Each MeshHeading has a DescriptorName
            fields = [
                m.find("DescriptorName").text
                for m in medline.findall(".//MeshHeading")
                if m.find("DescriptorName") is not None
            ]

            # Skip papers without abstracts
            if not abstract:
                continue

            first_author = ""
            if authors:
                parts = authors[0]["name"].split()
                first_author = parts[-1] if parts else ""

            paper_id = make_paper_id(
                doi=doi,
                title=title,
                first_author=first_author,
                year=year
            )

            paper = {
                "paper_id":          paper_id,
                "source":            "pubmed",
                "source_id":         pmid,
                "doi":               doi or None,
                "title":             title,
                "abstract":          abstract,
                "year":              year,
                "authors":           authors,
                "fields":            fields,
                "language":          "en",
                "has_full_text":     False,
                "num_citations":     0,
            }
            paper["completeness_score"] = compute_completeness(paper)
            papers.append(paper)

        except Exception as e:
            # Log and skip malformed records —
            # do not crash the whole run for one bad paper
            print(f"  Skipping one record: {e}")
            continue

    return papers


def run():
    session  = get_session()
    per_term = max(1, TARGET_PER_SOURCE // len(PUBMED_SEARCH_TERMS))

    total_inserted = 0

    for query in PUBMED_SEARCH_TERMS:
        print(f"\nSearching: '{query}' ({per_term} papers)...")

        pmids = search_pmids(query, per_term)
        if not pmids:
            print("  No results for this query.")
            continue

        print(f"  Found {len(pmids)} PMIDs. Fetching details...")

        # Fetch in batches of 100 — EFetch limit
        for i in tqdm(range(0, len(pmids), 100),
                      desc="Fetching batches"):
            batch    = pmids[i:i+100]
            xml_text = fetch_details_xml(batch)
            papers   = parse_pubmed_xml(xml_text)

            for paper in papers:
                upsert_paper(session, paper)
                total_inserted += 1

            time.sleep(DELAY)

    session.close()
    print(f"\nPubMed done. Papers inserted/updated: {total_inserted}")


if __name__ == "__main__":
    run()