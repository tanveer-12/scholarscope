"""
Dagster assets for multi-source paper ingestion.
THREE-TIER SOURCE STRATEGY:
  Why this exists: Phases 7-12 failed statistically
  because citation counts were inconsistent across sources.
  arXiv stores 0, PubMed stores 0, Semantic Scholar uses
  a different counting methodology than OpenAlex. Mixing
  these into one regression model corrupts the target variable.

  Tier 1 — openalex_primary: OpenAlex only, citation counts
    are authoritative. Powers the statistical pipeline.
    Year range 2000-2022, minimum 3 citations.

  Tier 2 — supplementary: Semantic Scholar + PubMed papers
    that are NOT in OpenAlex. Resolved to OpenAlex by DOI
    where possible. Only use SS/PubMed citation count if
    OpenAlex has no record.

  Tier 3 — semantic_only: arXiv preprints. No citation data.
    Used for Domain Navigator mode and soft edges only.
    Never included in regression or Wilcoxon test.

KEY LESSON FROM PHASE 11:
  Previous ingestion sorted by cited_by_count DESCENDING.
  This pulled the most famous papers first (LSTM, ResNet).
  Bridge papers among famous papers are citation magnets,
  not overlooked papers. The hypothesis was reversed.

  New ingestion sorts by cited_by_count ASCENDING for
  historical/core tiers. This pulls less-cited papers
  first — papers more likely to be genuinely overlooked.
  We exclude completely uncited papers via the >2 filter
  to avoid zero-citation noise.
"""

import os
import sys
import time
import json
import hashlib
import re
import requests
import structlog
from ..resources.database import DatabaseResource
from dagster import(
    asset,
    AssetExecutionContext,
    Output,
    MetadataValue,
)
from sqlalchemy import text

# Allowing importing from phase7_ingestion helpers
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "../../../"
))

logger = structlog.get_logger()

# ── Target domains ──
# Ten concepts chosen for genuine semantic distance from each other.
# A paper bridging machine_learning and molecular_biology crosses a community distance of ~0.7.
# A paper bridging two medical subfields crosses ~0.15.
# We need concepts with distances > 0.4 for meaningful bridge detection.
OPENALEX_CONCEPTS = {
    "machine_learning":       "C119857082",
    "molecular_biology":      "C24890656",
    "clinical_medicine":      "C71924100",
    "economics":              "C162324750",
    "materials_science":      "C192562407",
    "neuroscience":           "C86803240",
    "environmental_science":  "C39432304",
    "computer_science":       "C41008148",
    "physics":                "C121332964",
    "genomics":               "C54355233",
}

# Fields requested from OpenAlex - only what we need to reduce API quota usage and repsonse size
OPENALEX_FIELDS = ",".join([
    "id",
    "doi",
    "title",
    "abstract_inverted_index",
    "publication_year",
    "authorships",
    "concepts",
    "cited_by_count",
    "referenced_works",
    "primary_location",
    "open_access",
])

# Per-concept paper targets
# historical: 2000 × 10 concepts = ~20,000 papers
# core: 5000 × 10 concepts = ~50,000 papers
# recent: 200 × 10 concepts = ~2,000 papers/day
TARGETS = {
    "historical": 2000,
    "core":       5000,
    "recent":     200,
}

# ══════════════════
# UTILITY FUNCTIONS
# ══════════════════

def make_paper_id(doi=None, arxiv_id=None, title=None, first_author=None, year=None):
    """
    Generate canonical paper ID
    same paper from different sources -> same ID
    Priority: DOI > arXiv ID > title+author+year hash
    """
    if doi:
        clean = doi.lower().strip()
        clean = clean.replace("https://doi.org/", "")
        clean = clean.replace("htpp://doi.org/", "")
        return f"doi:{clean}"
    if arxiv_id:
        return f"arxiv:{arxiv_id.strip()}"
    key = f"{_normalize(title)}_{first_author or ''}_{year or ''}"
    return f"hash:{hashlib.md5(key.encode()).hexdigest()[:16]}"

def _normalize(text):
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip()

def decode_abstract(inverted_index):
    """
    OpenAlex stores abstracts as inverted index
    Reconstruct plain text by sorting (position, word) pairs.
    """
    if not inverted_index:
        return None
    pairs = [
        (pos, word)
        for word, positions in inverted_index.items()
        for pos in positions
    ]
    return " ".join(word for _, word in sorted(pairs))


def compute_completeness(paper):
    """
    Score 0.0 - 1.0 based on how many fields are populated.
    """
    optional = ["abstract", "doi", "authors","fields","year"]
    filled = sum(1 for f in optional if paper.get(f))
    return round(filled /len(optional), 2)


def fetch_openalex_page(params, headers, context):
    """
    Fetch one page from OpenAlex with retry logic
    return (results, next_cursor) or ([], None) on failure
    """
    for attempt in range(4):
        try:
            resp = requests.get(
                "https://api.openalex.org/works",
                params=params,
                headers=headers,
                timeout=30,
            )
            if resp.status_code == 429:
                wait = 30 * (2 ** attempt)
                context.log.warning(
                    f"Rate limited. Waiting {wait}s (attempt {attempt+1})"
                )
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                context.log.error(
                    f"API error {resp.status_code}: {resp.text[:200]}"
                )
                return [], None
            data = resp.json()
            results = data.get("results", [])
            cursor = data.get("meta", {}).get("next_cursor")
            return results, cursor
        except requests.exceptions.Timeout:
            wait = 20 * (attempt + 1)
            context.log.warning(f"Timeout. Waiting {wait}s...")
            time.sleep(wait)
        except Exception as e:
            context.log.error(f"Request error: {e}")
            return [], None
    return [], None


def upsert_paper(session, paper, tier, concept_name, citation_source="openalex"):
    """
    Insert or update a paper in PostgreSQL
    ON CONFLICT updates only if new data is better quality
    also inserts a processing_status row for tracking.
    """
    eligible = (
        paper["num_citations"] >= 3
        and tier in ("historical", "core")
        and bool(paper.get("abstract"))
        and len(paper.get("abstract") or "") > 100
    )
    session.execute(text("""
        INSERT INTO papers(
            paper_id, source, source_id, doi, title,
            abstract, year, authors, fields, language, has_full_text,
            completeness_score, num_citations, tier, eligible_for_test, 
            primary_concept, citation_score
        ) VALUES (
            :paper_id, 'openalex', :source_id, :doi, :title, :abstract,
            :year, cast(:authors as jsonb), cast(:fields as jsonb),
            'en', :has_full_text, :completeness, :num_citations, :tier, :eligible,
            :concept, :citation_source
        )
        ON CONFLICT (paper_id) DO UPDATE SET
            num_citations = GREATEST(EXCLUDED.num_citations, papers.num_citations),
            completeness_score = GREATEST(EXCLUDED.completeness_score, papers.completeness_score),
            eligible_for_test = EXCLUDED.eligible_for_test,
            updated_at = NOW()
    """), {
        "paper_id": paper["paper_id"],        
        "source_id": paper["source_id"],        
        "doi": paper["doi"],        
        "title": paper["title"],        
        "abstract": paper["abstract"],        
        "year": paper["year"],        
        "authors": json.dumps(paper.get("authors", [])),     
        "fields": json.dumps(paper.get("fields", [])),        
        "has_full_text": paper.get("has_full_text", False),      
        "completeness": paper["completeness_score"],        
        "num_citations": paper["num_citations"],        
        "tier": tier,        
        "eligible": eligible,        
        "concept": concept_name,        
        "citation_source": citation_source,        
    })
    # Insert processing status row(ignore if exists)
    session.execute(text("""
        INSERT INTO processing_status (paper_id)
        VALUES (:pid)
        ON CONFLICT (paper_id) DO NOTHING
    """), {"pid": paper["paper_id"]})


def process_openalex_raw(raw):
    """Transform one OpenAlex API response dict → unified paper dict."""
    doi = (raw.get("doi") or "").replace("https://doi.org/", "").strip()

    authors = [
        {
            "name": a.get("author", {}).get("display_name", ""),
            "id":   a.get("author", {}).get("id", ""),
        }
        for a in (raw.get("authorships") or [])
    ]

    fields = [
        c.get("display_name", "")
        for c in (raw.get("concepts") or [])
        if c.get("score", 0) > 0.3
    ]

    abstract = decode_abstract(raw.get("abstract_inverted_index"))

    # Check open access
    oa = raw.get("open_access") or {}
    has_full_text = bool(oa.get("is_oa", False))

    paper = {
        "paper_id":       make_paper_id(doi=doi or None,
                                        title=raw.get("title"),
                                        year=raw.get("publication_year")),
        "source_id":      raw.get("id", ""),
        "doi":            doi or None,
        "title":          (raw.get("title") or "").strip(),
        "abstract":       abstract,
        "year":           raw.get("publication_year"),
        "authors":        authors,
        "fields":         fields,
        "has_full_text":  has_full_text,
        "num_citations":  raw.get("cited_by_count", 0),
    }
    paper["completeness_score"] = compute_completeness(paper)
    return paper, raw.get("referenced_works") or []

# CORE FETCH FUNCTION

def fetch_and_store_openalex(context, database, tier, year_start, year_end,
    sort, max_per_concept, from_date=None,
):
    """
    Fetch papers from OpenAlex for all ten concepts
    and store in PostgreSQL.

    SORT STRATEGY:
      historical/core: cited_by_count:asc
        Gets less-famous papers first. Bridge papers among
        overlooked works are what the hypothesis is about.
        Famous papers (LSTM, ResNet) are not hidden gems.

      recent: publication_date:desc
        Gets newest papers first for the daily update.

    CITATION FILTER: cited_by_count:>2
      Excludes papers with 0, 1, or 2 citations.
      Prevents zero-citation noise from corrupting regression.
      Papers below this threshold go into semantic_only tier
      if they arrive via other sources.
    """
    session = database.get_session()
    api_key = os.getenv("OPENALEX_API_KEY", "")
    headers = ({"Authorization": f"Bearer {api_key}"}
               if api_key else {})
    total_inserted = 0
    total_skipped = 0
    total_ref_edges = 0
    concept_counts = {}

    for concept_name, concept_id in OPENALEX_CONCEPTS.items():
        context.log.info(
            f"Fetching {concept_name} | "
            f"tier={tier} | "
            f"years={year_start}-{year_end} | "
            f"target={max_per_concept}"
        )
        filter_parts = [
            f"concepts.id:{concept_id}",
            "has_abstract:true",
            f"publication_year:{year_start}-{year_end}",
            "cited_by_count:>2",
        ]
        if from_date:
            filter_parts.append(f"from_updated_date:{from_date}")
        
        cursor = "*"
        fetched = 0
        while fetched < max_per_concept:
            params = {
                "filter": ",".join(filter_parts),
                "per-page": min(200, max_per_concept - fetched),
                "cursor": cursor,
                "select": OPENALEX_FIELDS,
                "sort": sort
            }
            results, next_cursor = fetch_openalex_page(params, headers, context)
            if not results:
                break

            for raw in results:
                # SKip papers without titles
                if not raw.get("title"):
                    total_skipped += 1
                    continue
                
                paper, refs = process_openalex_raw(raw)
                upsert_paper(session, paper, tier, concept_name)

                # insert hard citation edges
                for ref_id in refs:
                    if ref_id:
                        try:
                            session.execute(text("""
                                INSERT INTO citations(citing_paper_id, cited_paper_id, edge_type, weight)
                                VALUES (:citing, :cited, 'hard', 1.0)
                                ON CONFLICT DO NOTHING
                            """),{
                                "citing":paper["paper_id"],
                                "cited":ref_id,
                            })
                            total_ref_edges += 1
                        except Exception:
                            pass
                
                fetched += 1
                total_inserted += 1

            session.commit()
            context.log.info(
                f" {concept_name}: {fetched}/{max_per_concept} fetched"
            )

            cursor = next_cursor
            if not cursor:
                break
            time.sleep(0.1)
        
        concept_counts[concept_name] = fetched
        context.log.info(
            f" Completed {concept_name}: {fetched} papers"
        )

    session.close()
    context.log.info(
        f"Ingestion complete | "
        f"tier={tier} | "
        f"total={total_inserted} | "
        f"skipped={total_skipped} | "
        f"ref_edges={total_ref_edges}"
    )
    return total_inserted, concept_counts

# =========
# DAGSTER ASSETS — TIER 1: OPENALEX PRIMARY
# =========

@asset(
    group_name="ingestion_primary",
    description=(
        "Fetch historical papers (2000-2009) from OpenAlex. "
        "Run ONCE — these papers are stable, citation counts "
        "do not change significantly for 15+ year old papers. "
        "Target: ~20,000 papers across 10 concept domains. "
        "Sort: cited_by_count ascending (less-famous papers first)."
    ),
    compute_kind="python",
)
def historical_papers(context: AssetExecutionContext, database: DatabaseResource,):
    total, counts = fetch_and_store_openalex(
        context=context,
        database=database,
        tier="historical",
        year_start=2000,
        year_end=2009,
        sort="cited_by_count:asc",
        max_per_concept=TARGETS["historical"],
    )
    context.add_output_metadata({
        "papers_inserted": MetadataValue.int(total),
        "by_concept": MetadataValue.json(counts),
        "tier": MetadataValue.text("historical"),
        "year_range": MetadataValue.text("2000-2009"),
    })
    return Output(value=total)

@asset(
    group_name="ingestion_primary",
    description=(
        "Fetch core corpus papers (2010-2022) from OpenAlex. "
        "Run monthly to catch newly indexed papers. "
        "Target: ~50,000 papers across 10 concept domains. "
        "Sort: cited_by_count ascending (less-famous papers first). "
        "This is the primary corpus for statistical testing."
    ),
    compute_kind="python",
)
def core_papers(context: AssetExecutionContext,database: DatabaseResource,):
    total, counts = fetch_and_store_openalex(
        context=context,
        database=database,
        tier="core",
        year_start=2010,
        year_end=2022,
        sort="cited_by_count:asc",    # less-famous first
        max_per_concept=TARGETS["core"],
    )
    context.add_output_metadata({
        "papers_inserted": MetadataValue.int(total),
        "by_concept": MetadataValue.json(counts),
        "tier": MetadataValue.text("core"),
        "year_range": MetadataValue.text("2010-2022"),
    })

    return Output(value=total)

@asset(
    group_name="ingestion_primary",
    description=(
        "Fetch recent papers (2023+) from OpenAlex daily. "
        "Pulls papers published or updated in the last 48 hours. "
        "These go into tier=recent and eligible_for_test=FALSE "
        "because citation counts are not yet stable."
    ),
    compute_kind="python",
)
def recent_papers(context: AssetExecutionContext,database: DatabaseResource,):
    from datetime import datetime, timedelta
    since = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")

    total, counts = fetch_and_store_openalex(
        context=context,
        database=database,
        tier="recent",
        year_start=2023,
        year_end=datetime.now().year,
        sort="publication_date:desc",
        max_per_concept=TARGETS["recent"],
        from_date=since,
    )

    context.add_output_metadata({
        "papers_inserted": MetadataValue.int(total),
        "by_concept": MetadataValue.json(counts),
        "since": MetadataValue.text(since),
    })

    return Output(value=total)

@asset(
    group_name="ingestion_primary",
    deps=["core_papers"],
    description=(
        "Refresh citation counts for core/historical papers monthly. "
        "Citation counts change as new papers cite older ones. "
        "Accurate counts are critical for the regression residual signal. "
        "Processes papers not updated in the last 25 days."
    ),
    compute_kind="python",
)
def refresh_citation_counts(context: AssetExecutionContext,database: DatabaseResource,):
    session = database.get_session()
    api_key = os.getenv("OPENALEX_API_KEY", "")
    headers = ({"Authorization": f"Bearer {api_key}"}
               if api_key else {})

    # Papers not refreshed in 25 days
    result = session.execute(text("""
        SELECT paper_id, doi FROM papers
        WHERE tier IN ('core', 'historical')
          AND doi IS NOT NULL
          AND updated_at < NOW() - INTERVAL '25 days'
        ORDER BY updated_at ASC
        LIMIT 10000
    """))
    papers  = result.fetchall()
    updated = 0

    context.log.info(f"Refreshing citations for {len(papers)} papers")

    for i in range(0, len(papers), 50):
        batch  = papers[i:i+50]
        dois   = [p[1] for p in batch]
        filter_str = "|".join(f"doi:{d}" for d in dois)

        params = {
            "filter": filter_str,
            "select": "doi,cited_by_count",
            "per-page": 50,
        }
        results, _ = fetch_openalex_page(params, headers, context)

        for work in results:
            doi   = (work.get("doi") or "").replace(
                "https://doi.org/", ""
            ).strip()
            count = work.get("cited_by_count", 0)

            session.execute(text("""
                UPDATE papers SET
                    num_citations     = :count,
                    eligible_for_test = (
                        :count >= 3
                        AND tier IN ('core', 'historical')
                        AND abstract IS NOT NULL
                    ),
                    updated_at = NOW()
                WHERE doi = :doi
            """), {"count": count, "doi": doi})
            updated += 1

        session.commit()
        time.sleep(0.15)

    session.close()
    context.log.info(f"Citation counts refreshed: {updated}")
    context.add_output_metadata({
        "citations_updated": MetadataValue.int(updated)
    })
    return Output(value=updated)

# ==========
# DAGSTER ASSETS — TIER 2: SUPPLEMENTARY (SS + PUBMED)
# ==========

@asset(
    group_name="ingestion_supplementary",
    description=(
        "Fetch papers from Semantic Scholar that are NOT in OpenAlex. "
        "Resolves each SS paper to OpenAlex by DOI first. "
        "Only stores papers where DOI lookup fails (truly not in OA). "
        "Uses SS citation count as fallback, flagged citation_source=ss."
    ),
    compute_kind="python",
)
def supplementary_semantic_scholar(context: AssetExecutionContext,database: DatabaseResource,):
    """
    Semantic Scholar as discovery supplement.

    For every SS paper:
      1. Extract DOI
      2. Check if DOI already exists in our papers table
      3. If yes: skip (OpenAlex already has it)
      4. If no: try OpenAlex DOI lookup for citation count
      5. If OpenAlex has it: store with OA citation count
      6. If OpenAlex does not have it: store with SS count,
         flag citation_source = 'semantic_scholar'
    """
    from config import SEMANTIC_SCHOLAR_API_KEY

    session = database.get_session()
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = ({"x-api-key": api_key} if api_key else {})
    oa_key  = os.getenv("OPENALEX_API_KEY", "")
    oa_hdrs = ({"Authorization": f"Bearer {oa_key}"} if oa_key else {})

    # Cross-domain intersection queries
    # These target papers at the boundary of two fields —
    # exactly what the bridge hypothesis is about.
    INTERSECTION_QUERIES = [
        "machine learning cancer prognosis",
        "deep learning medical imaging diagnosis",
        "natural language processing clinical notes",
        "graph neural network drug discovery",
        "transformer protein structure",
        "reinforcement learning economics",
        "neural network materials discovery",
        "network science epidemiology outbreak",
        "computer vision pathology",
        "NLP genomics variant annotation",
    ]

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
    FIELDS   = "paperId,externalIds,title,abstract,year,authors,fieldsOfStudy,citationCount"

    total_new     = 0
    total_skipped = 0

    for query in INTERSECTION_QUERIES:
        context.log.info(f"SS query: '{query}'")
        token   = None
        fetched = 0

        while fetched < 200:
            params = {
                "query":  query,
                "fields": FIELDS,
                "limit":  100,
                "year":   "2010-2022",
            }
            if token:
                params["token"] = token

            for attempt in range(3):
                resp = requests.get(
                    BASE_URL, params=params,
                    headers=headers, timeout=30
                )
                if resp.status_code == 429:
                    time.sleep(30 * (attempt + 1))
                    continue
                if resp.status_code == 400:
                    context.log.warning(f"SS 400: {resp.text[:100]}")
                    break
                resp.raise_for_status()
                break

            data  = resp.json()
            items = data.get("data", [])
            if not items:
                break

            for item in items:
                if not item.get("abstract"):
                    total_skipped += 1
                    continue

                ext      = item.get("externalIds") or {}
                doi      = (ext.get("DOI") or "").strip()
                arxiv_id = (ext.get("ArXiv") or "").strip()

                # Check if already in our DB
                if doi:
                    existing = session.execute(text(
                        "SELECT 1 FROM papers WHERE doi = :doi LIMIT 1"
                    ), {"doi": doi.lower()}).fetchone()
                    if existing:
                        total_skipped += 1
                        fetched += 1
                        continue

                # Try OpenAlex resolution by DOI
                oa_count = None
                if doi:
                    oa_resp = requests.get(
                        "https://api.openalex.org/works",
                        params={"filter": f"doi:{doi}",
                                "select": "cited_by_count"},
                        headers=oa_hdrs,
                        timeout=15,
                    )
                    if oa_resp.status_code == 200:
                        oa_results = oa_resp.json().get("results", [])
                        if oa_results:
                            oa_count = oa_results[0].get("cited_by_count", 0)
                    time.sleep(0.1)

                citation_count  = oa_count if oa_count is not None \
                                  else item.get("citationCount", 0)
                citation_source = "openalex" if oa_count is not None \
                                  else "semantic_scholar"

                # Skip low-citation papers
                if citation_count < 3:
                    total_skipped += 1
                    fetched += 1
                    continue

                authors_raw  = item.get("authors") or []
                first_author = ""
                if authors_raw:
                    parts = authors_raw[0].get("name", "").split()
                    first_author = parts[-1] if parts else ""

                paper = {
                    "paper_id":   make_paper_id(
                        doi=doi or None,
                        arxiv_id=arxiv_id or None,
                        title=item.get("title"),
                        first_author=first_author,
                        year=item.get("year"),
                    ),
                    "source_id":  item.get("paperId", ""),
                    "doi":        doi or None,
                    "title":      (item.get("title") or "").strip(),
                    "abstract":   item.get("abstract"),
                    "year":       item.get("year"),
                    "authors":    [{"name": a.get("name", "")}
                                   for a in authors_raw],
                    "fields":     item.get("fieldsOfStudy") or [],
                    "has_full_text": False,
                    "num_citations": citation_count,
                }
                paper["completeness_score"] = compute_completeness(paper)

                upsert_paper(
                    session, paper, "core",
                    "cross_domain_intersection",
                    citation_source=citation_source
                )
                total_new += 1
                fetched   += 1

            session.commit()
            token = data.get("token")
            if not token:
                break
            time.sleep(0.5)

    session.close()
    context.log.info(
        f"SS supplementary done | new={total_new} | skipped={total_skipped}"
    )
    context.add_output_metadata({
        "new_papers": MetadataValue.int(total_new),
        "skipped": MetadataValue.int(total_skipped),
    })
    return Output(value=total_new)

# =========
# DAGSTER ASSETS — TIER 3: SEMANTIC ONLY (ARXIV)
# =========

@asset(
    group_name="ingestion_semantic_only",
    description=(
        "Fetch arXiv preprints — semantic-only pool. "
        "No citation data available from arXiv. "
        "These papers contribute to soft edges and Domain Navigator "
        "but are EXCLUDED from statistical tests. "
        "tier=semantic_only, eligible_for_test=FALSE always."
    ),
    compute_kind="python",
)
def semantic_only_arxiv(
    context: AssetExecutionContext,
    database: DatabaseResource,
):
    """
    arXiv papers for semantic enrichment only.

    These are stored with tier='semantic_only' and
    eligible_for_test=FALSE permanently.

    Why include them at all:
      arXiv CS papers are often the first to report
      cross-domain methods (ML applied to biology,
      NLP applied to medicine). They help form the
      semantic graph and make Domain Navigator useful
      for CS researchers even before a paper is published.
    """
    import xml.etree.ElementTree as ET

    CATEGORIES = [
        "cs.AI", "cs.LG", "cs.CL", "cs.CV",
        "q-bio.NC", "q-bio.GN", "q-bio.QM",
        "econ.GN", "physics.med-ph",
    ]
    NS = {
        "atom":  "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    session       = database.get_session()
    total_inserted = 0

    for cat in CATEGORIES:
        context.log.info(f"arXiv category: {cat}")
        fetched = 0

        for start in range(0, 300, 50):
            params = {
                "search_query": f"cat:{cat}",
                "start":        start,
                "max_results":  50,
                "sortBy":       "submittedDate",
                "sortOrder":    "descending",
            }

            for attempt in range(3):
                try:
                    resp = requests.get(
                        "http://export.arxiv.org/api/query",
                        params=params, timeout=60,
                    )
                    if resp.status_code == 429:
                        time.sleep(30 * (attempt + 1))
                        continue
                    resp.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    time.sleep(20 * (attempt + 1))
                    continue

            root    = ET.fromstring(resp.text)
            entries = root.findall("atom:entry", NS)
            if not entries:
                break

            for entry in entries:
                arxiv_url = (entry.find("atom:id", NS).text or "")
                arxiv_id  = None
                if "/abs/" in arxiv_url:
                    arxiv_id = arxiv_url.split("/abs/")[-1].split("v")[0]

                doi_el = entry.find("arxiv:doi", NS)
                doi    = (doi_el.text or "").strip() if doi_el is not None else None

                pub = (entry.find("atom:published", NS).text or "")
                year = int(pub[:4]) if pub else None

                abstract_el = entry.find("atom:summary", NS)
                abstract = " ".join((abstract_el.text or "").split()) \
                           if abstract_el is not None else None

                title_el = entry.find("atom:title", NS)
                title = " ".join((title_el.text or "").split()) \
                        if title_el is not None else None

                authors = [
                    {"name": a.find("atom:name", NS).text.strip()}
                    for a in entry.findall("atom:author", NS)
                    if a.find("atom:name", NS) is not None
                ]
                categories = [
                    t.get("term", "")
                    for t in entry.findall("atom:category", NS)
                ]

                if not abstract or not title:
                    continue

                paper = {
                    "paper_id":   make_paper_id(
                        doi=doi, arxiv_id=arxiv_id,
                        title=title,
                        first_author=authors[0]["name"].split()[-1]
                            if authors else "",
                        year=year,
                    ),
                    "source_id":  arxiv_id or "",
                    "doi":        doi,
                    "title":      title,
                    "abstract":   abstract,
                    "year":       year,
                    "authors":    authors,
                    "fields":     categories,
                    "has_full_text": False,
                    "num_citations": 0,
                }
                paper["completeness_score"] = compute_completeness(paper)

                session.execute(text("""
                    INSERT INTO papers (
                        paper_id, source, source_id, doi, title,
                        abstract, year, authors, fields, language,
                        has_full_text, completeness_score, num_citations,
                        tier, eligible_for_test, primary_concept,
                        citation_source
                    ) VALUES (
                        :paper_id, 'arxiv', :source_id, :doi, :title,
                        :abstract, :year,
                        cast(:authors as jsonb), cast(:fields as jsonb),
                        'en', :has_full_text, :completeness, 0,
                        'semantic_only', FALSE, :concept, 'none'
                    )
                    ON CONFLICT (paper_id) DO NOTHING
                """), {
                    "paper_id":    paper["paper_id"],
                    "source_id":   paper["source_id"],
                    "doi":         doi,
                    "title":       title,
                    "abstract":    abstract,
                    "year":        year,
                    "authors":     json.dumps(authors),
                    "fields":      json.dumps(categories),
                    "has_full_text": False,
                    "completeness": paper["completeness_score"],
                    "concept":     cat,
                })

                session.execute(text("""
                    INSERT INTO processing_status (paper_id)
                    VALUES (:pid)
                    ON CONFLICT (paper_id) DO NOTHING
                """), {"pid": paper["paper_id"]})

                total_inserted += 1
                fetched        += 1

            session.commit()
            time.sleep(8)  # arXiv rate limit

        context.log.info(f"  {cat}: {fetched} papers")

    session.close()
    context.add_output_metadata({
        "papers_inserted": MetadataValue.int(total_inserted)
    })
    return Output(value=total_inserted)