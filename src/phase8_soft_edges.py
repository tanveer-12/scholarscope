"""
phase8_soft_edges.py
--------------------
Builds semantic soft edges between papers using SPECTER2
embeddings and FAISS approximate nearest neighbor search.

WHAT THIS DOES IN PLAIN ENGLISH:
  Right now your graph has 4 isolated islands — one per source.
  Papers within OpenAlex cite each other. Papers within SS
  cite each other. But nothing connects CS papers to biology
  papers, or arXiv to PubMed.

  This script reads every paper's abstract, converts it to
  a 768-number vector that captures its meaning (embedding),
  then finds the 20 most similar papers for each paper.
  If two papers are semantically similar and from different
  communities, a "soft edge" is added between them.

  After this runs, your graph is one connected network
  spanning all four sources and all domains.

THE THREE STAGES:
  Stage A — Embed: run SPECTER2 on all abstracts
            → 768-dim vector per paper
  Stage B — Index: load all vectors into FAISS
            → enables fast similarity search
  Stage C — Search: for each paper find top-20 similar
            → insert soft edges where similarity > threshold

WHY SPECTER2 NOT SPECTER:
  SPECTER (original) was trained on biomedical + CS pairs.
  SPECTER2 was trained on a much broader corpus and handles
  cross-domain similarity better — a biology paper and a
  CS paper about the same statistical method will score
  higher similarity with SPECTER2 than SPECTER.
  This matters a lot for cross-domain bridge detection.

WHY FAISS:
  Brute-force similarity: compare every paper to every other
  paper = 3,599 × 3,599 = 13 million comparisons. Slow.
  FAISS builds an index that finds approximate nearest
  neighbors in milliseconds instead of seconds per query.
  At 3,599 papers it is fast either way, but the FAISS
  approach scales to millions of papers unchanged.

SOFT EDGE THRESHOLD:
  Only papers with cosine similarity > 0.75 get an edge.
  0.75 means the papers are genuinely about similar topics.
  Lower = more edges but noisier connections.
  Higher = fewer but more precise connections.
  0.75 is a good starting point — tune after seeing results.

CROSS-COMMUNITY FILTER:
  We only add soft edges between papers from DIFFERENT
  source systems (arxiv↔pubmed, ss↔openalex, etc.)
  OR between papers that will likely end up in different
  communities after Louvain runs.
  Adding soft edges within the same source just reinforces
  existing structure — cross-source edges are the new signal.
"""

import os
import sys
import json
import numpy as np
from tqdm import tqdm

# sentence-transformers provides SPECTER2
from sentence_transformers import SentenceTransformer

# FAISS for fast similarity search
import faiss

from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "phase7_ingestion.py"))
from db import get_session, insert_citation

# ── Config ────────────────────────────────────────────
SPECTER_MODEL    = "allenai-specter"   # downloads ~400MB first run
                                        # SPECTER2 needs extra setup;
                                        # allenai-specter works well
BATCH_SIZE       = 32     # papers encoded at once — reduce if OOM
TOP_K            = 20     # nearest neighbors to find per paper
SIM_THRESHOLD    = 0.75   # minimum cosine similarity for soft edge
CROSS_SOURCE_ONLY = True  # only add edges between different sources


# ══════════════════════════════════════════════════════
# STAGE A — LOAD PAPERS AND EMBED ABSTRACTS
# ══════════════════════════════════════════════════════

def load_papers(session):
    """
    Load all papers that have abstracts from PostgreSQL.
    Returns list of dicts with paper_id, source, abstract.
    We only embed papers with abstracts — no abstract
    means no meaningful embedding.
    """
    result = session.execute(text("""
        SELECT paper_id, source, title, abstract
        FROM papers
        WHERE abstract IS NOT NULL
          AND LENGTH(abstract) > 50
        ORDER BY paper_id
    """))
    rows = result.fetchall()
    print(f"Papers with usable abstracts: {len(rows):,}")
    return [
        {
            "paper_id": r[0],
            "source":   r[1],
            "title":    r[2] or "",
            "abstract": r[3],
        }
        for r in rows
    ]


def build_specter_inputs(papers):
    """
    Format papers for SPECTER input.
    SPECTER was trained on: "title [SEP] abstract"
    Using this format gives better embeddings than
    abstract alone because the title provides context.
    """
    return [
        f"{p['title']} [SEP] {p['abstract']}"
        for p in papers
    ]


def embed_papers(papers, model):
    """
    Encode all paper abstracts with SPECTER.

    encode() processes papers in batches internally.
    show_progress_bar=True shows a tqdm progress bar.
    convert_to_numpy=True returns numpy arrays (required by FAISS).
    normalize_embeddings=True scales each vector to length 1.
      This is important because we use cosine similarity,
      and for unit vectors cosine similarity = dot product,
      which FAISS computes very efficiently with IndexFlatIP.
    """
    texts = build_specter_inputs(papers)

    print(f"Encoding {len(texts):,} papers with SPECTER...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors for cosine sim
    )
    print(f"Embeddings shape: {embeddings.shape}")
    # Shape should be (n_papers, 768)
    return embeddings.astype("float32")  # FAISS requires float32


# ══════════════════════════════════════════════════════
# STAGE B — BUILD FAISS INDEX
# ══════════════════════════════════════════════════════

def build_faiss_index(embeddings):
    """
    Build a FAISS index for fast similarity search.

    IndexFlatIP = Inner Product index (exact, no approximation).
    For unit-normalized vectors, inner product = cosine similarity.
    "Flat" means it stores all vectors and does exact search —
    appropriate for 3,599 papers (fast enough).

    For millions of papers you would use IndexIVFFlat
    (approximate, partitioned) for speed, but at this
    scale exact search is fine and more accurate.

    The index is built once and queried once per paper
    in Stage C.
    """
    dim   = embeddings.shape[1]   # 768 for SPECTER
    index = faiss.IndexFlatIP(dim)

    # add() loads all vectors into the index
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal:,} vectors, "
          f"dimension {dim}")
    return index


# ══════════════════════════════════════════════════════
# STAGE C — SEARCH AND INSERT SOFT EDGES
# ══════════════════════════════════════════════════════

def build_soft_edges(papers, embeddings, index, session):
    """
    For each paper, find its TOP_K most similar papers
    and insert soft edges where similarity > SIM_THRESHOLD.

    index.search(query_vectors, k) returns:
      distances — shape (n_queries, k) — similarity scores
      indices   — shape (n_queries, k) — positions in index

    We process all papers in one batch query to FAISS,
    then loop through results to insert edges.
    """
    print(f"\nSearching top-{TOP_K} neighbors for each paper...")
    print(f"Similarity threshold: {SIM_THRESHOLD}")

    # Search all papers at once
    # distances[i][j] = similarity of paper i to its j-th neighbor
    # indices[i][j]   = index position of that neighbor
    distances, indices = index.search(embeddings, TOP_K + 1)
    # TOP_K + 1 because the closest match is always the paper itself
    # (similarity = 1.0), which we skip below

    # Build lookup: index position → paper dict
    pos_to_paper = {i: p for i, p in enumerate(papers)}

    edges_added   = 0
    edges_skipped = 0

    print("Inserting soft edges into database...")

    for i, paper in enumerate(tqdm(papers, desc="Processing")):
        for j in range(TOP_K + 1):
            neighbor_pos = indices[i][j]
            similarity   = float(distances[i][j])

            # Skip self-match (always the first result)
            if neighbor_pos == i:
                continue

            # Skip if below similarity threshold
            if similarity < SIM_THRESHOLD:
                continue

            neighbor = pos_to_paper[neighbor_pos]

            # Cross-source filter:
            # Only add edges between different source systems.
            # Same-source connections already exist as hard edges
            # or will be found by Louvain naturally.
            # Cross-source edges are the NEW signal we are adding.
            if CROSS_SOURCE_ONLY and paper["source"] == neighbor["source"]:
                edges_skipped += 1
                continue

            # Insert the soft edge in both directions
            # A similar to B means B similar to A —
            # undirected relationship stored as two directed edges
            insert_citation(
                session,
                paper["paper_id"],
                neighbor["paper_id"],
                edge_type="soft"
            )
            insert_citation(
                session,
                neighbor["paper_id"],
                paper["paper_id"],
                edge_type="soft"
            )
            edges_added += 2

    return edges_added, edges_skipped


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def run():
    session = get_session()

    # ── Stage A: Load and embed ───────────────────────
    print("=" * 55)
    print("STAGE A: Loading papers and embedding abstracts")
    print("=" * 55)

    papers = load_papers(session)
    if not papers:
        print("No papers with abstracts found. "
              "Run ingestion scripts first.")
        return

    print(f"\nLoading SPECTER model ({SPECTER_MODEL})...")
    print("First run downloads ~400MB — subsequent runs use cache.")
    model = SentenceTransformer(SPECTER_MODEL)

    embeddings = embed_papers(papers, model)

    # ── Stage B: Build FAISS index ───────────────────
    print("\n" + "=" * 55)
    print("STAGE B: Building FAISS index")
    print("=" * 55)

    index = build_faiss_index(embeddings)

    # ── Stage C: Find neighbors and insert edges ─────
    print("\n" + "=" * 55)
    print("STAGE C: Building soft edges")
    print("=" * 55)

    # Show source breakdown before edges
    sources = {}
    for p in papers:
        sources[p["source"]] = sources.get(p["source"], 0) + 1
    print("Papers by source:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count:,}")

    edges_added, edges_skipped = build_soft_edges(
        papers, embeddings, index, session
    )

    # ── Results ──────────────────────────────────────
    print("\n" + "=" * 55)
    print("RESULTS")
    print("=" * 55)
    print(f"Soft edges added:         {edges_added:,}")
    print(f"Same-source pairs skipped:{edges_skipped:,}")

    # Show final edge breakdown
    result = session.execute(text("""
        SELECT edge_type, COUNT(*) as count
        FROM citations
        GROUP BY edge_type
        ORDER BY count DESC
    """))
    print("\nCitation table breakdown:")
    for row in result:
        print(f"  {row[0]}: {row[1]:,}")

    # Show cross-source soft edges
    result = session.execute(text("""
        SELECT
            p1.source AS from_source,
            p2.source AS to_source,
            COUNT(*)  AS edges
        FROM citations c
        JOIN papers p1 ON c.citing_paper_id = p1.paper_id
        JOIN papers p2 ON c.cited_paper_id  = p2.paper_id
        WHERE c.edge_type = 'soft'
        GROUP BY p1.source, p2.source
        ORDER BY edges DESC
        LIMIT 20
    """))
    print("\nCross-source soft edges:")
    for row in result:
        print(f"  {row[0]} → {row[1]}: {row[2]:,}")

    session.close()
    print("\nPhase 8 complete.")
    print("Next: re-run Louvain community detection on the")
    print("new connected multi-source graph.")


if __name__ == "__main__":
    run()