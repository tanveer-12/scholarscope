"""
phase9_community_detection.py
------------------------------
Loads the multi-source citation graph from PostgreSQL
into NetworkX and runs hierarchical Louvain community
detection on it.

WHY THIS IS DIFFERENT FROM THE ORIGINAL PHASE 2:
  Original: 36,823 biomedical papers, all adjacent fields,
            communities were subtle subfields of medicine.

  Now: 3,599 papers across CS, biology, medicine,
       economics, physics connected by semantic soft edges.
       Communities should reflect genuine domain boundaries —
       CS papers cluster together, biology papers together,
       etc. — because the soft edges connect papers that
       share actual meaning, not just citation culture.

  This sharp domain separation is what makes bridge papers
  detectable — a paper connecting a CS cluster and a
  biology cluster is crossing a real boundary.
"""

import os
import sys
import pickle
import networkx as nx
import community as community_louvain
import pandas as pd
import numpy as np
from collections import Counter
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "phase7_ingestion.py"))
from db import get_session

OUTPUT_DIR = "data/processed/phase9"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════
# STAGE A — BUILD NETWORKX GRAPH FROM POSTGRESQL
# ══════════════════════════════════════════════════════

def build_graph(session):
    """
    Pull all papers and citation edges from PostgreSQL
    and build a NetworkX graph.

    Nodes = papers (with metadata as node attributes)
    Edges = citations (hard and soft, with edge_type)

    We load ALL edges — hard and soft — because for
    community detection we want the full connectivity
    picture. Edge direction is less important than
    connection for Louvain, so we convert to undirected.
    """
    print("Loading papers from database...")
    papers = session.execute(text("""
        SELECT paper_id, source, year, num_citations,
               abstract IS NOT NULL as has_abstract
        FROM papers
    """)).fetchall()

    print(f"  Papers: {len(papers):,}")

    # Create directed graph
    G = nx.DiGraph()

    # Add nodes with metadata
    for p in papers:
        G.add_node(p[0], source=p[1], year=p[2],
                   num_citations=p[3], has_abstract=p[4])

    print("Loading citation edges...")
    edges = session.execute(text("""
        SELECT citing_paper_id, cited_paper_id,
               edge_type, weight
        FROM citations
    """)).fetchall()

    print(f"  Edges: {len(edges):,}")

    for e in edges:
        G.add_edge(e[0], e[1],
                   edge_type=e[2],
                   weight=e[3] if e[3] else 1.0)

    print(f"\nGraph built:")
    print(f"  Nodes: {G.number_of_nodes():,}")
    print(f"  Edges: {G.number_of_edges():,}")
    print(f"  Is directed: {G.is_directed()}")

    return G


# ══════════════════════════════════════════════════════
# STAGE B — RUN HIERARCHICAL LOUVAIN
# ══════════════════════════════════════════════════════

def run_louvain(G, resolution=1.0, size_threshold=300):
    """
    Run hierarchical Louvain on the undirected graph.

    Same approach as original Phase 2 but with lower
    size_threshold (300 instead of 800) because we have
    fewer papers total. A community of 300 in a 3,599
    paper corpus is proportionally the same as 800 in
    a 36,823 paper corpus.
    """
    G_undirected = G.to_undirected()

    print(f"\nRunning Louvain (resolution={resolution})...")
    partition = community_louvain.best_partition(
        G_undirected,
        resolution=resolution,
        random_state=42,
        weight="weight"     # use edge weights (similarity scores)
    )

    n_communities = len(set(partition.values()))
    modularity    = community_louvain.modularity(
        partition, G_undirected, weight="weight"
    )

    print(f"  Communities found: {n_communities}")
    print(f"  Modularity score:  {modularity:.4f}")

    sizes = Counter(partition.values())
    sorted_sizes = sorted(sizes.values(), reverse=True)
    print(f"  Largest community: {sorted_sizes[0]} papers")
    print(f"  Median size:       {np.median(sorted_sizes):.0f} papers")
    print(f"  Singletons:        "
          f"{sum(1 for s in sorted_sizes if s == 1)}")

    # Check if hierarchical split is needed
    oversized = [cid for cid, size in sizes.items()
                 if size > size_threshold]

    if not oversized:
        print(f"\n  No communities exceed threshold ({size_threshold}).")
        print(f"  Flat Louvain sufficient — skipping hierarchical step.")
        return {node: str(comm) for node, comm in partition.items()}

    print(f"\n  {len(oversized)} communities exceed {size_threshold} papers.")
    print(f"  Running hierarchical split...")

    # Hierarchical pass on oversized communities
    final_partition = {
        node: str(comm) for node, comm in partition.items()
    }

    for parent_comm in oversized:
        nodes_in_comm = [
            node for node, comm in partition.items()
            if comm == parent_comm
        ]
        G_sub = G_undirected.subgraph(nodes_in_comm)

        if G_sub.number_of_edges() == 0:
            continue

        sub_partition = community_louvain.best_partition(
            G_sub, resolution=2.0, random_state=42,
            weight="weight"
        )
        sub_sizes = Counter(sub_partition.values())
        print(f"    Community {parent_comm} "
              f"({len(nodes_in_comm)} papers) → "
              f"{len(sub_sizes)} sub-communities")

        for node, sub_comm in sub_partition.items():
            final_partition[node] = f"{parent_comm}_{sub_comm}"

    return final_partition


# ══════════════════════════════════════════════════════
# STAGE C — ANALYZE COMMUNITIES BY SOURCE
# ══════════════════════════════════════════════════════

def analyze_communities(G, partition, session):
    """
    Show what each community looks like in terms of
    which sources and domains are represented.

    This tells us whether Louvain found genuine domain
    communities (CS papers together, biology together)
    or mixed communities (everything jumbled).

    Genuine domain communities = sharp boundaries =
    bridge papers are meaningful.
    Mixed communities = soft edges dominated by noise =
    need to adjust threshold.
    """
    # Load source info for all papers
    papers_df = pd.read_sql(text("""
        SELECT paper_id, source, fields
        FROM papers
    """), session.bind)

    papers_df["community"] = papers_df["paper_id"].map(partition)

    print("\nCommunity composition by source:")
    print("(showing top 15 communities by size)\n")

    comm_sizes = Counter(partition.values())
    top_comms  = [c for c, _ in comm_sizes.most_common(15)]

    for comm in top_comms:
        comm_papers = papers_df[papers_df["community"] == comm]
        size        = len(comm_papers)
        source_dist = comm_papers["source"].value_counts()

        print(f"  Community {comm} ({size} papers):")
        for src, count in source_dist.items():
            pct = count / size * 100
            bar = "█" * int(pct / 5)
            print(f"    {src:20s} {count:4d} ({pct:5.1f}%) {bar}")
        print()

    return papers_df


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def run():
    session = get_session()

    # Stage A: Build graph
    print("=" * 55)
    print("STAGE A: Building NetworkX graph from PostgreSQL")
    print("=" * 55)
    G = build_graph(session)

    # Save graph for later stages
    graph_path = os.path.join(OUTPUT_DIR, "multisource_graph.pkl")
    with open(graph_path, "wb") as f:
        pickle.dump(G, f)
    print(f"\nGraph saved → {graph_path}")

    # Stage B: Louvain
    print("\n" + "=" * 55)
    print("STAGE B: Hierarchical Louvain Community Detection")
    print("=" * 55)
    partition = run_louvain(G, resolution=1.0, size_threshold=300)

    n_communities = len(set(partition.values()))
    print(f"\nFinal communities: {n_communities}")

    # Stage C: Analyze
    print("\n" + "=" * 55)
    print("STAGE C: Community Analysis")
    print("=" * 55)
    papers_df = analyze_communities(G, partition, session)

    # Save partition and enriched paper table
    partition_path = os.path.join(OUTPUT_DIR, "partition.pkl")
    with open(partition_path, "wb") as f:
        pickle.dump(partition, f)
    print(f"Partition saved → {partition_path}")

    papers_df.to_parquet(
        os.path.join(OUTPUT_DIR, "papers_with_communities.parquet"),
        index=False
    )
    print(f"Papers with communities saved → "
          f"{OUTPUT_DIR}/papers_with_communities.parquet")

    # Summary
    print("\n" + "=" * 55)
    print("SUMMARY")
    print("=" * 55)
    print(f"Total papers:      {G.number_of_nodes():,}")
    print(f"Total edges:       {G.number_of_edges():,}")
    print(f"Communities found: {n_communities}")
    print(f"\nNext step: Phase 10 — weighted bridge scoring")

    session.close()


if __name__ == "__main__":
    run()