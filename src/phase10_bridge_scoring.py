"""
phase10_bridge_scoring.py
--------------------------
Computes weighted bridge scores for all papers in the
multi-source corpus using three independent signals.

HOW THIS DIFFERS FROM THE ORIGINAL PHASE 3:

Original bridge score:
  BS(x) = normalize(CDS) × normalize(CD)
  CDS = how many communities cite x (unweighted count)
  CD  = how many communities neighbor x (unweighted count)

New weighted bridge score:
  BS_v2(x) = normalize(wCDS) × normalize(wCD) × normalize(BC)

  wCDS = weighted citation diversity — each citation weighted
         by semantic DISTANCE between communities
         (far-away citations count more than nearby ones)

  wCD  = weighted cluster diversity — each neighbor weighted
         by semantic DISTANCE to home community
         (distant neighbors count more than close ones)

  BC   = betweenness centrality — how often x sits on
         shortest paths between other papers
         (now meaningful because graph is dense enough)

WHY WEIGHT BY COMMUNITY DISTANCE:
  In the original corpus all communities were biomedical
  and adjacent (distance ~0.1-0.2). Every crossing looked
  equal. In the new corpus CS and medicine are genuinely
  distant (distance ~0.7-0.9). A paper bridging those
  fields should score dramatically higher than one bridging
  two medical subfields. The weighting captures this.

WHY THREE SIGNALS INSTEAD OF TWO:
  With 27,424 edges in a 3,599-paper graph, betweenness
  centrality is now computable and meaningful. It was
  abandoned in the original project because only 6 papers
  had nonzero scores in the sparse 2,589-edge graph.
  Now it is a genuine independent signal.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "phase7_ingestion.py"))
from db import get_session

INPUT_DIR  = "data/processed/phase9"
OUTPUT_DIR = "data/processed/phase10"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPECTER_MODEL = "allenai-specter"
BC_SAMPLES    = 500   # k for approximate betweenness
                      # exact is too slow; 500 samples is accurate
                      # enough for ranking purposes


# ══════════════════════════════════════════════════════
# STAGE A — LOAD GRAPH AND PARTITION
# ══════════════════════════════════════════════════════

def load_data():
    print("Loading graph and partition...")

    with open(f"{INPUT_DIR}/multisource_graph.pkl", "rb") as f:
        G = pickle.load(f)

    with open(f"{INPUT_DIR}/partition.pkl", "rb") as f:
        partition = pickle.load(f)

    papers_df = pd.read_parquet(
        f"{INPUT_DIR}/papers_with_communities.parquet"
    )

    print(f"  Graph: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")
    print(f"  Communities: {len(set(partition.values())):,}")
    print(f"  Papers: {len(papers_df):,}")

    return G, partition, papers_df


# ══════════════════════════════════════════════════════
# STAGE B — COMMUNITY CENTROID EMBEDDINGS
# ══════════════════════════════════════════════════════

def build_community_centroids(papers_df, model, session):
    """
    For each community, compute the average SPECTER
    embedding of all its papers. This centroid vector
    represents what that research community talks about.

    Centroid is used to compute semantic DISTANCE between
    communities: distance = 1 - cosine_similarity(A, B)

    Communities sharing vocabulary score distance ~0.1
    Genuinely distant communities score distance ~0.7-0.9
    """
    print("\nBuilding community centroid embeddings...")

    # Load abstracts for all papers
    result = session.execute(text("""
        SELECT paper_id, title, abstract
        FROM papers
        WHERE abstract IS NOT NULL
    """))
    abstract_map = {
        r[0]: f"{r[1] or ''} [SEP] {r[2]}"
        for r in result.fetchall()
    }

    # Group papers by community
    community_papers = defaultdict(list)
    for _, row in papers_df.iterrows():
        if pd.notna(row["community"]):
            community_papers[row["community"]].append(
                row["paper_id"]
            )

    centroids = {}
    communities = list(community_papers.keys())

    print(f"  Computing centroids for "
          f"{len(communities):,} communities...")

    for comm in tqdm(communities, desc="Centroids"):
        paper_ids = community_papers[comm]

        # Get texts for papers in this community
        texts = [
            abstract_map[pid]
            for pid in paper_ids
            if pid in abstract_map
        ]

        if not texts:
            continue

        # Encode all papers in this community
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # Centroid = mean of all embeddings
        centroids[comm] = embeddings.mean(axis=0)

    print(f"  Centroids built: {len(centroids):,}")
    return centroids


def compute_community_distances(centroids):
    """
    Compute semantic distance between every pair of
    communities that appear in our data.

    distance(A, B) = 1 - cosine_similarity(centroid_A, centroid_B)

    Returns a dict: (comm_A, comm_B) → distance float

    We only compute distances for community pairs that
    actually appear as neighbor relationships in the graph —
    not all N² pairs, which would be 449² = 200,000+ pairs.
    """
    print("\n  Computing inter-community distances...")

    comm_list = list(centroids.keys())
    comm_vecs  = np.array([centroids[c] for c in comm_list])

    # Compute full similarity matrix
    # shape: (n_communities, n_communities)
    sim_matrix = cosine_similarity(comm_vecs, comm_vecs)

    # Convert to distance dict
    distances = {}
    for i, c1 in enumerate(comm_list):
        for j, c2 in enumerate(comm_list):
            if i != j:
                dist = float(1.0 - sim_matrix[i][j])
                distances[(c1, c2)] = dist

    print(f"  Distance pairs computed: {len(distances):,}")

    # Show distance range as sanity check
    all_dists = list(distances.values())
    print(f"  Min distance: {min(all_dists):.3f} "
          f"(very similar communities)")
    print(f"  Max distance: {max(all_dists):.3f} "
          f"(most different communities)")
    print(f"  Mean distance: {np.mean(all_dists):.3f}")

    return distances


# ══════════════════════════════════════════════════════
# STAGE C — WEIGHTED BRIDGE SCORES
# ══════════════════════════════════════════════════════

def compute_weighted_cds(G, partition, distances):
    """
    Weighted Citation Diversity Score (wCDS).

    For each paper X:
      For each paper Y that cites X (in-neighbors):
        Get Y's community
        Get distance between X's community and Y's community
        wCDS(X) += distance × log(1 + total_real_citers)

    Papers cited from distant communities score much higher
    than papers cited only from nearby communities.

    Why in-neighbors (papers that cite X) not out-neighbors:
      We want to know which communities RECOGNIZE paper X
      as useful. Citations coming IN to X represent fields
      that found X relevant to their work.
    """
    print("\nComputing weighted citation diversity scores...")

    wCDS = {}
    home_comm = {node: partition.get(node) for node in G.nodes()}

    for node in tqdm(G.nodes(), desc="wCDS"):
        node_comm = home_comm.get(node)
        if not node_comm:
            wCDS[node] = 0.0
            continue

        # Papers that cite this node (in-neighbors)
        citers = list(G.predecessors(node))
        if not citers:
            wCDS[node] = 0.0
            continue

        real_citers = [
            c for c in citers
            if home_comm.get(c) is not None
        ]

        if len(real_citers) < 2:
            wCDS[node] = 0.0
            continue

        # Sum of distances to citing communities
        total_weighted = 0.0
        seen_comms     = set()

        for citer in real_citers:
            citer_comm = home_comm.get(citer)
            if not citer_comm or citer_comm == node_comm:
                continue
            if citer_comm in seen_comms:
                continue
            seen_comms.add(citer_comm)

            dist = distances.get((node_comm, citer_comm), 0.0)
            total_weighted += dist

        weight = np.log1p(len(real_citers))
        wCDS[node] = total_weighted * weight

    return wCDS


def compute_weighted_cd(G, partition, distances):
    """
    Weighted Cluster Diversity (wCD).

    For each paper X:
      For each neighbor Y (both directions, undirected):
        Get Y's community
        Get distance between X's community and Y's community
        wCD(X) += distance

    Papers structurally surrounded by distant communities
    score higher than papers surrounded by nearby communities.

    Uses UNDIRECTED graph because neighborhood composition
    is symmetric — whether X cites Y or Y cites X, they
    are neighbors in the research landscape.
    """
    print("\nComputing weighted cluster diversity scores...")

    G_undirected = G.to_undirected()
    home_comm    = {node: partition.get(node) for node in G.nodes()}
    wCD          = {}

    for node in tqdm(G.nodes(), desc="wCD"):
        node_comm = home_comm.get(node)
        if not node_comm:
            wCD[node] = 0.0
            continue

        neighbors = list(G_undirected.neighbors(node))
        if not neighbors:
            wCD[node] = 0.0
            continue

        total_dist = 0.0
        seen_comms = set()

        for neighbor in neighbors:
            n_comm = home_comm.get(neighbor)
            if not n_comm or n_comm == node_comm:
                continue
            if n_comm in seen_comms:
                continue
            seen_comms.add(n_comm)

            dist = distances.get((node_comm, n_comm), 0.0)
            total_dist += dist

        wCD[node] = total_dist

    return wCD


def compute_betweenness(G):
    """
    Approximate betweenness centrality using k=500 samples.

    With 27,424 edges in a 3,599-node graph, betweenness
    is now meaningful — there are real paths between papers
    for the algorithm to count.

    Uses the DIRECTED graph because citation direction
    determines which paths are valid.
    normalized=True puts scores in [0,1] range.
    seed=42 for reproducibility.
    """
    print(f"\nComputing betweenness centrality "
          f"(k={BC_SAMPLES} samples)...")
    print("  This may take 3-5 minutes...")

    bc = nx.betweenness_centrality(
        G,
        k=BC_SAMPLES,
        normalized=True,
        seed=42
    )

    nonzero = sum(1 for v in bc.values() if v > 0)
    print(f"  Papers with BC > 0: {nonzero:,}")
    print(f"  Max BC: {max(bc.values()):.6f}")

    return bc


def normalize_scores(scores_dict):
    """
    Normalize a dict of scores to [0,1] range.
    Divides every value by the maximum value.
    Papers with score = 0 stay at 0.
    Paper with highest score gets exactly 1.0.
    """
    values = list(scores_dict.values())
    max_val = max(values) if values else 1.0
    if max_val == 0:
        return {k: 0.0 for k in scores_dict}
    return {k: v / max_val for k, v in scores_dict.items()}


def compute_bridge_scores(wCDS, wCD, bc):
    """
    Combine three normalized signals multiplicatively.

    BS_v2 = norm_wCDS × norm_wCD × norm_BC

    Multiplication means ALL THREE must be high.
    A paper scoring 0 on any one signal gets BS_v2 = 0.

    This is stricter than the original two-signal formula
    and produces a cleaner separation between genuine
    bridges and papers that are merely popular or peripheral.
    """
    norm_wcds = normalize_scores(wCDS)
    norm_wcd  = normalize_scores(wCD)
    norm_bc   = normalize_scores(bc)

    bridge_scores = {}
    for paper_id in wCDS:
        s1 = norm_wcds.get(paper_id, 0.0)
        s2 = norm_wcd.get(paper_id, 0.0)
        s3 = norm_bc.get(paper_id, 0.0)
        bridge_scores[paper_id] = s1 * s2 * s3

    return bridge_scores, norm_wcds, norm_wcd, norm_bc


# ══════════════════════════════════════════════════════
# STAGE D — SAVE RESULTS TO DATABASE AND PARQUET
# ══════════════════════════════════════════════════════

def save_results(papers_df, bridge_scores, norm_wcds,
                 norm_wcd, norm_bc, wCDS, wCD, bc, session):
    """
    Attach all bridge score columns to the papers dataframe
    and write results to PostgreSQL and parquet.
    """
    print("\nSaving results...")

    papers_df["weighted_cds"]       = papers_df["paper_id"].map(wCDS).fillna(0)
    papers_df["weighted_cd"]        = papers_df["paper_id"].map(wCD).fillna(0)
    papers_df["betweenness_v2"]     = papers_df["paper_id"].map(bc).fillna(0)
    papers_df["norm_wcds"]          = papers_df["paper_id"].map(norm_wcds).fillna(0)
    papers_df["norm_wcd"]           = papers_df["paper_id"].map(norm_wcd).fillna(0)
    papers_df["norm_bc"]            = papers_df["paper_id"].map(norm_bc).fillna(0)
    papers_df["bridge_score_v2"]    = papers_df["paper_id"].map(bridge_scores).fillna(0)

    # Flag top 10% as bridge candidates
    threshold = papers_df["bridge_score_v2"].quantile(0.90)
    papers_df["is_bridge_v2"] = papers_df["bridge_score_v2"] >= threshold

    nonzero = (papers_df["bridge_score_v2"] > 0).sum()
    print(f"  Papers with bridge_score_v2 > 0: {nonzero:,}")
    print(f"  Top 10% threshold: {threshold:.6f}")
    print(f"  Bridge candidates: "
          f"{papers_df['is_bridge_v2'].sum():,}")

    # Show top 10 bridge papers
    top10 = papers_df.nlargest(10, "bridge_score_v2")[
        ["paper_id", "source", "community",
         "weighted_cds", "weighted_cd",
         "betweenness_v2", "bridge_score_v2"]
    ]
    print("\nTop 10 bridge papers (v2):")
    print(top10.to_string(index=False))

    # Save parquet
    out_path = f"{OUTPUT_DIR}/papers_with_bridge_scores.parquet"
    papers_df.to_parquet(out_path, index=False)
    print(f"\nSaved → {out_path}")

    # Update PostgreSQL papers table with bridge scores
    print("Updating PostgreSQL...")

    # Add columns if they don't exist
    for col, dtype in [
        ("bridge_score_v2", "FLOAT"),
        ("is_bridge_v2",    "BOOLEAN"),
        ("weighted_cds",    "FLOAT"),
        ("weighted_cd",     "FLOAT"),
        ("betweenness_v2",  "FLOAT"),
    ]:
        try:
            session.execute(text(
                f"ALTER TABLE papers ADD COLUMN IF NOT EXISTS "
                f"{col} {dtype} DEFAULT 0"
            ))
            session.commit()
        except Exception:
            session.rollback()

    # Batch update
    batch_size = 200
    rows = papers_df[["paper_id", "bridge_score_v2",
                       "is_bridge_v2", "weighted_cds",
                       "weighted_cd", "betweenness_v2"]].to_dict("records")

    for i in tqdm(range(0, len(rows), batch_size),
                  desc="DB update"):
        batch = rows[i:i+batch_size]
        for row in batch:
            session.execute(text("""
                UPDATE papers SET
                    bridge_score_v2 = :bs,
                    is_bridge_v2    = :ib,
                    weighted_cds    = :wcds,
                    weighted_cd     = :wcd,
                    betweenness_v2  = :bc
                WHERE paper_id = :pid
            """), {
                "bs":   row["bridge_score_v2"],
                "ib":   bool(row["is_bridge_v2"]),
                "wcds": row["weighted_cds"],
                "wcd":  row["weighted_cd"],
                "bc":   row["betweenness_v2"],
                "pid":  row["paper_id"],
            })
        session.commit()

    print("PostgreSQL updated.")
    return papers_df


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def run():
    session = get_session()

    # Stage A
    print("=" * 55)
    print("STAGE A: Loading data")
    print("=" * 55)
    G, partition, papers_df = load_data()

    # Stage B — centroids and distances
    print("\n" + "=" * 55)
    print("STAGE B: Community centroid embeddings")
    print("=" * 55)
    print("Loading SPECTER model...")
    model = SentenceTransformer(SPECTER_MODEL)

    centroids = build_community_centroids(
        papers_df, model, session
    )
    distances = compute_community_distances(centroids)

    # Stage C — three signals
    print("\n" + "=" * 55)
    print("STAGE C: Computing bridge score signals")
    print("=" * 55)

    wCDS = compute_weighted_cds(G, partition, distances)
    wCD  = compute_weighted_cd(G, partition, distances)
    bc   = compute_betweenness(G)

    bridge_scores, norm_wcds, norm_wcd, norm_bc = \
        compute_bridge_scores(wCDS, wCD, bc)

    nonzero_bs = sum(1 for v in bridge_scores.values() if v > 0)
    print(f"\nBridge scores computed:")
    print(f"  Papers with score > 0: {nonzero_bs:,}")
    print(f"  Max score: {max(bridge_scores.values()):.6f}")

    # Stage D — save
    print("\n" + "=" * 55)
    print("STAGE D: Saving results")
    print("=" * 55)

    papers_df = save_results(
        papers_df, bridge_scores, norm_wcds,
        norm_wcd, norm_bc, wCDS, wCD, bc, session
    )

    print("\n" + "=" * 55)
    print("Phase 10 complete.")
    print("=" * 55)
    print("Next: Phase 11 — regression model retraining")

    session.close()


if __name__ == "__main__":
    run()