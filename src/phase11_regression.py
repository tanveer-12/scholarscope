"""
phase11_regression.py
----------------------
Retrains the citation regression model on the multi-source
corpus with additional features that capture source
heterogeneity and cross-domain position.

WHY RETRAIN:
  The original model (Phase 4) was trained on 36,823
  biomedical papers with 5 features.
  R² = 0.19 — honest but limited signal.

  The new model trains on 3,599 papers across 4 sources
  spanning CS, biology, medicine, economics, physics.
  New features capture:
    - Source prestige (venue quality differences)
    - Cross-source citations (direct interdisciplinarity signal)
    - Weighted community distance (how cross-domain the paper is)
    - Has full text (discoverability)

TARGET VARIABLE:
  log(1 + num_citations) — same as original.
  num_citations here is the worldwide citation count
  from whichever source reported it (OpenAlex or SS
  have the most complete counts).

RESIDUAL:
  residual = actual_log_citations - predicted_log_citations
  Negative = got fewer citations than predicted = under-cited
  This is the signal the Wilcoxon test in Phase 12 uses.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Windows
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                "phase7_ingestion.py"))
from db import get_session

OUTPUT_DIR = "data/processed/phase11"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_END_YEAR = 2026   # current year for age calculation


# ══════════════════════════════════════════════════════
# STAGE A — LOAD AND ENGINEER FEATURES
# ══════════════════════════════════════════════════════

def load_papers(session):
    """
    Load all papers with their Phase 10 bridge scores
    and community assignments from PostgreSQL.
    """
    print("Loading papers from PostgreSQL...")

    result = session.execute(text("""
        SELECT
            p.paper_id,
            p.source,
            p.year,
            p.num_citations,
            p.has_full_text,
            p.completeness_score,
            p.bridge_score_v2,
            p.weighted_cds,
            p.weighted_cd,
            p.betweenness_v2,
            p.is_bridge_v2,
            p.community_id,
            -- Cross-source citation count:
            -- how many papers from OTHER sources cite this paper
            COUNT(DISTINCT c.citing_paper_id)
                FILTER (WHERE p2.source != p.source)
                AS cross_source_citations
        FROM papers p
        LEFT JOIN citations c
            ON c.cited_paper_id = p.paper_id
            AND c.edge_type = 'soft'
        LEFT JOIN papers p2
            ON p2.paper_id = c.citing_paper_id
        WHERE
            -- Only papers where citation count is meaningful:
            -- exclude arXiv (no citation data)
            -- exclude papers with zero citations AND no DOI
            -- (likely data quality issues)
            NOT (p.source = 'arxiv')
            AND p.num_citations IS NOT NULL
            AND p.num_citations > 0
        GROUP BY
            p.paper_id, p.source, p.year, p.num_citations,
            p.has_full_text, p.completeness_score,
            p.bridge_score_v2, p.weighted_cds, p.weighted_cd,
            p.betweenness_v2, p.is_bridge_v2, p.community_id
    """))

    rows = result.fetchall()
    cols = [
        "paper_id", "source", "year", "num_citations",
        "has_full_text", "completeness_score",
        "bridge_score_v2", "weighted_cds", "weighted_cd",
        "betweenness_v2", "is_bridge_v2", "community_id",
        "cross_source_citations"
    ]
    df = pd.DataFrame(rows, columns=cols)
    print(f"  Papers loaded: {len(df):,}")
    print(f"  (arXiv excluded — no citation data available)")
    return df


def engineer_features(df):
    """
    Build all features for the regression model.

    FEATURES EXPLAINED:

    age — years since publication
      Most important confounder. A 2010 paper has had
      16 years to accumulate citations. A 2024 paper
      has had 2 years. Without controlling for age,
      recent papers always look under-cited.

    source_prestige — publication venue quality tier
      Papers in top venues get more citations regardless
      of content. Must control for this or prestigious
      venue papers look over-cited by the model.
      Tiers: 3=top venues, 2=peer-reviewed, 1=preprint

    community_size — papers in same community
      Larger research communities have more potential
      citers. A paper in a community of 200 has 200
      peers who might cite it; a paper in a community
      of 5 has 5.

    bridge_score_v2 — our computed bridge signal
      Including this lets the model learn the relationship
      between bridging and citations. The residual then
      captures REMAINING under-citation after controlling
      for bridge status — a cleaner signal.

    weighted_cd — structural cross-community position
      How many distant communities surround this paper.

    cross_source_citations — soft edges from other sources
      A direct empirical signal of interdisciplinarity.
      Papers with many cross-source soft edges are
      genuinely relevant to multiple domains.

    has_full_text — discoverability proxy
      Full-text papers are findable via keyword search,
      not just title/abstract. More discoverable = more
      citations independent of content quality.

    completeness_score — data quality indicator
      Papers with more complete metadata are better
      indexed and more likely to be found and cited.
    """
    print("\nEngineering features...")

    # Age
    df["age"] = (DATASET_END_YEAR -
                 df["year"].fillna(DATASET_END_YEAR)).clip(lower=0)

    # Source prestige tiers
    prestige_map = {
        "openalex":         2,   # peer-reviewed, multi-venue
        "pubmed":           2,   # peer-reviewed biomedical
        "semantic_scholar": 2,   # peer-reviewed, multi-domain
        "arxiv":            1,   # preprint, not peer-reviewed
    }
    df["source_prestige"] = df["source"].map(prestige_map).fillna(1)

    # Community size
    comm_sizes = df["community_id"].value_counts().to_dict()
    df["community_size"] = df["community_id"].map(
        comm_sizes
    ).fillna(1)

    # Encode community as integer for LightGBM
    le = LabelEncoder()
    df["community_encoded"] = le.fit_transform(
        df["community_id"].fillna("unknown")
    )

    # Fill NaN in numeric columns
    numeric_cols = [
        "bridge_score_v2", "weighted_cds", "weighted_cd",
        "betweenness_v2", "cross_source_citations",
        "completeness_score"
    ]
    for col in numeric_cols:
        df[col] = df[col].fillna(0)

    df["has_full_text"] = df["has_full_text"].fillna(False).astype(int)
    df["num_citations"] = df["num_citations"].fillna(0).clip(lower=0)

    # Log-transform target
    df["log_citations"] = np.log1p(df["num_citations"])

    # Citation distribution summary
    print(f"\n  Target (num_citations) distribution:")
    print(f"    Mean:   {df['num_citations'].mean():.1f}")
    print(f"    Median: {df['num_citations'].median():.1f}")
    print(f"    Max:    {df['num_citations'].max():.0f}")
    print(f"    Zero:   {(df['num_citations'] == 0).sum():,} papers")
    print(f"    1-10:   {((df['num_citations'] >= 1) & (df['num_citations'] <= 10)).sum():,} papers")
    print(f"    10+:    {(df['num_citations'] > 10).sum():,} papers")

    return df


# ══════════════════════════════════════════════════════
# STAGE B — TRAIN MODEL
# ══════════════════════════════════════════════════════

def train_model(df):
    """
    Train LightGBM on log-transformed citation count.

    STRATIFIED SPLIT:
      We stratify by source so every source appears in
      both train and test sets proportionally.
      Without this, if arxiv is 25% of data, it should
      be 25% of both train and test — not concentrated
      in one side.

    WHY LIGHTGBM:
      Citations are non-linearly related to all features.
      Age × community_size interactions matter.
      LightGBM handles this naturally without manual
      feature engineering.
    """
    feature_cols = [
        "age",
        "source_prestige",
        "community_size",
        "completeness_score",
    ]

    print(f"\nFeatures: {feature_cols}")

    X = df[feature_cols].values
    y = df["log_citations"].values

    # Stratify by source for balanced split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=df["source"]
    )

    print(f"Train: {len(X_train):,}  Test: {len(X_test):,}")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=10,   # lower than original because
                                # smaller dataset (3,599 vs 36,823)
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(100),
        ]
    )

    # Evaluate
    y_pred = model.predict(X_test)
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nModel performance:")
    print(f"  R² score:  {r2:.4f}")
    print(f"  RMSE:      {rmse:.4f} (log-citation units)")

    if r2 > 0.5:
        print("  → Strong predictive power")
    elif r2 > 0.2:
        print("  → Moderate — residuals are usable signal")
    elif r2 > 0.05:
        print("  → Weak but nonzero — residuals still valid")
    else:
        print("  → Very low — citations hard to predict")
        print("    This is common for small diverse corpora")

    # Feature importances
    importances = pd.DataFrame({
        "feature":    feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\nFeature importances:")
    print(importances.to_string(index=False))

    # Save importance plot
    fig, ax = plt.subplots(figsize=(8, 5))
    imp_sorted = importances.sort_values("importance")
    ax.barh(imp_sorted["feature"], imp_sorted["importance"],
            color="steelblue")
    ax.set_xlabel("Importance (LightGBM splits)")
    ax.set_title("Phase 11: Feature importances\n"
                 "Multi-source citation regression")
    plt.tight_layout()
    plot_path = f"{OUTPUT_DIR}/feature_importance_v2.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFeature importance plot saved → {plot_path}")

    return model, feature_cols, r2, rmse


# ══════════════════════════════════════════════════════
# STAGE C — COMPUTE RESIDUALS
# ══════════════════════════════════════════════════════

def compute_residuals(df, model, feature_cols):
    """
    Compute residuals for ALL papers (not just test set).

    We need residuals for every paper to run the
    Wilcoxon test in Phase 12. Train/test split was
    only for evaluating model quality.

    residual = log(1 + actual) - predicted_log
    Negative = under-cited (got less than predicted)
    Positive = over-cited (got more than predicted)
    """
    print("\nComputing residuals for all papers...")

    X_all = df[feature_cols].values
    df["predicted_log"] = model.predict(X_all)
    df["predicted_citations"] = np.expm1(df["predicted_log"])
    df["residual"] = df["log_citations"] - df["predicted_log"]

    # Summary
    print(f"  Mean residual:   {df['residual'].mean():.4f} "
          f"(should be near 0)")
    print(f"  Std residual:    {df['residual'].std():.4f}")
    print(f"  Under-cited:     {(df['residual'] < 0).sum():,} papers")
    print(f"  Over-cited:      {(df['residual'] > 0).sum():,} papers")

    # Bridge vs non-bridge comparison
    bridge     = df[df["is_bridge_v2"] == True]["residual"]
    nonbridge  = df[df["is_bridge_v2"] == False]["residual"]

    print(f"\n  Bridge papers (n={len(bridge):,}):")
    print(f"    Mean residual:   {bridge.mean():.4f}")
    print(f"    Median residual: {bridge.median():.4f}")

    print(f"\n  Non-bridge papers (n={len(nonbridge):,}):")
    print(f"    Mean residual:   {nonbridge.mean():.4f}")
    print(f"    Median residual: {nonbridge.median():.4f}")

    diff = bridge.mean() - nonbridge.mean()
    print(f"\n  Mean difference (bridge - non-bridge): {diff:.4f}")
    if diff < 0:
        print("  → Bridge papers have MORE NEGATIVE residuals")
        print("  → Preliminary evidence: bridge papers under-cited ✓")
    else:
        print("  → Bridge papers have less negative residuals")
        print("  → Unexpected — check bridge score distribution")

    print(f"\n  Papers with residuals: {len(df):,}")
    print(f"  arXiv papers (excluded from test): ~909 (no citation data)")
    print(f"  These will be excluded from Phase 12 Wilcoxon test")
    return df


# ══════════════════════════════════════════════════════
# STAGE D — SAVE RESULTS
# ══════════════════════════════════════════════════════

def save_results(df, session):
    """
    Save residuals to parquet and PostgreSQL.
    The residual column is what Phase 12 Wilcoxon test uses.
    """
    print("\nSaving residuals...")

    # Add residual columns to PostgreSQL
    for col, dtype in [
        ("residual",           "FLOAT"),
        ("predicted_citations", "FLOAT"),
    ]:
        try:
            session.execute(text(
                f"ALTER TABLE papers ADD COLUMN IF NOT EXISTS "
                f"{col} {dtype} DEFAULT 0"
            ))
            session.commit()
        except Exception:
            session.rollback()

    # Batch update residuals
    rows = df[["paper_id", "residual",
               "predicted_citations"]].to_dict("records")

    batch_size = 200
    from tqdm import tqdm
    for i in tqdm(range(0, len(rows), batch_size),
                  desc="Saving residuals"):
        batch = rows[i:i+batch_size]
        for row in batch:
            session.execute(text("""
                UPDATE papers SET
                    residual            = :res,
                    predicted_citations = :pred
                WHERE paper_id = :pid
            """), {
                "res":  float(row["residual"]),
                "pred": float(row["predicted_citations"]),
                "pid":  row["paper_id"],
            })
        session.commit()

    # Save full parquet
    out_path = f"{OUTPUT_DIR}/papers_with_residuals.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved → {out_path}")

    print("\nDone. Phase 11 complete.")
    print("Next: Phase 12 — Wilcoxon test + hidden gems")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def run():
    session = get_session()

    print("=" * 55)
    print("STAGE A: Loading data and engineering features")
    print("=" * 55)
    df = load_papers(session)
    df = engineer_features(df)

    print("\n" + "=" * 55)
    print("STAGE B: Training LightGBM regression model")
    print("=" * 55)
    model, feature_cols, r2, rmse = train_model(df)

    print("\n" + "=" * 55)
    print("STAGE C: Computing residuals for all papers")
    print("=" * 55)
    df = compute_residuals(df, model, feature_cols)

    print("\n" + "=" * 55)
    print("STAGE D: Saving results")
    print("=" * 55)
    save_results(df, session)

    session.close()


if __name__ == "__main__":
    run()