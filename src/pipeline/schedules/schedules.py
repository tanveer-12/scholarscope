"""
Three automated schedules for continuous corpus growth.
"""

from dagster import (
    ScheduleDefinition,
    define_asset_job,
    AssetSelection,
)

# ── Daily: recent papers ──────────────────────────────────────
daily_job = define_asset_job(
    name="daily_recent_ingestion",
    selection=AssetSelection.assets("recent_papers"),
)

daily_schedule = ScheduleDefinition(
    job=daily_job,
    cron_schedule="0 3 * * *",   # 3 AM every day
    name="daily_recent_papers",
    description="Fetch papers published in last 48 hours from OpenAlex",
)

# ── Monthly: core papers + citation refresh ───────────────────
monthly_job = define_asset_job(
    name="monthly_core_ingestion",
    selection=AssetSelection.assets(
        "core_papers",
        "refresh_citation_counts",
        "supplementary_semantic_scholar",
    ),
)

monthly_schedule = ScheduleDefinition(
    job=monthly_job,
    cron_schedule="0 2 1 * *",   # 2 AM on 1st of each month
    name="monthly_core_papers",
    description="Fetch core papers, refresh citations, supplement from SS",
)

# ── Weekly: arXiv semantic pool ───────────────────────────────
weekly_job = define_asset_job(
    name="weekly_arxiv_ingestion",
    selection=AssetSelection.assets("semantic_only_arxiv"),
)

weekly_arxiv_schedule = ScheduleDefinition(
    job=weekly_job,
    cron_schedule="0 4 * * 0",   # 4 AM every Sunday
    name="weekly_arxiv_papers",
    description="Fetch recent arXiv preprints for semantic pool",
)