"""
Dagster entry point. Registers all assets, resources,
and schedules. This is the only file Dagster reads at startup.
"""

import os
from dagster import Definitions, EnvVar

from .assets.ingestion import (
    historical_papers,
    core_papers,
    recent_papers,
    refresh_citation_counts,
    supplementary_semantic_scholar,
    semantic_only_arxiv,
)
from .resources.database import DatabaseResource
from .schedules.schedules import (
    daily_schedule,
    monthly_schedule,
    weekly_arxiv_schedule,
)

defs = Definitions(
    assets=[
        # Tier 1 — OpenAlex primary
        historical_papers,
        core_papers,
        recent_papers,
        refresh_citation_counts,
        # Tier 2 — Supplementary
        supplementary_semantic_scholar,
        # Tier 3 — Semantic only
        semantic_only_arxiv,
    ],
    resources={
        "database": DatabaseResource(
            postgres_url=EnvVar("POSTGRES_URL")
        ),
    },
    schedules=[
        daily_schedule,
        monthly_schedule,
        weekly_arxiv_schedule,
    ],
)

