"""
config.py
-----
Shared config for all ingestion scripts
every script imports from here so settings are
changed in one place only
"""

import os
from dotenv import load_dotenv

"""
load_dotenv() reads your .env file and puts every line into os.environ so 
os.getenv() can find them. 
without this line, your keys would not be accessible.
"""
load_dotenv()

POSTGRES_URL = os.getenv("POSTGRES_URL")

# API Keys
# The second argument "" is a fallback default so
# the script does not crash if a key is not set yet
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY","")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
NCBI_API_KEY = os.getenv("NCBI_API_KEY","")


# collection size : how many papers to collect per total
# 1000 - small test batch. change to 10000 to medium.
TARGET_PER_SOURCE = 1000

# OpenAlex tags every paper with "concepts" — these are
# their internal IDs for broad research areas.
# We fetch papers from each concept to get cross-domain coverage.
# Format: "C" + numeric ID from OpenAlex concept taxonomy.
OPENALEX_CONCEPTS = [
    "C41008148",   # Computer Science
    "C86803240",   # Biology
    "C71924100",   # Medicine
    "C162324750",  # Economics
    "C185592680",  # Chemistry
    "C121332964",  # Physics
]

# arXiv organizes papers into subject categories.
# Each code maps to a specific research area.
# cs.AI = Artificial Intelligence
# cs.LG = Machine Learning
# cs.CL = Computation and Language (NLP)
# q-bio.NC = Neurons and Cognition (neuroscience)
# q-bio.GN = Genomics
# econ.GN = General Economics
# physics.med-ph = Medical Physics
ARXIV_CATEGORIES = [
    "cs.AI",
    "cs.LG",
    "cs.CL",
    "q-bio.NC",
    "q-bio.GN",
    "econ.GN",
    "physics.med-ph",
]

# These are plain-language searches sent to PubMed.
# Each returns papers matching that topic.
# We collect papers across all three to get biomedical
# papers that overlap with other domains.
PUBMED_SEARCH_TERMS = [
    "machine learning AND clinical trials",
    "genomics AND epidemiology",
    "neural networks AND radiology",
]

# SS uses plain-text field names (not IDs like OpenAlex).
# These are passed as search queries to their paper search endpoint.
SS_FIELDS_OF_STUDY = [
    "Computer Science",
    "Biology",
    "Medicine",
    "Economics",
    "Physics",
]