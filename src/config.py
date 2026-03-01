"""Configuration constants and environment variables."""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "")

# API URLs
OPENFDA_BASE_URL = "https://api.fda.gov/drug/label.json"
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"

# Models
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "indexes"
CACHE_DIR = DATA_DIR / "cache"

# Ingestion
TOP_DRUGS_LIMIT = 500
LABELS_PER_DRUG = 2
API_DELAY = 0.25  # seconds between openFDA calls during bulk fetch

# Sections
SECTIONS_TO_EXTRACT = [
    "drug_interactions",
    "warnings",
    "warnings_and_cautions",
    "adverse_reactions",
    "contraindications",
    "indications_and_usage",
    "boxed_warning",
]
