from pathlib import Path

from part1.config import BI_ENCODER_MODEL_ID, CROSS_ENCODER_MODEL_ID  # noqa: F401

# Paths
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
CACHE_DIR: Path = REPO_ROOT / ".cache" / "part2"
ENV_FILE: Path = REPO_ROOT / ".env"

# Corpus
DATASET_ID: str = "stanfordnlp/imdb"
DATASET_SPLIT: str = "train"
N_REVIEWS: int = 5000 # podzbiór, cały split = 25k
SEED: int = 42

# mpnet ucina na 512 tokenach, recenzje bywają dłuższe -> chunking
CHUNK_WORDS: int = 200
CHUNK_OVERLAP: int = 50

# Retriever
RETRIEVE_K: int = 20 # kandydaci z FAISS
CONTEXT_K: int = 5 # top_k po rerankingu -> prompt

# CLARIN API
CLARIN_BASE_URL: str = "https://services.clarin-pl.eu/api/v1/oapi"
CLARIN_API_KEY_ENV: str = "CLARIN_API_KEY"
CLARIN_MODEL_ENV: str = "CLARIN_MODEL"
LLM_TEMPERATURE: float = 0.0 # deterministycznie -> porównywalne warianty
LLM_MAX_TOKENS: int = 512
LLM_TIMEOUT_S: float = 120.0

JUDGE_MODEL: str = "gemma-4-31b-it"
