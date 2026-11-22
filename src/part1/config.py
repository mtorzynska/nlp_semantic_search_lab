from typing import Literal
from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available

DATASET_ID: str = "sentence-transformers/squad"
TOP_K: int = 5


BATCH_SIZE: int = 128

# --- Host <-> device streaming ---
NUM_WORKERS: int = 4                    # DataLoader workers

CROSS_ENCODER_BATCH_SIZE: int = 1024    # Device memory load is   batch * sequence length
MAX_BATCH_TOKENS: int = 16384

QUERY_CHUNK_SIZE: int = 2048            # no-index scan - rows of query embeddings
CORPUS_CHUNK_SIZE: int = 8192           # rows of corpus embeddings

Backend = Literal["cuda", "mps", "cpu"]

# --- Models ---
BI_ENCODER_MODEL_ID: str = (
    "sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
CROSS_ENCODER_MODEL_ID: str = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


def resolve_backend(backend: Backend | None = None) -> Backend:
    """
    Pick compute backend: explicit argument wins.

    Autodetect otherwise:
        - Ubuntu    (CUDA)
        - Arch      (CPU)
        - Win       (CPU)
        - M5 Pro    (MPS)
    """
    if backend is not None:
        return backend
    if cuda_is_available():
        return "cuda"
    if mps_is_available():
        return "mps"
    return "cpu"
