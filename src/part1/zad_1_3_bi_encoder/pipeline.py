from collections.abc import Sequence
from pathlib import Path

import datasets
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from part1.config import BATCH_SIZE, Backend


# Embedding build/load: embeddings in RAM, on device only one batch at a time

def embed_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    device: Backend,
    batch_size: int = BATCH_SIZE,
    show_progress: bool = True,
) -> np.ndarray:
    """(n, hidden_dim) float32 host array: fp32, L2-unnormalized (dot-v1 raw dot).

    One `encode` call, not a Python loop over batches: `encode` keeps exactly one
    batch of token ids and hidden states on the device, and returns the result to
    host memory. Losing that loop also loses 685 per-call device syncs.
    """
    if not texts:
        raise ValueError("texts must be non-empty")

    embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        device=device,
        show_progress_bar=show_progress,
    )
    return np.ascontiguousarray(embeddings, dtype=np.float32)


def embeddings_tensor(
    ds: datasets.Dataset,
    *,
    column: str = "embeddings",
) -> torch.Tensor:
    """
    Convert a dataset column of embeddings to a PyTorch tensor.

    Dataset uses Arrow that implicitly promotes float32 to float64 when converting to a Python list.
    This function returns a (n, d) float32 CPU tensor read from the Arrow cache.
    """
    matrix = np.asarray(ds[column], dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(
            f"expected a 2-D embedding column, got shape {matrix.shape}"
        )
    return torch.from_numpy(matrix)


def load_or_build_embeddings(
    model: SentenceTransformer,
    texts: Sequence[str],
    cache_dir: Path,
    *,
    device: Backend,
    batch_size: int = BATCH_SIZE,
) -> datasets.Dataset:
    """Load from cache or calculate embeddings and save to disk. Returns a Dataset with columns `text` and `embeddings`."""

    if cache_dir.exists():
        cached = datasets.load_from_disk(str(cache_dir))
        if not isinstance(cached, datasets.Dataset):
            raise TypeError(
                f"expected a single Dataset under {cache_dir}, got {type(cached).__name__}"
            )
        if cached.num_rows != len(texts):
            raise ValueError(
                f"{cache_dir} holds {cached.num_rows} embeddings, expected {len(texts)}"
            )
        return cached

    embeddings = embed_texts(
        model, texts, device=device, batch_size=batch_size
    )
    ds = datasets.Dataset.from_dict({"text": list(texts), "embeddings": embeddings})
    ds.save_to_disk(str(cache_dir))
    return ds
