"""Cross-encoder reranking: host-resident texts, token-budgeted device batches.

`CrossEncoder.predict` takes all pairs at once, materialises them as one Python list,
length-sorts the whole list in the main process and then tokenises serially while the
device idles. For 87 599 queries x top-5 that is 437 995 pairs and a fixed
`batch_size` that ignores sequence length — the fixed count is what blows up VRAM,
because activations scale with batch x sequence, not with pair count.

This module instead streams:
    dataset item  -> (question, corpus[rankings[row, col]]) built on demand, no list
    batch sampler -> length-bucketed batches closed at MAX_BATCH_TOKENS
    workers       -> tokenisation in `num_workers` processes, overlapped with compute
    main thread   -> non-blocking H2D of one batch, forward, scores back to host
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import numpy as np
import torch
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from part1.config import (
    CROSS_ENCODER_BATCH_SIZE,
    MAX_BATCH_TOKENS,
    NUM_WORKERS
)

# BERT mappuje ~4 chars per token, daje tu do określenia ogólnego kosztu wykonania na GPU.
CHARS_PER_TOKEN = 4


class _PairDataset(Dataset[tuple[str, str]]):
    """Query-document pairs addressed by flat index"""

    def __init__(
        self,
        questions: Sequence[str],
        corpus: Sequence[str],
        rankings: np.ndarray,
    ) -> None:
        """Store references to the caller's sequences (no copies)."""
        self.questions = questions
        self.corpus = corpus
        self.rankings = rankings
        self.top_k = rankings.shape[1]

    def __len__(self) -> int:
        """Total number of pairs."""
        return int(self.rankings.size)

    def __getitem__(self, index: int) -> tuple[str, str]:
        """One (question, document) pair."""
        row, col = divmod(int(index), self.top_k)
        return self.questions[row], self.corpus[int(self.rankings[row, col])]


def _pair_token_count_estimator(
    questions: Sequence[str],
    corpus: Sequence[str],
    rankings: np.ndarray,
) -> np.ndarray:
    """(n_queries * top_k,) estimated token counts."""
    query_chars = np.fromiter(
        (len(q) for q in questions), dtype=np.int64, count=len(questions)
    )
    corpus_chars = np.fromiter(
        (len(d) for d in corpus), dtype=np.int64, count=len(corpus)
    )
    pair_chars = query_chars[:, None] + corpus_chars[rankings]
    return np.ceil(pair_chars / CHARS_PER_TOKEN).astype(np.int64).ravel()


def _token_limited_batch_spliter(
    lengths: np.ndarray,
    *,
    batch_size: int,
    max_batch_tokens: int,
) -> list[np.ndarray]:
    """Split flat pair indices into length-sorted batches under a token budget.

    Sorting ascending keeps padding low (only pairs of similar length share a batch);
    the summed-token cap is what bounds device memory. A single pair longer than the
    budget still gets its own batch of one.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    
    if max_batch_tokens <= 0:
        raise ValueError(f"max_batch_tokens must be positive, got {max_batch_tokens}")

    order = np.argsort(lengths, kind="stable")
    batches: list[np.ndarray] = []
    current: list[int] = []
    used = 0

    for index in order.tolist():
        cost = int(min(lengths[index], max_batch_tokens))
        if current and (len(current) >= batch_size or used + cost > max_batch_tokens):
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
            used = 0
        current.append(index)
        used += cost

    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


### === Inference ===


def _to_device(
    features: dict[str, object],
    device: torch.device,
    *,
    non_blocking: bool,
) -> dict[str, object]:
    """Move tensor values of a preprocessed batch; keep strings (e.g. `modality`)."""
    return {
        key: value.to(device, non_blocking=non_blocking)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in features.items()
    }


def rerank(
    model: CrossEncoder,
    questions: Sequence[str],
    corpus: Sequence[str],
    rankings: torch.Tensor | np.ndarray,
    *,
    batch_size: int = CROSS_ENCODER_BATCH_SIZE,
    max_batch_tokens: int = MAX_BATCH_TOKENS,
    num_workers: int = NUM_WORKERS,
    device: str | torch.device | None = None,
    show_progress_bar: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Rerank previously retrieved documents using a cross-encoder.

    Args:
        model: pretrained CrossEncoder
        questions: original query texts
        corpus: document collection
        rankings: retrieved document IDs, shape (n_queries, top_k)
        batch_size: upper bound on pairs per batch (count only; see max_batch_tokens)
        max_batch_tokens: token budget per batch, the actual device-memory bound
        num_workers: tokenisation worker processes (0 = tokenise in the main process)
        device: compute device; None = the device the model is already on
        show_progress_bar: draw a tqdm over batches

    Returns:
        scores: cross-encoder scores, shape (n_queries, top_k), best-first
        rankings: reranked document IDs, shape (n_queries, top_k), best-first
    """
    if isinstance(rankings, torch.Tensor):
        rankings = rankings.detach().cpu().numpy()

    rankings = np.asarray(rankings)
    if rankings.ndim != 2:
        raise ValueError("rankings must have shape (n_queries, top_k)")
    if len(questions) != rankings.shape[0]:
        raise ValueError("Number of questions must match rankings")
    if rankings.size == 0:
        raise ValueError("rankings must be non-empty")

    n_queries, top_k = rankings.shape
    rankings = rankings.astype(np.int64, copy=False)

    target = torch.device(device) if device is not None else torch.device(model.device)
    model.eval()

    dataset = _PairDataset(questions, corpus, rankings)
    lengths = _pair_token_count_estimator(questions, corpus, rankings)
    batches = _token_limited_batch_spliter(
        lengths, batch_size=batch_size, max_batch_tokens=max_batch_tokens
    )

    loader_kwargs: dict[str, object] = {
        "batch_sampler": batches,
        "num_workers": num_workers,
        "collate_fn": model.preprocess, # CPU work
        "pin_memory": target.type == "cuda",
    }

    if num_workers > 0 and sys.platform != "win32":
        loader_kwargs["multiprocessing_context"] = "fork"

    loader = DataLoader(dataset, **loader_kwargs)  # type: ignore[arg-type]

    flat_scores = np.empty(n_queries * top_k, dtype=np.float32)
    non_blocking = target.type == "cuda" # na mps batchowanie jest synchroniczne

    processed = 0
    with torch.inference_mode():
        progress = tqdm(loader, total=len(batches), disable=not show_progress_bar, desc="Rerank")
        for indices, features in zip(batches, progress):
            features = _to_device(features, target, non_blocking=non_blocking)
            scores = model(features)["scores"]

            if model.activation_fn is not None:
                scores = model.activation_fn(scores)
            if model.num_labels == 1 and scores.ndim > 1:
                scores = scores.squeeze(-1)

            flat_scores[indices] = scores.float().to("cpu").numpy()
            processed += 1

    if processed != len(batches):
        raise RuntimeError(f"executed {processed} of {len(batches)} batches")

    scores = flat_scores.reshape(n_queries, top_k)
    order = np.argsort(-scores, axis=1, kind="stable")
    reranked_ids = np.take_along_axis(rankings, order, axis=1)
    reranked_scores = np.take_along_axis(scores, order, axis=1)

    return torch.from_numpy(reranked_scores), torch.from_numpy(reranked_ids)
