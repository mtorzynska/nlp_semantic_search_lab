"""Dokładny top-k wyszukiwania iloczynu skalarnego z cappem na pamięć"""

from __future__ import annotations

import numpy as np
import torch


def host_matrix(matrix: "torch.Tensor | np.ndarray", name: str) -> torch.Tensor:
    """Normalizacja i convert do contiguous float32 CPU tensor."""
    if isinstance(matrix, np.ndarray):

        if matrix.ndim != 2:
            raise ValueError(f"{name} must be 2-D, got shape {matrix.shape}")
        
        array = np.ascontiguousarray(matrix, dtype=np.float32)
        return torch.from_numpy(array)
    
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {matrix.shape}")
    
    if matrix.device.type != "cpu":
        raise ValueError(
            f"{name} must stay on the host, got device={matrix.device}; "
            "move blocks to the device inside the search instead"
        )
    return matrix.detach().to(torch.float32).contiguous() # CAP na float32


def topk_search(
    queries: "torch.Tensor | np.ndarray",
    corpus: "torch.Tensor | np.ndarray",
    top_k: int,
    *,
    device: str | torch.device = "cpu",
    query_chunk_size: int = 2048,
    corpus_chunk_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inner-product top-k, streamed through the device block-wise.

    Args:
        queries: (n_queries, dim) host embeddings.
        corpus: (n_docs, dim) host embeddings.
        top_k: candidates per query.
        device: compute device; only blocks are ever copied there.
        query_chunk_size: rows of `queries` resident on the device at a time.
        corpus_chunk_size: rows of `corpus` resident at a time; None = whole corpus
            (one block), which costs `query_chunk_size * n_docs * 4` bytes.

    Returns:
        scores: (n_queries, top_k) float32 CPU tensor, descending.
        ids: (n_queries, top_k) int64 CPU tensor of corpus row ids.
    """
    host_queries = host_matrix(queries, "queries")
    host_corpus = host_matrix(corpus, "corpus")

    n_queries, dim = host_queries.shape
    n_docs, corpus_dim = host_corpus.shape

    if dim != corpus_dim:
        raise ValueError(
            f"dimension mismatch: queries {tuple(host_queries.shape)} vs "
            f"corpus {tuple(host_corpus.shape)}"
        )
    
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    
    if top_k > n_docs:
        raise ValueError(f"top_k={top_k} exceeds corpus size {n_docs}")
    
    if query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    
    if corpus_chunk_size is None:
        corpus_chunk_size = n_docs

    if corpus_chunk_size <= 0:
        raise ValueError(f"corpus_chunk_size must be positive, got {corpus_chunk_size}")

    target = torch.device(device)

    out_scores = torch.empty((n_queries, top_k), dtype=torch.float32)
    out_ids = torch.empty((n_queries, top_k), dtype=torch.int64)

    with torch.inference_mode():
        for q_start in range(0, n_queries, query_chunk_size):
            q_end = min(q_start + query_chunk_size, n_queries)
            q_block = host_queries[q_start:q_end].to(target)

            best_scores: torch.Tensor | None = None
            best_ids: torch.Tensor | None = None

            for c_start in range(0, n_docs, corpus_chunk_size):
                c_end = min(c_start + corpus_chunk_size, n_docs)
                c_block = host_corpus[c_start:c_end].to(target)

                block = q_block @ c_block.T
                scores, local_ids = block.topk(top_k, dim=1)
                ids = local_ids + c_start
                del block, local_ids, c_block

                if best_scores is None:
                    best_scores, best_ids = scores, ids
                    continue

                assert best_ids is not None
                merged_scores = torch.cat((best_scores, scores), dim=1)
                merged_ids = torch.cat((best_ids, ids), dim=1)
                best_scores, keep = merged_scores.topk(top_k, dim=1)
                best_ids = merged_ids.gather(1, keep)
                del merged_scores, merged_ids, scores, ids

            assert best_scores is not None and best_ids is not None
            out_scores[q_start:q_end] = best_scores.to("cpu")
            out_ids[q_start:q_end] = best_ids.to("cpu")
            del q_block, best_scores, best_ids

    return out_scores, out_ids
