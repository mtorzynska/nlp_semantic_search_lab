### === Exact inner-product search index: faiss IndexFlatIP with a torch fallback ===
# Index-choice guideline: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index


import numpy as np
import torch
from typing import Any

try:
    import faiss as _faiss

    faiss_available: bool = True
except ImportError:  # wheel missing -> fallback path (Mac ARM under uv)
    _faiss = None  # type: ignore[assignment]
    faiss_available = False


def _to_contiguous_f32(matrix: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Normalize any accepted input to contiguous float32 numpy (faiss requires exactly that)."""
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()
    return np.ascontiguousarray(matrix, dtype=np.float32)


class FlatIPIndex:
    """Exact inner-product index (IndexFlatIP semantics) over a small corpus."""

    def __init__(self, embeddings: "torch.Tensor | np.ndarray") -> None:
        """Store corpus embeddings (n, d); faiss.IndexFlatIP + add, or a torch tensor if faiss is unavailable."""
        matrix = _to_contiguous_f32(embeddings)
        self.dim: int = matrix.shape[1]
        self._index: Any = None  # untyped C-extension boundary
        self._embeddings: torch.Tensor | None = None
        if faiss_available and _faiss is not None:
            index = _faiss.IndexFlatIP(self.dim)
            index.add(matrix)
            self._index = index
        else:
            self._embeddings = torch.from_numpy(matrix)

    def search(
        self, query_embeddings: "torch.Tensor | np.ndarray", top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (scores, indices), both (nq, top_k), descending by score."""
        queries = _to_contiguous_f32(query_embeddings)
        if self._index is not None:
            result = self._index.search(queries, top_k)
            assert isinstance(result, tuple) and len(result) == 2
            scores, indices = (np.asarray(result[0]), np.asarray(result[1]))
            return scores, indices
        return self._torch_search(torch.from_numpy(queries), top_k)

    # --- Torch fallback ---
    def _torch_search(
        self, queries: torch.Tensor, top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Brute-force inner product with torch.topk; corpus is small (~19k) so exhaustive search is cheap."""
        assert self._embeddings is not None
        scores, indices = torch.topk(queries @ self._embeddings.T, k=top_k, dim=1)
        return scores.numpy(), indices.numpy()


def build_index(embeddings: "torch.Tensor | np.ndarray") -> FlatIPIndex:
    """One-call constructor for the notebook; same semantics as FlatIPIndex.__init__."""
    return FlatIPIndex(embeddings)
