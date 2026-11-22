from typing import Sequence

import torch

Rankings = torch.Tensor  # (n_zapytań, k) - int, indeksy dopasowanych dokumentów, best-first
Relevant = torch.Tensor  # (n_zapytań,) - int, id JEDNEGO istotnego dokumentu per zapytanie


def _validate(rankings: Rankings, relevant: Relevant, ks: Sequence[int]) -> None:
    if rankings.dim() != 2 or relevant.dim() != 1:
        raise ValueError(
            f"expected rankings (n_queries, k) 2-D and relevant (n_queries,) 1-D, "
            f"got {rankings.dim()}-D and {relevant.dim()}-D"
        )
    int_dtypes = (torch.int32, torch.int64)
    if rankings.dtype not in int_dtypes:
        raise ValueError(f"rankings must hold integer document ids, got {rankings.dtype}")
    if relevant.dtype not in int_dtypes:
        raise ValueError(f"relevant must hold integer target ids, got {relevant.dtype}")
    if relevant.shape[0] != rankings.shape[0]:
        raise ValueError(f"targets length {relevant.shape[0]} != n_queries {rankings.shape[0]}")
    k_max = max(ks)
    if k_max <= 0:
        raise ValueError(f"cut-offs must be positive, got k_max={k_max}")
    if k_max > rankings.shape[1]:
        raise ValueError(f"k={k_max} exceeds ranking window width {rankings.shape[1]}")


class PrecisionRecallF1:
    """Order-unaware Query-averaged Precision/Recall/F1 at top-k."""

    def __init__(self, ks: Sequence[int] = (1, 3, 5)) -> None:
        """Init metric with top-ks to compute."""
        self.ks = tuple(ks)

    def __call__(self, rankings: Rankings, relevant: Relevant) -> dict[int, dict[str, float]]:
        """{k: {"precision": P@k, "recall": R@k, "f1": F1@k}}.

        Single-target ground truth => exactly one relevant doc per query, so P@k = R@k / k
        and recall is binary per query. No zero-relevant queries can occur by construction.
        """
        _validate(rankings, relevant, self.ks)

        hits = (rankings == relevant.unsqueeze(1)).to(torch.long)
        n_hits = hits.cumsum(dim=1)

        out: dict[int, dict[str, float]] = {}
        for k in self.ks:
            hits_k = n_hits[:, k - 1].to(torch.float64)
            precision = hits_k / float(k)
            recall = hits_k / 1.0  # |relevant set| == 1 for every query
            f1_den = precision + recall
            f1 = torch.where(f1_den > 0, 2.0 * precision * recall / f1_den, torch.zeros_like(f1_den))
            out[k] = {
                "precision": float(precision.mean()),
                "recall": float(recall.mean()),
                "f1": float(f1.mean()),
            }
        return out


class MRR:
    """Windowed Mean Reciprocal Rank."""

    def __call__(self, rankings: Rankings, relevant: Relevant) -> float:
        _validate(rankings, relevant, (1,))

        hits = (rankings == relevant.unsqueeze(1)).to(torch.long)
        is_hit = hits.sum(dim=1) > 0
        first_rank = torch.argmax(hits.to(torch.float64), dim=1).to(torch.float64)
        reciprocal = torch.where(is_hit, 1.0 / (first_rank + 1.0), torch.zeros_like(first_rank))
        return float(reciprocal.mean())
