# Retriever: bi-encoder + FAISS (part1) -> opcjonalnie cross-encoder (part1)

from dataclasses import dataclass

import numpy as np
import torch
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from sentence_transformers import CrossEncoder, SentenceTransformer

from part1.config import Backend
from part1.zad_1_3_bi_encoder.faiss_module import FlatIPIndex, build_index
from part1.zad_1_4_cross_encoder.reranker import rerank
from part2.config import CONTEXT_K, RETRIEVE_K
from part2.corpus import LABELS, Corpus


@dataclass(frozen=True)
class Hit:
    doc_id: int  # wiersz w korpusie / indeksie
    text: str
    review_id: int
    chunk_id: int
    label: str
    bi_rank: int  # pozycja po bi-encoderze, 1 = best
    bi_score: float
    ce_score: float | None  # None bez rerankingu


# FAISS IndexFlatIP top RETRIEVE_K -> rerank -> top CONTEXT_K
class SemanticSearch:
    def __init__(
        self,
        bi_encoder: SentenceTransformer,
        corpus: Corpus,
        embeddings: torch.Tensor | np.ndarray,
        *,
        cross_encoder: CrossEncoder | None = None,
        device: Backend = "cpu",
        retrieve_k: int = RETRIEVE_K,
        context_k: int = CONTEXT_K,
    ) -> None:
        if len(corpus) != embeddings.shape[0]:
            raise ValueError(f"corpus has {len(corpus)} chunks, embeddings {embeddings.shape[0]} rows")
        if not 0 < context_k <= retrieve_k:
            raise ValueError(f"need 0 < context_k <= retrieve_k, got {context_k=}, {retrieve_k=}")
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.corpus = corpus
        self.index: FlatIPIndex = build_index(embeddings)
        self.device = device
        self.retrieve_k = retrieve_k
        self.context_k = context_k

    # use_reranker: None = jeśli jest cross_encoder, False = sam bi-encoder (top context_k jak w part1)
    # dedupe: max 1 chunk na recenzję, chunki z overlapem nie zajmują kilku miejsc w top_k
    def search(
        self, query: str, *, use_reranker: bool | None = None, dedupe: bool = False
    ) -> list[Hit]:
        rerank_on = self.cross_encoder is not None if use_reranker is None else use_reranker
        if rerank_on and self.cross_encoder is None:
            raise ValueError("reranking requested but no cross_encoder was given")

        q_emb = self.bi_encoder.encode(
            [query], convert_to_numpy=True, device=self.device, show_progress_bar=False
        )
        k = self.retrieve_k if (rerank_on or dedupe) else self.context_k
        bi_scores, bi_ids = self.index.search(q_emb, k)
        bi_ids_row, bi_scores_row = bi_ids[0], bi_scores[0]
        bi_rank = {int(doc): rank for rank, doc in enumerate(bi_ids_row, start=1)}
        bi_score = {int(doc): float(s) for doc, s in zip(bi_ids_row, bi_scores_row)}

        if rerank_on:
            assert self.cross_encoder is not None
            ce_scores, ce_ids = rerank(
                self.cross_encoder,
                [query],
                self.corpus.texts,
                bi_ids,
                num_workers=0,  # 1 zapytanie - workery tylko spowalniają
                show_progress_bar=False,
            )
            order = [int(d) for d in ce_ids[0]]
            ce: list[float | None] = [float(s) for s in ce_scores[0]]
        else:
            order = [int(d) for d in bi_ids_row]
            ce = [None] * len(order)

        if dedupe:
            seen: set[int] = set()
            keep = []
            for i, doc in enumerate(order):
                if self.corpus.review_ids[doc] not in seen:
                    seen.add(self.corpus.review_ids[doc])
                    keep.append(i)
            order, ce = [order[i] for i in keep], [ce[i] for i in keep]
        order, ce = order[: self.context_k], ce[: self.context_k]

        return [
            Hit(
                doc_id=doc,
                text=self.corpus.texts[doc],
                review_id=self.corpus.review_ids[doc],
                chunk_id=self.corpus.chunk_ids[doc],
                label=LABELS[self.corpus.labels[doc]],
                bi_rank=bi_rank[doc],
                bi_score=bi_score[doc],
                ce_score=score,
            )
            for doc, score in zip(order, ce)
        ]


def hit_to_document(hit: Hit) -> Document:
    return Document(
        page_content=hit.text,
        metadata={
            "doc_id": hit.doc_id,
            "review_id": hit.review_id,
            "chunk_id": hit.chunk_id,
            "label": hit.label,
            "bi_rank": hit.bi_rank,
            "bi_score": hit.bi_score,
            "ce_score": hit.ce_score,
        },
    )


# Adapter LangChain -> SemanticSearch
class PipelineRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    search_engine: SemanticSearch
    use_reranker: bool = True
    dedupe: bool = False

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.search_engine.search(query, use_reranker=self.use_reranker, dedupe=self.dedupe)
        return [hit_to_document(h) for h in hits]
