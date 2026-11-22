# IMDB corpus: podzbiór, czyszczenie, chunking, embeddingi

import re
from dataclasses import dataclass, field

import datasets
import torch
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

from part1.config import Backend
from part1.zad_1_3_bi_encoder import pipeline
from part2.config import (
    CACHE_DIR,
    CHUNK_OVERLAP,
    CHUNK_WORDS,
    DATASET_ID,
    DATASET_SPLIT,
    N_REVIEWS,
    SEED,
)

LABELS: dict[int, str] = {0: "negative", 1: "positive"}
_HTML_BREAK = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


# i-ty element każdej listy = i-ty chunk = i-ty wiersz indeksu
@dataclass
class Corpus:
    texts: list[str] = field(default_factory=list)
    review_ids: list[int] = field(default_factory=list)   # recenzja w podzbiorze
    chunk_ids: list[int] = field(default_factory=list)    # nr chunka w recenzji
    labels: list[int] = field(default_factory=list)       # 0 = neg, 1 = pos
    reviews: list[str] = field(default_factory=list)      # pełne recenzje po czyszczeniu

    def __len__(self) -> int:
        return len(self.texts)


def clean_review(text: str) -> str:
    return _WHITESPACE.sub(" ", _HTML_BREAK.sub(" ", text)).strip()


# Sliding window po słowach: size słów, overlap słów zakładki
def chunk_words(text: str, size: int = CHUNK_WORDS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError(f"need size > 0 and 0 <= overlap < size, got {size=}, {overlap=}")
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


# Split train jest posortowany po etykiecie -> shuffle przed select
def load_reviews(n: int = N_REVIEWS, seed: int = SEED) -> datasets.Dataset:
    ds = load_dataset(DATASET_ID, split=DATASET_SPLIT)
    assert isinstance(ds, datasets.Dataset)
    return ds.shuffle(seed=seed).select(range(min(n, ds.num_rows)))


def build_corpus(
    reviews: datasets.Dataset,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> Corpus:
    corpus = Corpus()
    for review_id, (text, label) in enumerate(zip(reviews["text"], reviews["label"])):
        clean = clean_review(text)
        corpus.reviews.append(clean)
        for chunk_id, chunk in enumerate(chunk_words(clean, size, overlap)):
            corpus.texts.append(chunk)
            corpus.review_ids.append(review_id)
            corpus.chunk_ids.append(chunk_id)
            corpus.labels.append(int(label))
    return corpus


# (n_chunks, dim) float32, cache przez part1.pipeline
def corpus_embeddings(
    model: SentenceTransformer,
    corpus: Corpus,
    *,
    device: Backend,
    n_reviews: int = N_REVIEWS,
    size: int = CHUNK_WORDS,
    overlap: int = CHUNK_OVERLAP,
) -> torch.Tensor:
    cache = CACHE_DIR / f"imdb_{DATASET_SPLIT}_{n_reviews}_w{size}_o{overlap}"
    ds = pipeline.load_or_build_embeddings(model, corpus.texts, cache, device=device)
    return pipeline.embeddings_tensor(ds)
