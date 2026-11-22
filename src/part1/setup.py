from datasets import Dataset, load_dataset
from sentence_transformers import SentenceTransformer,  CrossEncoder

from part1.config import (
    DATASET_ID,
    Backend,
    resolve_backend,
    BI_ENCODER_MODEL_ID,
    CROSS_ENCODER_MODEL_ID
)

def get_model(model_id: str, device: Backend | None = None) -> SentenceTransformer | CrossEncoder:
    """Load the bi-encoder / cross-encoder."""
    if model_id == BI_ENCODER_MODEL_ID:
        return SentenceTransformer(model_id, device=resolve_backend(device))
    elif model_id == CROSS_ENCODER_MODEL_ID:
        return CrossEncoder(model_id, device=resolve_backend(device))
    else:
        raise ValueError(f"Unknown model_id: {model_id}")


def get_dataset() -> Dataset:
    """Load the SQuAD pair dataset."""
    return load_dataset(DATASET_ID, split="train")

def get_cross_encoder(
    device: Backend | None = None
) -> CrossEncoder:
    """Load the cross-encoder used for reranking."""
    return CrossEncoder(
        CROSS_ENCODER_MODEL_ID,
        device=resolve_backend(device)
    )


if __name__ == "__main__":
    # Autodetect backend & download models
    bi_encoder = get_model(BI_ENCODER_MODEL_ID)
    cross_encoder = get_model(CROSS_ENCODER_MODEL_ID)

    print(f"bi-encoder: {BI_ENCODER_MODEL_ID} on '{bi_encoder.device}'")
    print(f"cross-encoder: {CROSS_ENCODER_MODEL_ID} on '{cross_encoder.device}'")

    ds = get_dataset()
    print(f"dataset: {DATASET_ID} (train) -> shape {ds.shape}")
    print(f"first row: {ds[0]}")

