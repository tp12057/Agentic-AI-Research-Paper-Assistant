"""
Generates and caches sentence embeddings for research papers using
Sentence-Transformers.
"""

import os

import numpy as np
from sentence_transformers import SentenceTransformer

from . import config


def load_embedding_model(model_name: str = config.EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load the Sentence-Transformers embedding model."""
    return SentenceTransformer(model_name)


def generate_embeddings(
    model: SentenceTransformer,
    texts: list,
    cache_path: str = config.EMBEDDINGS_PATH,
) -> np.ndarray:
    """
    Generate embeddings for a list of texts, using a cached .npy file if
    it already exists.
    """
    if os.path.exists(cache_path):
        print("Loading saved embeddings...")
        embeddings = np.load(cache_path)
    else:
        print("Generating embeddings...")
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, embeddings)
        print("Embeddings saved successfully!")

    return embeddings
