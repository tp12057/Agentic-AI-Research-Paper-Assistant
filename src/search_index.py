"""
Builds and queries a FAISS similarity search index over paper embeddings.
"""

import os

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from . import config


def build_or_load_index(
    embeddings: np.ndarray,
    index_path: str = config.FAISS_INDEX_PATH,
    embedding_dim: int = config.EMBEDDING_DIM,
) -> faiss.Index:
    """
    Build a FAISS FlatIP (cosine similarity via normalized inner product)
    index, or load one from disk if it already exists.
    """
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_path)
    else:
        print("Creating new FAISS index...")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings)

        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index, index_path)
        print("FAISS index saved successfully!")

    return index


def search_papers(
    query: str,
    model: SentenceTransformer,
    index: faiss.Index,
    k: int = 5,
):
    """Search the FAISS index for the top-k papers matching a query."""
    query_embedding = model.encode([query])
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)
    return distances, indices


def print_search_results(query: str, model: SentenceTransformer, index: faiss.Index, df: pd.DataFrame, k: int = 5):
    """Search and pretty-print results (title + abstract preview)."""
    distances, indices = search_papers(query, model, index, k)

    for score, idx in zip(distances[0], indices[0]):
        print("=" * 100)
        print("Similarity:", round(float(score), 4))
        print("Title:", df.iloc[idx]["title"])
        print()
        print("Abstract:")
        print(df.iloc[idx]["abstract"][:500])
        print()
