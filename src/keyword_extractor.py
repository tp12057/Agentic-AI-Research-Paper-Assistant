"""
Extracts keywords from text using KeyBERT.
"""

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from . import config


def load_keyword_model(embedding_model: SentenceTransformer) -> KeyBERT:
    """Load a KeyBERT model built on top of the shared embedding model."""
    return KeyBERT(embedding_model)


def extract_keywords(
    kw_model: KeyBERT,
    text: str,
    ngram_range=config.KEYWORD_NGRAM_RANGE,
    stop_words: str = config.KEYWORD_STOP_WORDS,
    top_n: int = config.KEYWORD_TOP_N,
):
    """Extract the top-n keywords/keyphrases from a piece of text."""
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=ngram_range,
        stop_words=stop_words,
        top_n=top_n,
    )
