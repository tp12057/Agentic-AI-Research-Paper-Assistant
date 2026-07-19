"""
Shared configuration and constants for the Research Paper Intelligence System.
"""

# Dataset
DATASET_NAME = "CShorten/ML-ArXiv-Papers"
DATASET_SPLIT = "train"
DATASET_ROW_LIMIT = 15000

# Embedding model
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# File paths for cached artifacts
EMBEDDINGS_PATH = "data/paper_embeddings.npy"
FAISS_INDEX_PATH = "data/paper_faiss.index"

# Summarization model
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
SUMMARY_MAX_LENGTH = 120
SUMMARY_MIN_LENGTH = 40

# Keyword extraction
KEYWORD_NGRAM_RANGE = (1, 2)
KEYWORD_STOP_WORDS = "english"
KEYWORD_TOP_N = 5

# LLM (Groq)
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
GROQ_TEMPERATURE = 0
