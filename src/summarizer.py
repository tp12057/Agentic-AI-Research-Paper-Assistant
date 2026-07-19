"""
Summarizes paper abstracts using a BART summarization pipeline.
"""

from transformers import pipeline

from . import config


def load_summarizer(model_name: str = config.SUMMARIZER_MODEL_NAME):
    """Load the Hugging Face summarization pipeline."""
    return pipeline("summarization", model=model_name)


def summarize_text(
    summarizer,
    text: str,
    max_length: int = config.SUMMARY_MAX_LENGTH,
    min_length: int = config.SUMMARY_MIN_LENGTH,
) -> str:
    """Summarize a single piece of text (e.g. an abstract)."""
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
    )
    return summary[0]["summary_text"]
