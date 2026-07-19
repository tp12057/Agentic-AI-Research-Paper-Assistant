"""
Named Entity Recognition pipeline for extracting entities from text
(e.g. model names, datasets, institutions).
"""

from transformers import pipeline


def load_ner_pipeline():
    """Load the Hugging Face NER pipeline with simple entity aggregation."""
    return pipeline("ner", aggregation_strategy="simple")


def extract_entities(ner_pipeline, text: str):
    """Run NER on a piece of text and return the detected entities."""
    return ner_pipeline(text)
