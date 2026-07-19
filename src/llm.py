"""
Sets up the LLM (Groq-hosted Llama model) used by the research agent.
"""

import os

from langchain_groq import ChatGroq

from . import config


def load_llm(api_key: str = None) -> ChatGroq:
    """
    Load the Groq chat model. Reads the API key from the GROQ_API_KEY
    environment variable if not provided explicitly.
    """
    api_key = api_key or os.environ.get("GROQ_API_KEY")

    return ChatGroq(
        model=config.GROQ_MODEL_NAME,
        api_key=api_key,
        temperature=config.GROQ_TEMPERATURE,
    )
