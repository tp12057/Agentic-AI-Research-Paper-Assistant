"""
Loads the ML-ArXiv-Papers dataset and prepares it as a Pandas DataFrame.
"""

import pandas as pd
from datasets import load_dataset

from . import config


def load_papers_dataframe(row_limit: int = config.DATASET_ROW_LIMIT) -> pd.DataFrame:
    """
    Load the research paper dataset and return a cleaned DataFrame with a
    single 'paper_text' column (title + abstract combined).
    """
    dataset = load_dataset(config.DATASET_NAME, split=config.DATASET_SPLIT)

    df = pd.DataFrame(dataset)
    df = df[["title", "abstract"]]
    df = df.head(row_limit)

    df["paper_text"] = df["title"] + " " + df["abstract"]
    df["paper_text"] = df["paper_text"].str.replace("\n", " ", regex=False)
    df["paper_text"] = df["paper_text"].str.strip()

    return df
