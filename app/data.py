"""Loads the parquet dataset into memory and exposes typed accessors."""

import os
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import pandas as pd

DEFAULT_PARQUET_FILENAME = "whole_process_contributions_with_comments_250_deepsseek-v4-pro_deepsseek-v4-pro.parquet"
print("ENV: " + os.environ.get("PARQUET_PATH"))
PARQUET_PATH = Path(
    os.environ.get("PARQUET_PATH")
    or Path(__file__).parent.parent / DEFAULT_PARQUET_FILENAME
)

# Filenames follow "whole_process_contributions_with_comments_250[_<model>].parquet".
# The model suffix is sometimes duplicated (e.g. "..._deepseek-v4-pro_deepseek-v4-pro.parquet").
BASE_DATASET_STEM = "whole_process_contributions_with_comments_250"


def get_model_name() -> str:
    """Derive the evaluated model's name from the parquet filename, or "default" if none is present."""
    remainder = PARQUET_PATH.stem
    if remainder.startswith(BASE_DATASET_STEM):
        remainder = remainder[len(BASE_DATASET_STEM):]
    remainder = remainder.strip("_")
    if not remainder:
        return "default"
    deduped_tokens = [token for token, _ in groupby(remainder.split("_"))]
    return "_".join(deduped_tokens)


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    df = pd.read_parquet(PARQUET_PATH)
    # Normalise column types
    df["contribution_id"] = df["contribution_id"].astype(int)
    return df


def get_contributions() -> list[dict]:
    """Return one record per contribution with its full text."""
    df = get_dataframe()
    return (
        df[["contribution_id", "contributionContent"]]
        .drop_duplicates("contribution_id")
        .sort_values("contribution_id")
        .rename(columns={"contributionContent": "contribution_content"})
        .to_dict(orient="records")
    )


def get_key_messages(contribution_id: int) -> list[dict]:
    """Return all predicted key messages for a contribution."""
    df = get_dataframe()
    subset = df[df["contribution_id"] == contribution_id]
    return (
        subset[
            [
                "contribution_id",
                "contributionContent",
                "key_message",
                "key_message_type",
                "related_sentences",
                "key_message_explanation",
            ]
        ]
        .drop_duplicates("key_message")
        .rename(columns={"contributionContent": "contribution_content"})
        .to_dict(orient="records")
    )


def get_stances(contribution_id: int, key_message: str) -> list[dict]:
    """Return all predicted stances for a (contribution, key_message) pair."""
    df = get_dataframe()
    subset = df[
        (df["contribution_id"] == contribution_id)
        & (df["key_message"] == key_message)
    ]
    return (
        subset[
            [
                "contribution_id",
                "key_message",
                "comment_text",
                "stance",
                "explanation",
            ]
        ]
        .to_dict(orient="records")
    )
