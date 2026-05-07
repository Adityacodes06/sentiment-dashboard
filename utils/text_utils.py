"""
Text Preprocessing Utilities
Cleans and prepares text input for sentiment analysis.
"""

import re


def clean_text(text: str) -> str:
    """Basic text cleaning — removes URLs, extra spaces, special chars."""
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)                  # Remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)             # Remove hashtag symbol
    text = re.sub(r"\s+", " ", text).strip()          # Collapse whitespace
    return text


def split_into_sentences(text: str) -> list:
    """Split a paragraph into individual sentences for batch analysis."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 5]


def parse_reviews(raw: str) -> list:
    """
    Parse newline-separated reviews from a text input.
    Each line is treated as one review.
    """
    lines = raw.strip().split("\n")
    return [clean_text(line) for line in lines if line.strip()]
