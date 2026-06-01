import re


def strip_noise(text: str) -> str:
    """
    Minimally clean the text by removing:
    - URLs
    - DOIs
    - Citation brackets like [1], [1, 2], etc.
    """
    # Remove URLs
    text = re.sub(r"http[s]?://\S+", "", text)
    # Remove DOIs
    text = re.sub(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", "", text, flags=re.IGNORECASE)
    # Remove citation brackets
    text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)

    # Clean up extra spaces left behind
    text = re.sub(r"\s+", " ", text).strip()
    return text
