import re

_URL_RE = re.compile(r"https?://\S+")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/\S+\b", re.IGNORECASE)
_CITATION_RE = re.compile(r"\[[\d,\s-]+\]")


def strip_noise(text: str) -> str:
    text = _URL_RE.sub("", text)
    text = _DOI_RE.sub("", text)
    text = _CITATION_RE.sub("", text)
    return " ".join(text.split())


def build_text(title: str, abstract: str, keywords: str | None = None) -> str:
    return f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords or ''}"
