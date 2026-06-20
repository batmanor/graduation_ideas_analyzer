def build_text(title: str, abstract: str, keywords: str | None = None) -> str:
    return f"Title: {title}\nAbstract: {abstract}\nKeywords: {keywords or ''}"
