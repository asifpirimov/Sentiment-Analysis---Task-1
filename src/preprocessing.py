import re

whitespace = re.compile(r"\s+")
numbers = re.compile(r"\d+")
punct = re.compile(r"[^\w\s]")


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = numbers.sub("", text)
    text = punct.sub(" ", text)
    text = whitespace.sub(" ", text).strip()
    return text