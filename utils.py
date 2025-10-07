import re

def simple_sentence_split(text: str):
    """Naively split sentences using punctuation."""
    parts = re.split(r'(?<=[.!?;])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]
