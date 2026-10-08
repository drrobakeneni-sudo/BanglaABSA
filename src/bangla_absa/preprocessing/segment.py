import re

from bangla_absa import ASPECTS

DEMO_CUES = {
    "Product": ["পণ্য", "মান", "product", "quality"],
    "Shop": ["দোকান", "বিক্রেতা", "seller", "shop"],
    "Transport": ["ডেলিভারি", "পরিবহন", "delivery", "transport"],
}


def segment(text, cues=None):
    """Rules only. An unaligned clause remains available as global context."""
    cues = cues or DEMO_CUES
    output = []
    for match in re.finditer(r"[^।.!?;]+[।.!?;]?", text):
        part = match.group()
        for clause in re.finditer(
            r".+?(?=\s+(?:কিন্তু|তবে|but|however)\s+|$)", part, flags=re.IGNORECASE
        ):
            start, end = match.start() + clause.start(), match.start() + clause.end()
            fragment = text[start:end]
            aligned = [a for a in ASPECTS if any(c.lower() in fragment.lower() for c in cues[a])]
            output.append({"start": start, "end": end, "aspects": aligned})
    return output


def token_alignment(text, offsets, cues=None):
    clauses = segment(text, cues)
    return [
        [
            any(start < c["end"] and end > c["start"] and a in c["aspects"] for c in clauses)
            if end > start
            else False
            for start, end in offsets
        ]
        for a in ASPECTS
    ]
