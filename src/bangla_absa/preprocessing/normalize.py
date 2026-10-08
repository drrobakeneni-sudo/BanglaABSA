import re
import unicodedata


def normalize(text):
    # NFC preserves Bangla signs and negation; do not remove emoji, ZWJ, or punctuation.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()


def tokens(text):
    # Retain combining marks in words and emoji as individual tokens.
    return re.findall(r"[\w\u0980-\u09ff]+|[^\w\s]", normalize(text), re.UNICODE)


def eligible(text, min_words=5, max_words=256, language="any"):
    if language not in {"any", "bangla_or_latin", "bangla"}:
        raise ValueError("Unknown language filter")
    n = len(normalize(text).split())
    script = bool(re.search(r"[\u0980-\u09ff]", text))
    if language == "bangla_or_latin":
        script = script or bool(re.search(r"[A-Za-z]", text))
    return min_words <= n <= max_words and (language == "any" or script)


def redact_contacts(text):
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", text)
    text = re.sub(r"https?://\S+", "[URL]", text)
    return re.sub(r"(?<!\w)\+?\d[\d ()-]{7,}\d", "[PHONE]", text)
