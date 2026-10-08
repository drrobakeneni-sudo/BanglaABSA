import re
from collections import Counter

# Newly authored demonstration resources; not the manuscript authors' dictionary.
DEMO = {
    "bhalo": "ভালো",
    "valo": "ভালো",
    "kharap": "খারাপ",
    "onek": "অনেক",
    "ponno": "পণ্য",
    "dokan": "দোকান",
    "deri": "দেরি",
    "na": "না",
}


class Transliterator:
    def __init__(self, mapping=None, rules=None, candidates=None):
        self.mapping = DEMO.copy() if mapping is None else dict(mapping)
        self.rules = rules or []  # ordered literal phoneme→grapheme pairs
        self.candidates = set(candidates or [])  # explicitly known Banglish words only

    def transform(self, text):
        unknown = Counter()

        def replace(match):
            word = match.group()
            key = word.lower()
            if key in self.mapping:
                return self.mapping[key]
            if key in self.candidates:
                converted = key
                for source, target in self.rules:
                    converted = converted.replace(source, target)
                if not re.search("[a-z]", converted):
                    return converted
            unknown[key] += 1
            return word

        result = re.sub(r"\b[A-Za-z]+\b", replace, text)
        return result, sum(unknown.values())

    def state(self):
        return {"mapping": self.mapping, "rules": self.rules, "candidates": sorted(self.candidates)}
