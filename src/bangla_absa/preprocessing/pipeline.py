from bangla_absa.io import fingerprint, read_json, write_json
from bangla_absa.preprocessing.cues import CueLexicon
from bangla_absa.preprocessing.normalize import eligible, normalize
from bangla_absa.preprocessing.transliterate import Transliterator


class Pipeline:
    version = "reconstruction-v1"

    def __init__(self, transliterator=None, lexicon=None):
        self.transliterator = transliterator or Transliterator()
        self.lexicon = lexicon or CueLexicon()

    def transform(self, records, min_words=1, max_words=100000, language="any"):
        result, excluded, unknown = [], 0, 0
        for row in records:
            text, n = self.transliterator.transform(normalize(row["text"]))
            unknown += n
            if not eligible(text, min_words, max_words, language):
                excluded += 1
                continue
            result.append({**row, "text": text})
        return result, {
            "input": len(records),
            "retained": len(result),
            "excluded": excluded,
            "unknown_latin_tokens": unknown,
            "version": self.version,
        }

    def fit(self, records, role="train"):
        self.lexicon.fit(records, role)
        return self

    def state(self):
        return {
            "version": self.version,
            "transliterator": self.transliterator.state(),
            "cues": self.lexicon.mapping,
            "min_count": self.lexicon.min_count,
        }

    def save(self, path):
        state = self.state()
        write_json(path, {"state": state, "checksum": fingerprint(state)})

    @classmethod
    def load(cls, path):
        data = read_json(path)
        state = data["state"]
        if data["checksum"] != fingerprint(state) or state["version"] != cls.version:
            raise ValueError("Preprocessing checksum or version mismatch")
        return cls(
            Transliterator(**state["transliterator"]), CueLexicon(state["cues"], state["min_count"])
        )
