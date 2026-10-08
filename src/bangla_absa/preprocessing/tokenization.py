import re
from pathlib import Path

import torch

from bangla_absa.io import read_json, write_json
from bangla_absa.preprocessing.segment import token_alignment

TARGETS = ("পণ্য মান", "দোকান সেবা", "পরিবহন ডেলিভারি")
QUESTIONS = tuple(f"{target} সম্পর্কে অনুভূতি কেমন?" for target in TARGETS)
TARGET_MODELS = {"memnet", "atae_lstm", "ian", "ram", "aen"}
PAIR_MODELS = {"spc", "pair_qa_m", "lcf_bert"}


def word_spans(text):
    return list(re.finditer(r"[\w\u0980-\u09ff]+|[^\w\s]", text))


class Features:
    def __init__(self, cfg, pipeline, vocabulary=None, tokenizer=None):
        self.cfg, self.pipeline = cfg, pipeline
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer

    def fit(self, records, role="train"):
        if role != "train":
            raise ValueError("Tokenizer vocabulary may only fit on train")
        if self.cfg.encoder == "hf":
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.checkpoint, revision=self.cfg.revision, use_fast=True
            )
            if not self.tokenizer.is_fast:
                raise ValueError("A fast tokenizer with offsets is required for cue alignment")
        else:
            from collections import Counter

            counts = Counter(m.group() for r in records for m in word_spans(r["text"]))
            vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
            for text in (*TARGETS, *QUESTIONS):
                for m in word_spans(text):
                    if m.group() not in vocabulary:
                        vocabulary[m.group()] = len(vocabulary)
            for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
                if token not in vocabulary and len(vocabulary) < self.cfg.vocab_size:
                    vocabulary[token] = len(vocabulary)
            if len(vocabulary) > self.cfg.vocab_size:
                raise ValueError("vocab_size too small for reserved category phrases")
            self.vocabulary = vocabulary
        return self

    def _one(self, text, pair=None):
        maximum = self.cfg.max_length
        if self.tokenizer is not None:
            encoded = self.tokenizer(
                text,
                pair,
                max_length=maximum,
                truncation="only_first" if pair else True,
                padding="max_length",
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            sequence = encoded.sequence_ids()
            offsets = [offset if sequence[i] == 0 else (0, 0) for i, offset in enumerate(offsets)]
            values = dict(encoded)
        else:
            if self.vocabulary is None:
                raise ValueError("Features must be fitted or loaded")
            first, second = word_spans(text), word_spans(pair) if pair else []
            first = first[: maximum - len(second) - (3 if pair else 2)]
            if len(second) > maximum - 4:
                raise ValueError("Sequence too short for category prompt")
            ids = [2] + [self.vocabulary.get(m.group(), 1) for m in first] + [3]
            offsets = [(0, 0)] + [m.span() for m in first] + [(0, 0)]
            types = [0] * len(ids)
            if pair:
                ids += [self.vocabulary.get(m.group(), 1) for m in second] + [3]
                types += [1] * (len(second) + 1)
                offsets += [(0, 0)] * (len(second) + 1)
            padding = maximum - len(ids)
            values = {
                "input_ids": ids + [0] * padding,
                "attention_mask": [1] * len(ids) + [0] * padding,
                "token_type_ids": types + [0] * padding,
            }
            offsets += [(0, 0)] * padding
        values["cue_ids"] = self.pipeline.lexicon.encode(text, offsets)
        return values, offsets

    def encode(self, records):
        rows = []
        for row in records:
            values, offsets = self._one(row["text"])
            values["alignment"] = token_alignment(row["text"], offsets)
            for prefix, required in (("target", TARGET_MODELS), ("pair", PAIR_MODELS)):
                if self.cfg.name not in required:
                    continue
                parts = []
                for a in range(3):
                    if prefix == "target":
                        part, _ = self._one(TARGETS[a])
                    else:
                        prompt = QUESTIONS[a] if self.cfg.name == "pair_qa_m" else TARGETS[a]
                        part, _ = self._one(row["text"], prompt)
                    parts.append(part)
                for key in parts[0]:
                    values[f"{prefix}_{key}"] = [part[key] for part in parts]
            rows.append(values)
        return {key: torch.tensor([r[key] for r in rows], dtype=torch.long) for key in rows[0]}

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(directory / "tokenizer")
        write_json(
            directory / "features.json",
            {
                "vocabulary": self.vocabulary,
                "checkpoint": self.cfg.checkpoint,
                "revision": self.cfg.revision,
                "kind": self.cfg.encoder,
                "max_length": self.cfg.max_length,
            },
        )

    @classmethod
    def load(cls, directory, cfg, pipeline):
        directory = Path(directory)
        state = read_json(directory / "features.json")
        if state["revision"] != cfg.revision or state["max_length"] != cfg.max_length:
            raise ValueError("Tokenizer configuration mismatch")
        tokenizer = None
        if state["kind"] == "hf":
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                directory / "tokenizer", local_files_only=True
            )
        return cls(cfg, pipeline, state["vocabulary"], tokenizer)
