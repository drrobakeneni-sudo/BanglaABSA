import json

import krippendorff
import numpy as np
import pytest

from bangla_absa.config import ModelConfig, TrainingConfig, load_config, tiny_config
from bangla_absa.data.annotation import nominal_agreement
from bangla_absa.data.importers import read_records, write_records
from bangla_absa.data.public_adapter import adapt_long_rows
from bangla_absa.data.schema import validate_records
from bangla_absa.data.splits import deduplicate, make_splits, validate_splits
from bangla_absa.preprocessing.cues import CueLexicon
from bangla_absa.preprocessing.normalize import eligible, normalize, redact_contacts
from bangla_absa.preprocessing.pipeline import Pipeline
from bangla_absa.preprocessing.segment import segment
from bangla_absa.preprocessing.tokenization import Features
from bangla_absa.preprocessing.transliterate import Transliterator


def test_normalization_and_conservative_transliteration():
    assert normalize(" পণ্য\tভালো  না। 🙂 ") == "পণ্য ভালো না। 🙂"
    t = Transliterator()
    converted, unknown = t.transform("onek bhalo but delivery late")
    assert converted == "অনেক ভালো but delivery late" and unknown == 3
    rule = Transliterator(mapping={}, rules=[["bha", "ভা"], ["lo", "লো"]], candidates=["bhalo"])
    assert rule.transform("bhalo bank")[0] == "ভালো bank"
    assert eligible("পণ্য ভালো না", 3, 3, "bangla")
    assert not eligible("hello there", 2, 3, "bangla")
    with pytest.raises(ValueError):
        eligible("abc", language="invented")
    assert "@" not in redact_contacts("test@example.com +880 1712 345678 https://example.com")


def test_segmentation_and_training_only_fit(records, tmp_path):
    assert {a for s in segment("পণ্য ভালো। দোকান খারাপ কিন্তু ডেলিভারি হয়েছে!") for a in s["aspects"]} == {
        "Product",
        "Shop",
        "Transport",
    }
    pipeline = Pipeline().fit(records)
    cfg = tiny_config()
    features = Features(cfg, pipeline).fit(records)
    assert "শুধুপরীক্ষা" not in features.vocabulary
    features.encode([{"review_id": "new", "text": "শুধুপরীক্ষা"}])
    assert "শুধুপরীক্ষা" not in features.vocabulary
    with pytest.raises(ValueError):
        features.fit(records, "test")
    with pytest.raises(ValueError):
        CueLexicon().fit(records, "validation")
    pipeline.save(tmp_path / "pipeline.json")
    assert Pipeline.load(tmp_path / "pipeline.json").state() == pipeline.state()
    features.save(tmp_path)
    loaded = Features.load(tmp_path, cfg, pipeline)
    assert loaded.vocabulary == features.vocabulary
    data = json.loads((tmp_path / "pipeline.json").read_text(encoding="utf-8"))
    data["checksum"] = "broken"
    (tmp_path / "pipeline.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        Pipeline.load(tmp_path / "pipeline.json")


@pytest.mark.parametrize(
    "bad",
    [
        {"review_id": "x", "text": ""},
        {"review_id": "x", "text": "a", "labels": {"Price": "Neutral"}},
        {"review_id": "x", "text": "a", "group_id": 4},
        {"review_id": "x", "text": "a", "provenance": "no"},
    ],
)
def test_invalid_schema(bad):
    with pytest.raises(ValueError):
        validate_records([bad])


def test_local_import_roundtrip(records, tmp_path):
    path = tmp_path / "records.jsonl"
    write_records(path, records)
    assert read_records(path) == records
    (tmp_path / "r.csv").write_text(
        "review_id,text,Product,Shop,Transport\nx,পণ্য ভালো,Positive,,\n", encoding="utf-8"
    )
    assert read_records(tmp_path / "r.csv")[0]["labels"]["Shop"] is None
    with pytest.raises(ValueError):
        read_records(tmp_path / "bad.txt")


def test_splits_and_duplicates(records):
    with pytest.raises(ValueError):
        make_splits(records[:1])
    with pytest.raises(ValueError):
        make_splits(records, {"train": 0.3, "test": 0.3})
    with pytest.raises(ValueError):
        validate_splits(records, {})
    duplicate = {**records[0], "review_id": "dupe"}
    assert len(deduplicate([records[0], duplicate])) == 1
    duplicate["labels"] = {"Product": "Positive"}
    with pytest.raises(ValueError):
        deduplicate([records[0], duplicate])


def test_alpha_against_trusted_implementation():
    ratings = [[0, 0, 1], [1, None, 1], [2, 2, 2], [0, 1, 2], [None, 1, None]]
    array = np.array([[np.nan if v is None else v for v in row] for row in ratings]).T
    expected = krippendorff.alpha(array, level_of_measurement="nominal")
    result = nominal_agreement(ratings)
    assert result["alpha"] == pytest.approx(expected)
    assert result["raw_pair_agreement"] == pytest.approx(5 / 10)
    assert nominal_agreement([[None, 1]])["alpha"] is None
    assert nominal_agreement([[1, 1]])["alpha"] is None


def test_public_adapter_conflict_and_absence():
    rows = [
        {"review_id": "synthetic-1", "text": "সিনথেটিক", "aspect": a, "sentiment": s}
        for a, s in [
            ("product quality", "positive"),
            ("price", "negative"),
            ("product quality", "negative"),
        ]
    ]
    with pytest.raises(ValueError, match="Conflicting"):
        adapt_long_rows(rows)
    result, report = adapt_long_rows(rows, conflict="missing")
    assert all(v is None for v in result[0]["labels"].values())
    assert report == {"excluded_aspect_rows": 1, "conflicting_decisions": 1}


def test_config_validation(tmp_path):
    with pytest.raises(ValueError):
        ModelConfig(width=7, heads=2)
    with pytest.raises(ValueError):
        TrainingConfig(epochs=0)
    p = tmp_path / "config.json"
    p.write_text('{"mode":"exact", "model":{}}')
    with pytest.raises(ValueError, match="Exact reproduction"):
        load_config(p)


def test_deduplication_preserves_transitive_group_links():
    rows = [
        {"review_id": "a", "text": "duplicate", "group_id": "first"},
        {"review_id": "b", "text": "duplicate", "group_id": "second"},
        {"review_id": "c", "text": "different one", "group_id": "first"},
        {"review_id": "d", "text": "different two", "group_id": "second"},
    ]
    result = deduplicate(rows)
    assert len(result) == 3
    assert len({r["group_id"] for r in result}) == 1
