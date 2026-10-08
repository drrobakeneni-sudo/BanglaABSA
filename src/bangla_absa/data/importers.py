import csv
import json
from pathlib import Path

from bangla_absa import ASPECTS
from bangla_absa.data.schema import validate_records


def read_records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    elif path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            rows = []
            for record in csv.DictReader(f):
                labels = {a: record.pop(a, None) or None for a in ASPECTS}
                record["labels"] = labels
                rows.append(record)
    else:
        raise ValueError("Use canonical .jsonl, .json, or UTF-8 .csv")
    return validate_records(rows)


def write_records(path, rows):
    validate_records(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
