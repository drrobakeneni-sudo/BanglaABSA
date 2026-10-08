# Local data preparation

Only `examples/` (authored synthetic text) and `templates/` are tracked. Put real data under `private/` outside the tracked tree. The original corpus is not available in this repository.

Canonical JSONL fields are `review_id` (nonempty opaque string), `text` (nonempty string), optional `labels` with Product/Shop/Transport values Negative/Neutral/Positive/null, and optional string `group_id`, `source_platform`, `date`, plus an optional provenance object. Unknown fields and duplicated IDs are rejected. Missing label keys are unobserved, never Neutral. Label-free prediction records are valid.

CSV requires `review_id,text,Product,Shop,Transport`; blank labels become null. The separate annotation template includes annotator/adjudication columns and must be exported to canonical form before training. Never feed uncertainty notes or source star ratings as sentiment predictors.

```bash
bangla-absa prepare --input data/examples/synthetic_reviews.jsonl --out runs/example-prepared
bangla-absa train --data runs/example-prepared/reviews.jsonl --manifest runs/example-prepared/splits.json --config configs/smoke/m4.json --out runs/example-m4
```

For original data, replace the input path and use `configs/models/`. To preserve authorized original split IDs, provide `--manifest private/original-splits.json`. No published split IDs are invented. Group related reviews using `group_id`; duplicates and linked groups cannot cross partitions. `--deduplicate` merges compatible exact duplicate annotations and fails on contradictory labels. Retaining duplicates with common partition membership is the default.

## Optional related dataset

Download the files only from the [Mendeley landing page](https://data.mendeley.com/datasets/n4n5y34p3s/1) using your permitted access. Record the version, filename, license, actual column names and SHA-256 locally. We could verify landing metadata but could not retrieve files (HTTP 403), so no upstream schema is assumed.

Convert the downloaded annotations to one row per review/aspect with explicit ID/text/aspect/sentiment columns and call `data.public_adapter.adapt_long_rows`. Its named column parameters let you map actual fields. It excludes price/packaging, maps product quality/seller service/delivery, and counts conflicts; select `conflict="missing"` explicitly or adjudicate them first. Preserve the dataset citation/license and do not label this as the manuscript's original 200k corpus.

## fastText resources

The static-model configs expect `private/cc.bn.300.vec`, obtainable through the [official fastText vector listing](https://fasttext.cc/docs/en/crawl-vectors.html). Extract the 300-dimensional text vectors locally and preserve their CC BY-SA 3.0 terms. The streaming loader only retains train-vocabulary entries; OOV vectors are random trainable entries and the file checksum/coverage count are recorded. No automatic multi-gigabyte download occurs.
