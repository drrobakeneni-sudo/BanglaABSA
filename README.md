# BanglaABSA

Python research software for **Product, Shop, and Transport sentiment classification** in Bangla, Banglish, and mixed-language reviews. Each aspect has its own Negative / Neutral / Positive distribution. Unobserved aspects remain missing.

This is a **new implementation reconstructed from the supplied manuscript**, *Aspect Aware Ensemble Learning for Multi Aspect Sentiment Classification in E Commerce Reviews*. It is not recovered author code, a release of the original 200,000 reviews, or a claim that the published results have been reproduced. See [reproduction status](docs/reproduction_status.md) and the [paper-to-code map](docs/paper_to_code.md).

## What is included

- M1 logistic regression, M2 XGBoost, M3 Random Forest; M4 Bangla-ABSA-Net; M5 TransNet; M6 weighted probability stacking.
- All Table 13 comparator families: TF-IDF NB/SVM, fastText BiLSTM, MemNet, ATAE-LSTM, IAN, RAM, seven pretrained checkpoint variants, three-head BanglaBERT, SPC, QA-M, AEN, LCF, and three ensemble controls. Adaptations are stated in [baseline provenance](docs/baseline_provenance.md).
- Grouped split generation, conservative transliteration, clause alignment, train-fitted cue embeddings, annotation agreement, checkpoint/resume, held-out stacking, repeated seeds, development-only CV, and Tables 10–12 ablations.
- Independent metrics and denominators, prediction alignment, cluster bootstrap, paired tests, correction for multiple comparisons, ROC/PR/confusion plots, and measured inference benchmarks.
- Authored synthetic examples, annotation templates, tests, and an offline CPU CI workflow.

## Install

Use Python 3.10 or newer in a virtual environment. The reference dependency versions are in [requirements-lock.txt](requirements-lock.txt); hardware and historical-environment differences are documented in [verification](docs/verification.md).

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
```

Installation does not download language-model weights or datasets. Full experiments download only the explicitly selected Hugging Face checkpoint, at its recorded immutable revision. fastText models require a local `.vec` file.

## Offline CPU quickstart

```bash
bangla-absa smoke --out runs/smoke
python -m pytest --cov=bangla_absa --cov-report=term-missing
ruff check src tests scripts
```

The smoke command creates synthetic examples, prepares deterministic partitions, trains tiny M4/M5 branches, trains M6 on held-out predictions, reloads checkpoints, predicts without labels, and evaluates validation/test subsets. Inspect `runs/smoke/smoke-report.json`. **These toy metrics have no scientific performance interpretation.** Outputs under `runs/` are ignored by Git.

## From local data to predictions

Start with UTF-8 JSONL records:

```json
{"review_id":"local-opaque-id","text":"পণ্য ভালো কিন্তু ডেলিভারি দেরি।","labels":{"Product":"Positive","Shop":null,"Transport":"Negative"}}
```

The example text is synthetic. CSV and JSON adapters are also available. Read [data preparation](data/README.md), [dataset card](docs/dataset_card.md), and [annotation guidelines](docs/annotation_guidelines.md) before importing data.

```bash
bangla-absa prepare --input private/reviews.jsonl --out private/prepared
bangla-absa train --data private/prepared/reviews.jsonl --manifest private/prepared/splits.json --config configs/models/m4.json --out runs/m4
bangla-absa validate --data private/prepared/reviews.jsonl --run runs/m4 --out runs/m4-validation
bangla-absa test --data private/prepared/reviews.jsonl --run runs/m4 --out runs/m4-test
bangla-absa predict --data private/new-unlabeled.jsonl --run runs/m4 --out runs/new-predictions.json
bangla-absa evaluate --predictions runs/m4-test/predictions.json --out runs/m4-figures --plots
```

`predict` needs only `review_id` and `text`. `validate` and `test` use the saved manifest and reject changed input data. If original split IDs become available, use `prepare --manifest private/original-splits.json`; data changes that invalidate the manifest fail explicitly. Exact reproduction mode currently fails because essential original artifacts are unavailable.

Full-model configs preserve the reported learning rates, batch sizes, epoch limits, and dropout where specified. Unspecified choices are marked reconstructed. Edit `training.device` to `cuda` only when that device is available; enable `mixed_precision` explicitly if wanted. See [architecture](docs/architecture.md), [preprocessing](docs/preprocessing.md), and [experiments](docs/experiments.md).

## Ensemble, baselines, ablations, seeds, and CV

```bash
bangla-absa ensemble --data private/prepared/reviews.jsonl --manifest private/prepared/splits.json --config configs/models/m6.json --out runs/m6
python scripts/run_baselines.py --data private/prepared/reviews.jsonl --manifest private/prepared/splits.json --configs configs/models --out runs/baselines
python scripts/run_ablations.py --data private/prepared/reviews.jsonl --manifest private/prepared/splits.json --out runs/ablations
python scripts/run_seeds.py --data private/prepared/reviews.jsonl --manifest private/prepared/splits.json --config configs/models/m5.json --out runs/seeds
python scripts/run_cv.py --data private/prepared/reviews.jsonl --manifest private/prepared/splits.json --config configs/models/m6.json --out runs/cv
python scripts/export_tables.py --predictions runs/m4-test/predictions.json --out runs/tables.csv
```

These commands explicitly request training and can be expensive. They are not run during installation. M6 creates its own M4/M5 runs: checkpoint selection uses the designated meta-validation subset, and fusion fitting uses separate held-out subsets. Five-fold CV refits the entire pipeline inside each development fold and never consumes the original test split.

## Evidence and licensing

[Verification](docs/verification.md) records executed commands, test coverage, CPU smoke outcomes, and limitations. No full-corpus training, external validation, or reproduction of manuscript metrics has been performed. No original model checkpoints are included.

The repository owner has not selected a source-code license; publication does not grant an additional open-source license. Third-party models and libraries retain their own terms, including the BanglaBERT repository's noncommercial research restriction. See [third-party notices](THIRD_PARTY_NOTICES.md). The original manuscript and restricted review text are not redistributed here.
