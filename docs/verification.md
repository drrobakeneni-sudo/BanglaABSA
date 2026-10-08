# Release verification

Local verification date: **2026-10-08**. This is software verification on synthetic fixtures, not reproduction of manuscript performance.

| Check | Executed outcome |
|---|---|
| Editable package install/import | Passed: `python -m pip install -e . --no-deps` |
| Lint | Passed: `python -m ruff check src tests scripts` |
| Format | Passed: `python -m ruff format --check src tests scripts` |
| Offline test suite | **76 passed**, 5 upstream deprecation warnings, 50.22 seconds |
| Executable package coverage | **93.97%**: 1,932 of 2,056 statements; no core modules excluded |
| CPU integration | M4 and held-out M6 smoke flow completed, including checkpoint reload, validation/test, and label-free prediction |
| Comparator coverage | Every registered individual model family runs a finite forward/predict check; all five classical estimators round-trip; frozen ensemble and fusion controls tested |
| Orchestration | CLI prepare/train/validate/test/predict/evaluate/benchmark/suite/ablations/repeat/CV/smoke; five-seed voting; M6 nested outer-fold fitting tested on synthetic records |
| Real backbone classes offline | Small random BERT, ELECTRA, XLM-R, and DeBERTa-v2 configurations tested without downloads |
| Annotation agreement | Nominal alpha checked against krippendorff 0.8.2 including missing ratings |
| Leakage checks | Duplicate/group union, transitive links after deduplication, held-out fusion partitions, original TEST exclusion from CV, and stale/misaligned cache rejection tested |

Exact final test invocation (from the repository in the prepared virtual environment):

```bash
# HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 were set in the environment.
python -m pytest --cov=bangla_absa --cov-report=term --cov-report=json:../../work/coverage.json --cov-fail-under=80 -q --tb=short
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
```

The initial targeted contract suite was run before implementation and failed because the package did not exist. Later model tests caught unsafe native forest deserialization and calibration allowlist issues; forest checkpoints now use validated portable JSON arrays. A regression test protects transitive grouping when duplicates are consolidated.

## Environment and limitations

Local runtime: Windows, Python **3.10.11**, PyTorch **2.13.0+cpu**, Transformers **4.43.4**, NumPy **1.26.4**, scikit-learn **1.7.2**, SciPy **1.15.3**, XGBoost **3.2.0**. Additional direct pins are recorded in `requirements-lock.txt`. That file pins tested direct dependencies; it is not a hashed transitive lock. The local environment used a virtual environment with access to preinstalled packages, and installed pytest-cov, krippendorff, and skops into the virtual environment. GitHub CI separately installs the listed dependencies on Ubuntu/Python 3.11.

This differs from the paper's Ubuntu 22.04/Python 3.11.6/PyTorch 2.2.1/CUDA 12.1/Transformers 4.40.0/A100 environment. CUDA/mixed-precision execution and full pretrained fine-tuning have not been tested here. Remaining coverage includes those hardware/network-dependent paths and some defensive error branches. Warnings originate from SentencePiece SWIG types and deprecated upstream TorchScript use; they were not suppressed or treated as project test successes.

Full reconstructed parameter counts were computed by instantiating verified checkpoint configurations on PyTorch's meta device: M4 **132,420,937**, M5 **146,173,257**, stacker **462,729**, M6 total **279,056,923**. This is parameter counting, not training or a memory/latency measurement. See [parameter evidence](parameter_counts.json).

## Re-run

```bash
bangla-absa smoke --out runs/fresh-smoke
python -m pytest --cov=bangla_absa --cov-fail-under=80 -q
```

Use a fresh output directory; existing run directories are not silently overwritten. The [GitHub Actions page](https://github.com/drrobakeneni-sudo/BanglaABSA/actions) reports remote CI independently of this local record. No original-corpus training, real external validation, paper-level accuracy confirmation, or full-model FLOP measurement is claimed.
