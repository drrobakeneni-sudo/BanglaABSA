# Experiment workflows and evidence

## Configuration

`configs/models/` covers all Table 13 methods. `configs/ablations/` covers full M4/M5/M6 controls and every removal in Tables 10–12. `configs/smoke/` has tiny offline M1/M4/M5/M6 configurations. JSON is validated against typed configuration fields; unknown keys fail. Fixed aspect and sentiment order is never inferred from sklearn class order.

Five proposed seeds are 17, 29, 43, 71, 101. They are **new**, not recovered original seeds. M4/M5/M6 documented optimizer settings are retained; other epoch limits, dimensions, warmup, clipping, monitor, and learning-rate schedule are reconstructed. The schedule is linear warmup/decay; gradient clipping uses norm 1; validation mean aspect macro F1 selects checkpoints. AEN uses .2 label smoothing. Checkpoint metadata records package versions, OS/Python, CPU/GPU, git commit when available, data/split hashes, IDs consumed by fitting/selection, and complete configuration.

`train --resume` restores last neural model parameters, AdamW, scheduler, gradient scaler, RNG states, best score, patience, and epoch. The config/data/split signature must match, including planned epoch count. It resumes an interrupted planned run; it does not silently extend a finished plan. Classical fitting is atomic. Ensemble resume checks the branch configurations and data before reusing them. Completed branches remain frozen for fusion training. Checkpoint selection never uses test metrics.

## Evaluation definitions

- Decision accuracy: correct observed review/aspect labels divided by all observed decisions.
- Pooled sentiment macro F1: pool observed aspects, then average the three sentiment F1 values.
- Mean aspect macro F1: compute a fixed three-sentiment macro F1 separately for each aspect, then average aspects with observations.
- Nine class statistics: precision, recall, F1, support, one-vs-rest AUC/AP for each aspect/sentiment combination.
- Review exact match: all three predictions correct, restricted to fully labeled reviews; the denominator is reported.

Absent classes have F1 zero under the explicitly fixed-class convention; absent positive/negative one-vs-rest examples make AUC/AP undefined, represented by null and a reason. This distinction prevents an undefined ROC area from becoming a misleading zero. Plots and table exports are generated only from saved predictions or executed history. No manuscript metric is a test threshold.

## Statistical comparison API

```python
from bangla_absa.evaluation.predictions import load_predictions, align_predictions
from bangla_absa.evaluation.statistics import mcnemar, cluster_bootstrap, paired_seed_test, holm
from bangla_absa.evaluation.agreement import branch_analysis

ids, gold, p4, p5 = align_predictions(load_predictions("runs/m4-test/predictions.json"),
                                     load_predictions("runs/m5-test/predictions.json"))
counts = mcnemar(gold, p4.argmax(-1), p5.argmax(-1))
interval = cluster_bootstrap(gold, p4.argmax(-1), p5.argmax(-1), seed=17)
agreement = branch_analysis(gold, p4, p5)
```

Alignment requires matching IDs, gold/masks, split, and seed. Exact McNemar uses the observed discordant count with a binomial test; its independence assumption is disclosed. Review-cluster bootstrap resamples whole reviews to retain dependence among their aspects. A paired-seed t-test requires matched seed keys and raw run values; zero variance returns an undefined statistic rather than inventing a finite result. `summarize_seeds` uses sample SD and a Student-t interval for the seed mean. This interval differs from the review bootstrap. `holm` corrects a prespecified family of p-values. Paper-style p-value bounds are not substituted for exact tests.

Branch analysis distinguishes both correct, same wrong answer, one correct, and disagree/neither correct. With M6 predictions it counts corrections, regressions, and selections of neither branch's argmax. The hard-label branch oracle is **not** an upper bound on probability stacking.

## Cost measurement

`bangla-absa benchmark --run runs/m4 --data private/prepared/reviews.jsonl --out runs/benchmark.json` measures a loaded individual neural model after warmup, with synchronization for CUDA, batch size/length/device and repeat counts recorded. The default benchmark excludes preprocessing; that flag is explicit. Parameter counts are actual instantiated counts. Full FLOPs are null with an unsupported reason; partial operator counts are not passed off as total architecture cost. Ensemble latency includes two branch executions and must be measured explicitly for the intended hardware.

## Evidence boundary

The full training commands are runnable orchestration, but no original-corpus experiment was performed in this release. Synthetic tests verify implementation behavior, not paper-level effectiveness. Baseline adaptation choices and deviations must accompany any future table generated with this repository.
