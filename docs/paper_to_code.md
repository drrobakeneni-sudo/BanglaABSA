# Paper-to-code evidence map

Paths below are relative to the repository root. **Implemented/tested** refers to synthetic tests, not scientific replication. All table values in the manuscript remain unverified author-reported results.

| Manuscript component | Source / config | Evidence status |
|---|---|---|
| Section 3; Tables 1–2 corpus/splits/distribution | `data/schema.py`, `data/splits.py`, `configs/data/canonical.json`, dataset card | Reported counts documented; original data/manifest awaiting artifact; split/mask code implemented/tested |
| Section 3.1 annotation protocol | `data/annotation.py`, annotation guidelines, empty templates | Guidelines documented/reconstructed; alpha computation tested against krippendorff; original ratings unavailable |
| Section 4.1 normalization/transliteration | `preprocessing/normalize.py`, `transliterate.py`, demo resources | Rules implemented/tested; original dictionary missing |
| Section 4.1.2 alignment | `preprocessing/segment.py` | Input-only rules implemented/tested; original dependency parser unavailable |
| Section 4.1.3 cue embeddings | `preprocessing/cues.py`, `tokenization.py`, `models/common.py` | Train-only fitting and subword mapping reconstructed/implemented/tested |
| M4 equations (1)–(2), tokenizer/context | `models/common.py:HFEncoder`, `preprocessing/tokenization.py`, `configs/models/m4.json` | ELECTRA checkpoint identity verified; original tokenizer preprocessing unavailable |
| M4 equations (3)–(4), three post-norm blocks | `models/m4_absa_net.py` | Block count/form documented and tested; head/feed-forward settings reconstructed |
| M4 equations (5)–(6), BiLSTM | `models/m4_absa_net.py`, `models/common.py:run_lstm` | Packed bidirectional recurrence implemented/tested; width reconstructed |
| M4 equations (7)–(9), additive attention | `models/common.py:AdditiveAttention` | Equation implemented/tested with masking |
| M4 equations (10)–(11), classifiers | `models/m4_absa_net.py` | Separate two-layer heads implemented/tested; hidden width reconstructed |
| M4 equation (12), objective | `models/common.py:masked_loss` | Sum of aspect means; missing-label extension explicit/tested |
| M5 Section 4.3 / Figure 3 | `models/m5_transnet.py`, `configs/models/m5.json` | Full path implemented/tested; AASA, internal dimensions, causal mask, projection location reconstructed |
| M6 Section 4.4 / Figure 4 equations | `models/m6_ensemble.py`, `training/ensemble.py`, `configs/models/m6.json` | 18-feature order and 462729-parameter MLP documented/tested; weight estimator/protocol reconstructed |
| Tables 3–4 environment/hyperparameters | `configs/training/protocol.json`, model configs, `training/trainer.py`, `training/checkpoints.py` | Historical values documented; different tested runtime disclosed; original exact layer settings missing |
| Table 5 computational cost | `evaluation/benchmark.py`, `docs/parameter_counts.json` | Parameter counting and CPU latency implemented/executed on stated configs; total FLOPs unsupported; original timings unverified |
| Table 6 aggregate performance | `evaluation/metrics.py`, `statistics.py`, `scripts/run_seeds.py` | Computations implemented/tested; synthetic runs executed; original five-seed values unavailable |
| Table 7 per-aspect F1 | `evaluation/metrics.py` | Explicit per-aspect macro implemented/tested; no forced matching to paper |
| Table 8 nine class statistics | `evaluation/metrics.py` | Precision/recall/F1/support/AUC/AP implemented/tested |
| Table 9 five-fold CV | `training/folds.py`, `scripts/run_cv.py` | Fresh inner fits; original TEST excluded; synthetic classical CV executed; full corpus CV not executed |
| Table 10 M4 ablations | `configs/ablations/m4*.json`, `models/m4_absa_net.py` | Shape-compatible computational removals implemented/tested; scientific retraining not executed |
| Table 11 M5 ablations | `configs/ablations/m5*.json`, `models/m5_transnet.py` | All removals and shared-head variant implemented/tested; full-scale comparisons not executed |
| Table 12 fusion ablations | ensemble configs, `training/ensemble.py` | Equal/weighted voting and unweighted stacking implemented/tested/executed synthetically |
| Table 13 all comparators | `models/registry.py`, `models/baselines/`, `configs/models/` | Distinct families implemented/tested; checkpoint IDs verified; task adaptations explicit; full benchmark not executed |
| Table 14 significance | `evaluation/statistics.py`, `predictions.py` | Matched-seed t test, exact McNemar, review bootstrap and Holm implemented/tested; paper's rounded-summary bounds not used as exact tests |
| Table 15 agreement/oracle | `evaluation/agreement.py`, `predictions.py` | Matched decision analysis implemented/tested; nonlinear stacker may choose neither branch |
| Figures 5–6 optimization curves | `evaluation/plots.py:plot_history` | Generated from real run history only; no copied/invented trajectories |
| Figures 7–8 ROC/confusion | `evaluation/plots.py:plot_predictions` | Generated from supplied probabilities/masks; undefined curves disclosed |

Module paths in this table live under `src/bangla_absa/` unless another root is shown. See [reproduction status](reproduction_status.md) for figure/text contradictions, [baseline provenance](baseline_provenance.md) for comparator sources, and [verification](verification.md) for executed release checks.
