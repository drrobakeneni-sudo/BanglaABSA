# Splits and leakage boundaries

The reported 70:15:15 split (140k/30k/30k) is manuscript metadata, not a requirement imposed on another dataset. Original IDs and the stratification seed are unavailable.

The reconstructed splitter first takes the transitive union of exact normalized-text duplicates and explicit `group_id` membership. It sorts records by opaque ID and orders groups by size with deterministic seeded tie-breaking. Greedy allocation minimizes the increase in normalized squared discrepancy from each partition's target counts across nine aspect/sentiment strata and review counts. This is an approximate grouped multi-aspect stratifier; sparse strata and indivisible groups can prevent exact proportions. It enforces nonempty partitions when enough independent groups exist, and otherwise fails. It does not promise that every class exists in every partition.

Manifests map each ID to exactly one partition. Validation rejects missing/extra IDs, invalid partitions, and duplicate/group overlap. If an original manifest is supplied, it is preserved only when it remains valid after preprocessing. Do not publish private IDs, raw text, or text hashes as a substitute for authorization.

## Held-out stacking reconstruction

The original stacking protocol is unspecified. This repository creates disjoint subsets of the original validation partition:

| Subset | Ratio within validation | Consumed by |
|---|---:|---|
| weight_fit | 25% | Per-aspect branch-weight NLL minimization |
| meta_train | 50% | Stacker fitting |
| meta_validation | 25% | M4/M5 checkpoint selection and stacker checkpoint selection |
| original TRAIN | unchanged | Branch parameters, tokenizer/vocabulary, cue lexicon |
| original TEST | unchanged | Final evaluation only |

Group boundaries are enforced across all five roles. Weight/meta fitting rejects IDs used to train or select branch checkpoints. Branches are retrained under this protocol, so a separately trained ordinary-validation M4 checkpoint is not silently reused. Shared branch/stacker use of meta-validation is disclosed and is a model-selection choice; the untouched test partition is still required. No out-of-fold stacking mode is claimed by this release.

Five-fold CV partitions only original TRAIN+VALIDATION. Each outer training fold receives fresh inner train/validation partitions and fresh fitted resources. M6 subdivides inner validation again. The untouched outer fold is evaluated only after fitting; original TEST is absent from every development-fold input.

Prediction caches include ordered IDs and input, config, checkpoint, and split fingerprints. A mismatch raises an error rather than reusing stale predictions. These cache files stay private.
