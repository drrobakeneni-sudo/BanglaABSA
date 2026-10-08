# Reproduction status

## Evidence vocabulary

**Documented** means stated in the supplied manuscript. **Reconstructed** means a new implementation choice. **Implemented** means source exists. **Tested** means exercised by tests. **Executed** refers to a recorded run, with data and scale identified. **Awaiting original artifact** means exact recovery cannot be checked. Passing synthetic tests does not establish research performance.

## Blocking issues for exact reproduction

1. **Review versus decision counts.** Sections 3 and 4 describe 30,000 test reviews with three outputs each, implying 90,000 decisions if fully labeled. Table 2 and several later tables contain 30,000 decisions, 10,000 per aspect. We do not resolve this by inventing labels. Null aspect labels are masked, and evaluation reports reviews, reviews with labels, valid aspect decisions, and fully labeled review counts separately. The original label policy and manifests are required.
2. **Missing original artifacts.** No original corpus, private split IDs, transliteration dictionary, parser/model, cue lexicon, alignment rules, fitted resources, checkpoints, predictions, or logs were provided. Training seeds, many dimensions, the AASA formulation, fusion-weight estimator, and stacker partition protocol are unspecified.
3. **Configuration discrepancies.** M4 equations specify three post-norm refinement blocks. Figure 2 displays a sequence length of 75 while the text/Table 4 specify 256; 256 including special tokens is used. Its attention dimension label is ambiguous. Figure 1 includes Evaly and LightGBM/CatBoost labels inconsistent with the dataset/model descriptions, and a Bi-GRU label inconsistent with the M5 text. The implemented models follow Sections 4.2–4.4 and Tables 4, 10–13. Figure 3 labels a second decoder and 128-to-3 heads but does not specify internal widths or the projection location.
4. **Parameter totals.** The 462,729-parameter M6 MLP is exact. Overall branch totals depend on unspecified dimensions and differ from the manuscript. No architecture is resized to force the reported 118.42M/142.06M totals. Constructed counts are recorded separately from training outcomes.
5. **Metric conventions.** Pooled sentiment macro F1, mean aspect macro F1, and the nine aspect/sentiment class statistics are distinct outputs. Missing ROC classes produce null with a reason. The paper's aggregate, per-aspect, and classwise summaries cannot be treated as interchangeable ground truth.
6. **Related public dataset.** Mendeley DOI 10.17632/n4n5y34p3s.1 describes 3,587 reviews and five aspects, not the 200k corpus. The landing page and CC BY 4.0 declaration were rechecked on 2026-10-08. Direct retrieval returned HTTP 403 in this environment, so the downloaded files, column schema, and checksums are **not verified**. The optional long-form adapter requires explicit input-column mapping and conflict handling. No external validation was executed.

## What runs now

The offline tests and synthetic CPU workflows exercise data import, normalization, grouping, training-only vocabulary/cue fitting, all registered model families, ablations, gradients, masks, checkpoint reload, calibration, held-out stacking, selection boundaries, metric calculations, statistics, plotting, and commands. Tiny contextual encoders and randomly initialized small real backbone classes are test fixtures only. See [verification](verification.md) for the actual measurements.

## What has not run

No full 200k training, five-seed scientific experiment, full-size pretrained fine-tuning, GPU benchmark, or independent real-data validation was performed. CV and seed orchestration are implemented and tested on synthetic data. Complete end-to-end FLOPs are unsupported and reported as null. No performance table or figure is represented as an independently reproduced manuscript result.

Setting a configuration's `mode` to `exact` deliberately fails. Removing that guard would require verifying the original artifacts and resolving the issues above; merely supplying similarly sized data is insufficient.
