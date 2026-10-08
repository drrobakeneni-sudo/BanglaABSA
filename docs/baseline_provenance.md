# Baseline provenance and task adaptations

These are newly written category-aspect implementations. None is represented as the original author's code or an exact target-span benchmark reproduction. The paper's baseline hyperparameters are mostly unavailable. Shared backbone loading and training code are reused; model-specific recurrence, attention, pairing, and fusion paths remain distinct.

| Comparator | Implementation and adaptation | Primary reference |
|---|---|---|
| NB / SVM | Sparse TF-IDF; MultinomialNB; LinearSVC with sigmoid calibration using only training folds | [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) |
| fastText BiLSTM | Local 300-d Bangla `.vec` initialization, packed bidirectional LSTM, masked pooled 9-logit output | [Official fastText vectors](https://fasttext.cc/docs/en/crawl-vectors.html) |
| MemNet | Category-phrase mean query, repeated additive memory attention, residual query transition | [Tang et al., 2016](https://aclanthology.org/D16-1021/) |
| ATAE-LSTM | Category vector concatenated at every recurrent input, aspect-conditioned additive attention, final recurrent-state fusion | [Wang et al., 2016](https://aclanthology.org/D16-1058/) |
| IAN | Separate context and category-phrase LSTMs; each attends using the other's pooled representation | [Ma et al., 2017](https://www.ijcai.org/proceedings/2017/568) |
| RAM | Bidirectional context memory, clause-distance weighting, recurrent attention with GRU updates | [Chen et al., 2017](https://aclanthology.org/D17-1047/) |
| PLM variants | Actual selected checkpoint architecture, masked mean pool, dropout, one linear 9-logit layer reshaped to 3×3 | Official checkpoints below |
| Three aspect heads | Shared contextual representation with three separate nonlinear classifiers | Manuscript Table 13; head widths reconstructed |
| SPC | `[review, category phrase]` tokenizer sentence pair; shared 3-class CLS classifier for each aspect | [BERT and ABSA, Sun et al.](https://aclanthology.org/N19-1035/) and [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch) |
| BERT Pair QA-M | `[review, category question]` pair; three-class answer classifier; no binary sentiment expansion | [Sun et al., 2019](https://aclanthology.org/N19-1035/) |
| AEN | Context self-attention, category-to-context attention, context-to-category interaction, pointwise feed-forward, concatenated pooled representations, .2 label smoothing | [Song et al., 2019](https://arxiv.org/abs/1902.09314) and [author implementation](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/aen.py) |
| LCF-BERT | Shared local/context encoder and category-pair encoder; input-cue-based local weighting, local/global fusion and attention | [LCF implementation](https://github.com/songyouwei/ABSA-PyTorch/blob/master/models/lcf_bert.py) |
| M5 five-seed vote | Five separately seeded M5 runs with the same manifest; mean aligned probability arrays | Manuscript Table 13; proposed seeds are new |
| M4/M5 equal vote | Mean probabilities of branches trained under the same held-out protocol as M6 | Manuscript Table 13 |
| Logistic stacking | Three LBFGS estimators on the same 18 held-out branch features; class order preserved | Manuscript Table 13; protocol reconstructed |

## Category adaptations

The category phrases are `পণ্য মান`, `দোকান সেবা`, and `পরিবহন ডেলিভারি`. QA-M asks a Bangla question about each phrase's sentiment. IAN and AEN encode these phrases separately rather than inventing their positions in the review. MemNet omits unavailable target-relative location weights. RAM substitutes distance to rule-aligned category clauses for target-span position and appends a normalized distance feature. LCF uses the same clause-distance weighting and broadcasts the global sentence-pair CLS vector before token fusion; this differs from the original span-aligned concatenation. When no category cue is found, the full context is retained. AEN uses PyTorch multihead dot-product attention rather than copying the reference implementation's MLP scoring module; that scoring choice is reconstructed and should be ablated in future scientific comparisons.

Static baselines share official fastText word-vector initialization in full configs; OOV entries remain seeded random trainable vectors. The `.vec` adapter does not synthesize subword OOV vectors from the fastText binary model. All learned resources use training data only. Calibration excludes validation/test labels, though the global TF-IDF vocabulary is fitted on the whole training partition before internal calibration folds; nested fold-specific vectorizer fitting is not claimed.

## Verified checkpoint IDs

The official config and immutable Hub revision for each ID were retrieved on 2026-10-08. See [machine-readable provenance](checkpoint_provenance.json).

| Config | Checkpoint | Architecture / width |
|---|---|---|
| mbert | `google-bert/bert-base-multilingual-cased` | BERT / 768 |
| xlmr_base | `FacebookAI/xlm-roberta-base` | XLM-R / 768 |
| xlmr_large | `FacebookAI/xlm-roberta-large` | XLM-R / 1024 |
| mdeberta | `microsoft/mdeberta-v3-base` | DeBERTa-v2 implementation / 768 |
| banglishbert | `csebuetnlp/banglishbert` | ELECTRA / 768 |
| banglabert, M4, M5 and aspect-conditioned comparators | `csebuetnlp/banglabert` | ELECTRA / 768 |
| banglabert_large | `csebuetnlp/banglabert_large` | ELECTRA / 1024 |

[Official BanglaBERT repository](https://github.com/csebuetnlp/banglabert) also recommends its original normalization pipeline. This release's independent conservative normalization is explicitly different; identical pretrained preprocessing is not claimed. No third-party model weights or source implementations are vendored.
