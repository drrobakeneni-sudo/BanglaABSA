# Architectures and reconstruction choices

## Output and loss contract

All neural logits are `[batch, 3, 3]`, ordered Product, Shop, Transport and Negative, Neutral, Positive. Softmax is applied on the last axis. Labels use `-1` internally for unobserved aspects and JSON null externally. Loss computes each aspect's mean over its observed reviews. M4 sums those means (matching the sum of three cross-entropies for full labels); M5, other neural comparators, and M6 average over aspects represented in the batch. An entirely unlabeled batch has differentiable zero loss and is skipped by training.

## M1–M3

All use sparse train-fitted TF-IDF 1–3 grams, with a reconstructed cap of 50,000 features. M1 uses LBFGS, C=1, and a reconstructed maximum of 1,000 optimizer iterations. M2 uses histogram XGBoost, learning rate .080, 500 trees, lambda 1, depth 6 (reconstructed), and validation early stopping 25. Validation labels absent from a training aspect's class set cannot guide its stopping; final evaluation still counts them. M3 uses 800 bootstrap trees and square-root feature sampling. No whole text matrix is densified. Single-class training aspects use an explicit constant estimator; an aspect with no training labels fails.

## M4

`AutoModel` loads the official ELECTRA-based `csebuetnlp/banglabert` contextual encoder at an immutable revision, width 768. A learned 16-wide polarity cue vector is concatenated to each token and projected from 784 to 768. Three distinct post-norm Transformer blocks use eight heads and feed-forward width 3,072. A packed BiLSTM has 64 units per direction; three separate additive attention modules summarize its 128-wide states. Each score is `v_a^T tanh(W_a h_t + b_a)` with masked token softmax. Three dedicated 128→128→3 ReLU classifiers complete the model. Widths and head counts beyond the checkpoint and three-block count are reconstructed.

Packed sequences exclude right padding from both LSTM directions. Attention excludes padding; zero padding cannot change an evaluated prediction.

## M5

The same contextual encoder/cue projection is followed by an additional sinusoidal position encoding, two post-norm encoder layers (eight heads, width 768, feed-forward 3,072), and AASA. Here AASA is explicitly reconstructed as three learned aspect queries attending to token memory with multihead attention plus a query residual. This is a documented realization of the description, not a recovered original equation.

Two decoder layers apply causal aspect self-attention in Product→Shop→Transport order and unmasked cross-attention over nonpadding token memory. The causal mask is a 3×3 upper-triangular Boolean mask; target vectors themselves are never padded. After the decoder, a learned 768→128 projection precedes three independent 128→3 heads. Extra positional encoding, two-layer counts, causal order, and projection placement are reconstruction choices. The backbone already includes positions; removing *extra* positional encoding does not remove the backbone's own positions.

## M6

M4/M5 predictions are frozen. We minimize negative log likelihood of the weighted probability mixture on the designated weight-fitting subset separately for each aspect. Each pair of weights is nonnegative and sums to one, with endpoints included in the search. Weighted branch vectors remain separate:

`[M4 Product N/Neu/P, M4 Shop N/Neu/P, M4 Transport N/Neu/P, M5 Product N/Neu/P, M5 Shop N/Neu/P, M5 Transport N/Neu/P]`.

The resulting 18 features feed 18→768→576→9, with ReLU and dropout .30 after each hidden layer. The last layer is reshaped into three independent sentiment outputs. Including biases, `18*768+768 + 768*576+576 + 576*9+9 = 462729` trainable parameters. Full branch-plus-stacker totals are measured, not forced to manuscript totals.

## Ablation replacements

| Removal | Replacement |
|---|---|
| M4 contextual embedding | Random trainable token lookup, same output width |
| M4 refinement | Identity before sequential layer |
| M4 BiLSTM | Refined token sequence directly into correctly sized heads |
| M4 aspect attention | Masked mean pooling, three dedicated classifiers |
| M4 DNN | Linear classifier for each aspect |
| M5 pretrained initialization | Same official backbone configuration, random parameters |
| M5 extra positions / encoder / decoder | Identity at the removed stage |
| M5 AASA | Shared masked token mean expanded to three decoder targets |
| M5 single head | One shared 128→3 head applied separately to each aspect vector |
| M6 equal vote | Arithmetic mean, no learned weights or stacker |
| M6 weighted vote | Fitted mixture only, no stacker |
| M6 unweighted stacking | Both branch vectors scaled .5, preserving 18 features |

All ablations require retraining. Neither tiny smoke dimensions nor a random-backbone forward pass should be interpreted as full-model replication.
