# Annotation guidelines

## Manuscript-described practice

The manuscript assigns applicable Product, Shop, and Transport aspects Negative, Neutral, or Positive. It describes independent assessment of 10,000 reviews by three Bangla-proficient annotators, blinded to reference labels and each other, followed by consensus with a senior adjudicator. Reported nominal alpha values are .867 Product, .845 Shop, .831 Transport, and .848 pooled. These are manuscript claims; original rating matrices were not supplied and these coefficients have not been recomputed.

## Operational guidance created for this release

- **Product:** item quality, condition intrinsic to the item, functionality, or characteristics. **Shop:** seller/platform service, communication, responsiveness, and ordering. **Transport:** delivery speed, logistics, and damage explicitly attributed to handling. Use context to resolve item-condition versus handling ambiguity; do not assume both aspects without evidence.
- **Positive:** favorable evaluation or clear favorable implication. **Negative:** complaint, dissatisfaction, or unfavorable implication. **Neutral:** an aspect is discussed but without favorable or unfavorable orientation. **Null:** absent aspect, unavailable annotation, or unresolved uncertainty. Absence is never Neutral.
- Read the whole review before assigning aspect labels. Interpret negation with its scope: `ভালো না` is not Positive. Preserve contrastive clauses; product praise must not overwrite a delivery complaint.
- For mixed polarity within one aspect, record both evidence clauses and flag for adjudication. This release recommends resolving the dominant final assessment only when explicit; otherwise leave unresolved/missing. This is a new operational policy, not a recovered original tie-break rule.
- Infer implicit polarity only when supported by ordinary context. Sarcasm, quoted speech, unfamiliar slang, and contradictory emoji require an uncertainty flag; emoji alone does not automatically override literal text.
- Treat Bangla, Banglish, and English code mixing as one utterance. Unknown transliterations should be reviewed by a proficient reader, not converted into a default sentiment.
- Annotators label independently without seeing model outputs or other labels. Keep individual decisions and evidence notes before adjudication. A separate adjudicator resolves disagreements and records the reason; never overwrite the raw annotation history.

## Synthetic worked examples

| Authored example | Product | Shop | Transport | Rationale |
|---|---|---|---|---|
| পণ্য ভালো কিন্তু ডেলিভারি দেরি। | Positive | null | Negative | Separate aspects and contrast |
| দোকান থেকে উত্তর পেয়েছি। | null | Neutral | null | Service mentioned without explicit evaluation |
| পণ্য ভালো না। | Negative | null | null | Negation |
| ponno bhalo, delivery late | Positive | null | Negative | Code mixing |
| বাহ, আবার ভাঙা পণ্য! | Negative | null | null | Sarcastic exclamation with damage; context still matters |

The examples are newly authored and do not come from platform reviews. Use `data/templates/annotation.csv` for annotation work; export final labels to the separate canonical template. Uncertainty and adjudication notes are not model input features.

## Agreement computation

`data.annotation.nominal_agreement` accepts units×annotators with missing decisions. Raw agreement is the fraction of agreeing **unordered observed annotator pairs**, across units with at least two ratings. This explicit definition may differ from the paper's unspecified raw agreement convention. Nominal Krippendorff alpha uses unit-normalized coincidence counts; singleton units do not contribute. Zero expected disagreement or no pairable units returns null with a reason. The implementation is tested against the `krippendorff` package including missing ratings. Calculate per aspect, and pool review/aspect units only when documenting the denominator.
