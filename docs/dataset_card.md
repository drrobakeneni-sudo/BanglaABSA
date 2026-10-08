# Dataset card

## Original manuscript corpus: unavailable

The manuscript describes BanglaABSA 200k, collected from Daraz, Foodpanda, and Bikroy.com during January 2023–December 2024. Reported filtering counts are 412,847 raw, 247,310 after duplicate/language filtering, 218,604 after length filtering, and 200,000 annotated. Reported splits are 140,000/30,000/30,000. These are author-reported facts, not counts measured by this software release.

Aspects: Product, Shop, Transport. Sentiments: Negative, Neutral, Positive. Absence or unavailable annotation is null. The original review/decision-count ambiguity is detailed in [reproduction status](reproduction_status.md).

The restricted corpus, identifiable platform IDs, manuscript, labels, manifests, and model weights are not distributed. This repository does not scrape commercial platforms. A local importer can process data the operator is permitted to use.

## Related public corpus: optional, separate

[Mendeley Data, version 1](https://data.mendeley.com/datasets/n4n5y34p3s/1), DOI **10.17632/n4n5y34p3s.1**, lists Md Nurnabi Shakil and Md. Ashraful Islam Talukder as contributors. Its landing page describes 3,587 annotated Bangla/Banglish reviews, five aspects, and CC BY 4.0. Those are landing-page metadata, rechecked 2026-10-08. Actual downloaded schema and checksums remain unverified because direct retrieval returned HTTP 403.

Optional semantic mapping: product quality→Product, seller service→Shop, delivery→Transport. Price/packaging rows are excluded and counted. Missing labels remain missing. Conflicting labels fail by default or become missing with an explicit recorded policy. Do not describe experiments on this dataset as reproducing the 200k corpus. No such experiments have run here.

## Included synthetic fixtures

`data/examples/synthetic_reviews.jsonl` contains 135 newly authored, mechanically combined examples with Bengali, Banglish substitutions, emoji, multi-aspect labels, and missing Shop labels. Every row is marked synthetic. Numeric suffixes ensure distinct examples and have no natural-review interpretation. These fixtures test software; they cannot support accuracy, generalization, fairness, or external-validation claims.

## Privacy and limitations

Canonical IDs should be opaque, locally generated, and separated from any platform/user identity. Exclude phone numbers, emails, account names, or other personal information from releasable examples. The included contact-redaction helper is a limited aid, not a guarantee of anonymization. Raw reviews are not logged by default. Fitted vocabularies, lexicons, prediction IDs, and text hashes can reveal information, so all real run artifacts remain local and ignored by Git.
