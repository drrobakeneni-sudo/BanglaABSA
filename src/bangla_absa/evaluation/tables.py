import csv
from pathlib import Path

from bangla_absa.evaluation.metrics import evaluate
from bangla_absa.evaluation.predictions import load_predictions


def export_tables(prediction_files, output):
    rows = []
    for filename in prediction_files:
        _, y, p, meta = load_predictions(filename)
        metrics = evaluate(y, p)
        rows.append(
            {
                "checkpoint_id": meta["checkpoint_id"],
                "seed": meta["seed"],
                "split": meta["split"],
                **{
                    k: metrics[k]
                    for k in (
                        "reviews",
                        "decisions",
                        "decision_accuracy",
                        "pooled_sentiment_macro_f1",
                        "mean_aspect_macro_f1",
                        "exact_match_review_accuracy",
                    )
                },
            }
        )
    if not rows:
        raise ValueError("Provide at least one executed prediction file")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows
