from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bangla_absa import SENTIMENTS
from bangla_absa.evaluation.metrics import curves, evaluate
from bangla_absa.io import write_json


def plot_predictions(y, p, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metrics, series = evaluate(y, p), curves(y, p)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), facecolor="white", layout="constrained")
    matrix = metrics["pooled"]["confusion"]
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].set(
        xticks=range(3),
        yticks=range(3),
        xticklabels=SENTIMENTS,
        yticklabels=SENTIMENTS,
        xlabel="Predicted",
        ylabel="Observed",
    )
    for i in range(3):
        for j in range(3):
            axes[0].text(j, i, str(matrix[i][j]), ha="center", va="center")
    for name, values in series.items():
        if "reason" in values:
            continue
        axes[1].plot(values["fpr"], values["tpr"], label=name)
        axes[2].plot(values["recall"], values["precision"], label=name)
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set(
        xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.02)
    )
    axes[2].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes[1:]:
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    fig.savefig(directory / "evaluation.png", dpi=180)
    fig.savefig(directory / "evaluation.svg")
    plt.close(fig)
    write_json(directory / "curves.json", series)
    (directory / "caption.txt").write_text(
        "Observed aspect decisions pooled across reviews. Confusion counts, one-vs-rest ROC, "
        "and precision-recall curves computed from the supplied predictions. Missing labels "
        "are excluded. Undefined curves are omitted; see curves.json. Synthetic runs are not "
        "scientific performance evidence.\n",
        encoding="utf-8",
    )


def plot_history(history, path):
    if not history:
        raise ValueError("No executed training history")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    axes[0].plot([r["epoch"] for r in history], [r["train_loss"] for r in history])
    axes[0].set(xlabel="Epoch", ylabel="Mean training batch loss")
    axes[1].plot(
        [r["epoch"] for r in history], [r["validation_mean_aspect_macro_f1"] for r in history]
    )
    axes[1].set(xlabel="Epoch", ylabel="Validation mean aspect macro F1")
    fig.savefig(path, dpi=180)
    plt.close(fig)
