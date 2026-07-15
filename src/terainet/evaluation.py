"""Evaluation, reporting, and experiment logging for TeraiNet models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from terainet.config import TrainingConfig
from terainet.training import TrainingRunResult


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics, labels, and report produced for one named evaluation split."""

    name: str
    metrics: dict[str, float]
    true_labels: np.ndarray
    predicted_labels: np.ndarray
    report: pd.DataFrame
    confusion_matrix: np.ndarray


def collect_predictions(model: Any, dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collect global class indices and predictions from a categorical or binary dataset.

    A single sigmoid output (binary classification) is detected from the prediction
    width and thresholded at 0.5; a multi-unit softmax output is read via argmax,
    matching the one-hot label contract used elsewhere in the pipeline.
    """
    true_labels: list[np.ndarray] = []
    predicted_labels: list[np.ndarray] = []
    for images, labels in dataset:
        label_array = np.asarray(labels.numpy())
        predictions = np.asarray(model.predict(images, verbose=0))
        if predictions.shape[-1] == 1:
            true_labels.append(label_array.reshape(-1).astype(int))
            predicted_labels.append((predictions.reshape(-1) >= 0.5).astype(int))
        else:
            true_labels.append(np.argmax(label_array, axis=-1))
            predicted_labels.append(np.argmax(predictions, axis=-1))

    if not true_labels:
        raise ValueError("Cannot evaluate an empty dataset.")
    return np.concatenate(true_labels), np.concatenate(predicted_labels)


def _classification_report(
    true_labels: np.ndarray, predicted_labels: np.ndarray, class_names: tuple[str, ...]
) -> pd.DataFrame:
    labels = list(range(len(class_names)))
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    report.pop("accuracy", None)
    report["micro avg"] = {
        "precision": precision_score(
            true_labels, predicted_labels, labels=labels, average="micro", zero_division=0
        ),
        "recall": recall_score(
            true_labels, predicted_labels, labels=labels, average="micro", zero_division=0
        ),
        "f1-score": f1_score(
            true_labels, predicted_labels, labels=labels, average="micro", zero_division=0
        ),
        "support": int(len(true_labels)),
    }
    ordered_keys = [*class_names, "micro avg", "macro avg", "weighted avg"]
    return pd.DataFrame({key: report[key] for key in ordered_keys}).transpose()


def evaluate_dataset(
    model: Any,
    dataset: Any,
    class_names: tuple[str, ...],
    name: str,
    restrict_macro_to_observed_classes: bool = False,
) -> EvaluationResult:
    """Evaluate a dataset and compute stable per-class and aggregate metrics.

    When `restrict_macro_to_observed_classes` is True, macro-averaged precision, recall,
    and f1 are computed only over classes that actually appear in `true_labels`. This
    keeps partial-coverage evaluations (e.g. an out-of-distribution split that only
    contains a subset of the configured classes) from being diluted by classes with no
    ground truth, which would otherwise force their recall to zero. The classification
    report and confusion matrix always retain the full configured class order.
    """
    true_labels, predicted_labels = collect_predictions(model, dataset)
    labels = list(range(len(class_names)))
    macro_labels = (
        sorted(set(true_labels.tolist())) if restrict_macro_to_observed_classes else labels
    )
    metrics = {
        "accuracy": float(np.mean(true_labels == predicted_labels)),
        "precision_macro": float(
            precision_score(
                true_labels, predicted_labels, labels=macro_labels, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                true_labels, predicted_labels, labels=macro_labels, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                true_labels, predicted_labels, labels=macro_labels, average="macro", zero_division=0
            )
        ),
    }
    return EvaluationResult(
        name=name,
        metrics=metrics,
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        report=_classification_report(true_labels, predicted_labels, class_names),
        confusion_matrix=confusion_matrix(true_labels, predicted_labels, labels=labels),
    )


def save_evaluation_artifacts(result: EvaluationResult, config: TrainingConfig) -> None:
    """Write held-out test report and confusion matrix artifacts to configured paths."""
    report_path = config.artifact_paths["classification_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    result.report.to_csv(report_path)

    confusion_matrix_path = config.artifact_paths["confusion_matrix"]
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        result.confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.class_names,
        yticklabels=config.class_names,
        ax=axis,
    )
    axis.set(xlabel="Predicted", ylabel="True", title=f"{result.name.title()} confusion matrix")
    figure.savefig(confusion_matrix_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def log_run_to_wandb(
    config: TrainingConfig,
    run_result: TrainingRunResult,
    test_result: EvaluationResult,
    ood_result: EvaluationResult | None,
) -> None:
    """Log an already-authenticated run and its artifacts to Weights & Biases."""
    if not config.wandb.get("enabled", False):
        return

    import wandb

    run = wandb.init(
        project=config.wandb["project"],
        config={
            "backbone": config.backbone,
            "backbone_parameters": run_result.model_metadata.backbone_parameters,
            "frozen_file_size": run_result.model_metadata.frozen_file_size,
            "image_size": config.image_size,
            "class_names": list(config.class_names),
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "seed": config.seed,
        },
    )
    log_payload: dict[str, float] = {"training_time_minutes": run_result.training_time_minutes}
    log_payload.update({f"test/{name}": value for name, value in test_result.metrics.items()})
    if ood_result is not None:
        log_payload.update({f"ood/{name}": value for name, value in ood_result.metrics.items()})
    wandb.log(log_payload)

    artifact = wandb.Artifact("classification_results", type="evaluation")
    for artifact_path in config.artifact_paths.values():
        if Path(artifact_path).is_file():
            artifact.add_file(str(artifact_path))
    wandb.log_artifact(artifact)
    run.finish()
