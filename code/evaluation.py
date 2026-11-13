
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def _balanced_accuracy_present_classes(y_true, y_pred) -> float:

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.unique(y_true)
    if not len(labels):
        return float("nan")
    return float(
        recall_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )
    )


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _probability_matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("model probabilities must be a finite, non-negative NxC matrix")
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0):
        raise ValueError("every probability row must have positive mass")
    return values / totals


def expected_calibration_error(y_true, probabilities, bins: int = 15) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _probability_matrix(probabilities)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y_true
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        include = (confidence > low) & (confidence <= high)
        if np.any(include):
            result += include.mean() * abs(correct[include].mean() - confidence[include].mean())
    return float(result)


def metric_set(y_true, probabilities) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = _probability_matrix(probabilities)
    if probabilities.ndim != 2 or len(probabilities) != len(y_true):
        raise ValueError("probabilities must be an NxC matrix aligned with y_true")
    predictions = probabilities.argmax(axis=1)
    labels = np.arange(probabilities.shape[1])
    result = {
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": _balanced_accuracy_present_classes(y_true, predictions),
        "macro_precision": precision_score(
            y_true, predictions, labels=labels, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, predictions, labels=labels, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(y_true, predictions, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            y_true, predictions, labels=labels, average="weighted", zero_division=0
        ),
        "cross_entropy": log_loss(y_true, probabilities, labels=labels),
        "ece_15_bin": expected_calibration_error(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels),
    }
    try:
        if probabilities.shape[1] == 2:
            result["roc_auc"] = roc_auc_score(y_true, probabilities[:, 1])
        else:
            result["macro_roc_auc_ovr"] = roc_auc_score(
                y_true, probabilities, labels=labels, multi_class="ovr", average="macro"
            )
    except ValueError:
        result["roc_auc" if probabilities.shape[1] == 2 else "macro_roc_auc_ovr"] = np.nan
    return result


class TemperatureScaler:

    def __init__(self):
        self.temperature = 1.0

    def fit(self, probabilities, targets):
        probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0)
        logits = np.log(probabilities)
        targets = np.asarray(targets, dtype=int)

        def objective(log_temperature):
            scaled = logits / np.exp(log_temperature)
            scaled -= scaled.max(axis=1, keepdims=True)
            probs = np.exp(scaled)
            probs /= probs.sum(axis=1, keepdims=True)
            return log_loss(targets, probs, labels=np.arange(probs.shape[1]))

        result = minimize_scalar(objective, bounds=(-3.0, 3.0), method="bounded")
        if result.success:
            self.temperature = float(np.exp(result.x))
        return self

    def transform(self, probabilities):
        logits = np.log(np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1.0))
        logits /= self.temperature
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)


def clustered_intervals(
    y_true,
    probabilities,
    clusters: Iterable[str],
    iterations: int,
    seed: int,
    confidence: float = 0.95,
):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    clusters = np.asarray(list(clusters), dtype=str)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {}
    rows = {group: np.flatnonzero(clusters == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1")}
    iterations = int(iterations)
    progress_step = max(1, iterations // 10)
    LOGGER.info(
        "Patient-clustered bootstrap started | patients=%d iterations=%d",
        len(unique),
        iterations,
    )
    for iteration in range(iterations):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        predicted = probabilities[indices].argmax(axis=1)
        values["accuracy"].append(float(accuracy_score(y_true[indices], predicted)))
        values["balanced_accuracy"].append(
            _balanced_accuracy_present_classes(y_true[indices], predicted)
        )
        values["macro_f1"].append(
            float(f1_score(y_true[indices], predicted, average="macro", zero_division=0))
        )
        if (iteration + 1) % progress_step == 0 or iteration + 1 == iterations:
            LOGGER.info("Bootstrap progress: %d/%d", iteration + 1, iterations)
    alpha = (1.0 - confidence) / 2.0
    return {
        name: {
            "lower": float(np.quantile(samples, alpha)),
            "upper": float(np.quantile(samples, 1.0 - alpha)),
        }
        for name, samples in values.items()
    }


def paired_clustered_delta(
    y_true,
    full_probabilities,
    ablated_probabilities,
    clusters,
    iterations: int,
    seed: int,
):

    y_true = np.asarray(y_true)
    full_probabilities = np.asarray(full_probabilities)
    ablated_probabilities = np.asarray(ablated_probabilities)
    clusters = np.asarray(clusters, dtype=str)
    unique = np.unique(clusters)
    if len(unique) < 2:
        return {}
    rows = {group: np.flatnonzero(clusters == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = {name: [] for name in ("accuracy", "balanced_accuracy", "macro_f1")}
    iterations = int(iterations)
    progress_step = max(1, iterations // 10)
    LOGGER.info(
        "Paired modality bootstrap started | patients=%d iterations=%d",
        len(unique),
        iterations,
    )
    for iteration in range(iterations):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([rows[group] for group in sampled])
        truth = y_true[indices]
        full_predicted = full_probabilities[indices].argmax(axis=1)
        ablated_predicted = ablated_probabilities[indices].argmax(axis=1)
        full = {
            "accuracy": accuracy_score(truth, full_predicted),
            "balanced_accuracy": _balanced_accuracy_present_classes(truth, full_predicted),
            "macro_f1": f1_score(truth, full_predicted, average="macro", zero_division=0),
        }
        ablated = {
            "accuracy": accuracy_score(truth, ablated_predicted),
            "balanced_accuracy": _balanced_accuracy_present_classes(truth, ablated_predicted),
            "macro_f1": f1_score(truth, ablated_predicted, average="macro", zero_division=0),
        }
        for name in deltas:
            deltas[name].append(float(full[name]) - float(ablated[name]))
        if (iteration + 1) % progress_step == 0 or iteration + 1 == iterations:
            LOGGER.info("Paired bootstrap progress: %d/%d", iteration + 1, iterations)
    return {
        name: {
            "mean_delta": float(np.mean(values)),
            "lower_95": float(np.quantile(values, 0.025)),
            "upper_95": float(np.quantile(values, 0.975)),
            "probability_delta_positive": float(np.mean(np.asarray(values) > 0)),
        }
        for name, values in deltas.items()
    }


def _predict(model, features, batch_size):
    values = model.predict(features, batch_size=batch_size, verbose=0)
    if isinstance(values, dict):
        values = values["fused"]
    return _probability_matrix(values)


def _save_standard_artifacts(
    y_true,
    probabilities,
    class_names,
    output_dir: Path,
    context: Optional[dict] = None,
):

    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true, dtype=int)
    probabilities = _probability_matrix(probabilities)
    labels = np.arange(probabilities.shape[1])
    predicted = probabilities.argmax(axis=1)
    if class_names is None:
        class_names = [f"class_{index}" for index in labels]
    class_names = list(class_names)
    if len(class_names) != len(labels):
        raise ValueError("class_names must match the model output count")

    report_text = classification_report(
        y_true,
        predicted,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    report_file_text = report_text
    if context:
        context_text = "\n".join(f"{key}: {value}" for key, value in context.items())
        report_file_text = (
            "EXPERIMENT CONTEXT\n"
            "==================\n"
            f"{context_text}\n\n"
            "CLASSIFICATION REPORT\n"
            "=====================\n"
            f"{report_text}"
        )
    (output_dir / "classification_report.txt").write_text(
        report_file_text, encoding="utf-8"
    )
    report_table = classification_report(
        y_true,
        predicted,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report_table).transpose().to_csv(output_dir / "classification_report.csv")

    matrix = confusion_matrix(y_true, predicted, labels=labels)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv", index_label="true/predicted"
    )
    figure, axis = plt.subplots(figsize=(max(6, len(labels) * 1.5), max(5, len(labels) * 1.2)))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    threshold = matrix.max() / 2.0 if matrix.size and matrix.max() else 0.0
    for row, column in np.ndindex(matrix.shape):
        axis.text(
            column,
            row,
            str(matrix[row, column]),
            ha="center",
            va="center",
            color="white" if matrix[row, column] > threshold else "black",
        )
    axis.set_xticks(labels, class_names, rotation=35, ha="right")
    axis.set_yticks(labels, class_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    plotted = False
    for index, name in enumerate(class_names):
        binary_target = (y_true == index).astype(int)
        if len(np.unique(binary_target)) < 2:
            continue
        false_positive, true_positive, _ = roc_curve(binary_target, probabilities[:, index])
        area = auc(false_positive, true_positive)
        axis.plot(false_positive, true_positive, lw=2, label=f"{name} (AUC={area:.3f})")
        plotted = True
    axis.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    axis.set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves")
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize=8, loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    confidence = probabilities.max(axis=1)
    correct = predicted == y_true
    figure, axis = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 16)
    axis.hist(confidence[correct], bins=bins, alpha=0.65, label="correct", color="#2878B5")
    axis.hist(confidence[~correct], bins=bins, alpha=0.65, label="incorrect", color="#C82423")
    axis.set(xlabel="Maximum predicted probability", ylabel="Patch count", title="Confidence distribution")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "confidence_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    bin_ids = np.minimum(np.digitize(confidence, bins) - 1, len(bins) - 2)
    observed, predicted_confidence = [], []
    for bin_index in range(len(bins) - 1):
        included = bin_ids == bin_index
        if np.any(included):
            observed.append(float(correct[included].mean()))
            predicted_confidence.append(float(confidence[included].mean()))
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    axis.plot(predicted_confidence, observed, "o-", color="#2878B5", label="model")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel="Mean predicted confidence",
        ylabel="Observed accuracy",
        title="Reliability diagram",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close(figure)

    return report_text


def evaluate_partition(
    model,
    partition,
    output_dir: Path,
    batch_size: int,
    bootstrap_iterations: int,
    seed: int,
    modality: str,
    temperature: Optional[TemperatureScaler] = None,
    class_names=None,
    partition_name: str = "evaluation",
    split_seed: Optional[int] = None,
    experiment_role: str = "single",
    ablation_variant: Optional[str] = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("=" * 72)
    LOGGER.info(
        "EVALUATION START | partition=%s split_seed=%s samples=%d patients=%d files=%d "
        "ACTIVE_MODALITY=%s role=%s variant=%s calibrated=%s output=%s",
        partition_name,
        split_seed if split_seed is not None else seed,
        len(partition.targets),
        len(np.unique(partition.patient_ids)),
        len(np.unique(partition.filenames)),
        modality,
        experiment_role,
        ablation_variant or "not_applicable",
        temperature is not None,
        output_dir,
    )
    LOGGER.info("Generating predictions in batches of %d", batch_size)
    full_raw = _predict(model, partition.features, batch_size)
    LOGGER.info("Prediction complete | probability_shape=%s", full_raw.shape)
    full = temperature.transform(full_raw) if temperature else full_raw
    metrics = metric_set(partition.targets, full)
    class_counts = np.bincount(
        np.asarray(partition.targets, dtype=int), minlength=full.shape[1]
    ).tolist()
    evaluation_context = {
        "split_seed": int(seed if split_seed is None else split_seed),
        "partition": str(partition_name),
        "active_modality": str(modality),
        "experiment_role": str(experiment_role),
        "ablation_variant": ablation_variant or "not_applicable",
        "samples": int(len(partition.targets)),
        "class_counts": class_counts,
        "patients": int(len(np.unique(partition.patient_ids))),
        "files": int(len(np.unique(partition.filenames))),
        "calibrated": bool(temperature is not None),
    }
    (output_dir / "evaluation_context.json").write_text(
        json.dumps(evaluation_context, indent=2), encoding="utf-8"
    )
    report = {
        "evaluation_context": evaluation_context,
        "metrics": metrics,
        "patient_clustered_95_ci": clustered_intervals(
            partition.targets,
            full,
            partition.patient_ids,
            bootstrap_iterations,
            seed,
        ),
        "temperature": None if temperature is None else temperature.temperature,
        "bootstrap_balanced_accuracy_definition": (
            "macro recall over classes present in each patient-cluster resample"
        ),
    }
    predictions = pd.DataFrame(
        {
            "patch_uid": partition.patch_ids,
            "patient_id": partition.patient_ids,
            "filename": partition.filenames,
            "target": partition.targets,
            "prediction": full.argmax(axis=1),
            "confidence": full.max(axis=1),
        }
    )
    for index in range(full.shape[1]):
        predictions[f"probability_class_{index}"] = full[:, index]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    LOGGER.info("Saving classification report, confusion matrix, ROC, confidence, and calibration plots")
    report_text = _save_standard_artifacts(
        partition.targets,
        full,
        class_names,
        output_dir,
        context=evaluation_context,
    )
    scalar_metrics = {
        key: value for key, value in metrics.items() if np.isscalar(value)
    }
    summary_lines = [
        "Per-seed evaluation metrics",
        "=" * 32,
        *[f"{key}: {value}" for key, value in evaluation_context.items()],
        "",
    ]
    summary_lines.extend(
        f"{name}: {float(value):.6f}" if np.isfinite(value) else f"{name}: unavailable"
        for name, value in scalar_metrics.items()
    )
    summary_lines.extend(["", "Classification report", "-" * 32, report_text])
    (output_dir / "metrics_summary.txt").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )

    if modality == "both":
        LOGGER.info("Running modality contribution audit: without RGB and without spectra")
        ablations = {}
        variants = {}
        without_rgb = partition.features.copy()
        without_rgb[..., :3] = 0.0
        variants["without_rgb"] = without_rgb
        without_spectra = partition.features.copy()
        without_spectra[..., 3:-1] = 0.0
        variants["without_spectra"] = without_spectra
        for name, features in variants.items():
            LOGGER.info("Ablation prediction: %s", name)
            raw = _predict(model, features, batch_size)
            probabilities = temperature.transform(raw) if temperature else raw
            ablations[name] = {
                "metrics": metric_set(partition.targets, probabilities),
                "full_minus_ablation": paired_clustered_delta(
                    partition.targets,
                    full,
                    probabilities,
                    partition.patient_ids,
                    bootstrap_iterations,
                    seed,
                ),
            }
        report["modality_ablation"] = ablations

    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2, default=_json_value), encoding="utf-8"
    )
    LOGGER.info(
        "EVALUATION COMPLETE | accuracy=%.4f balanced_accuracy=%.4f macro_f1=%.4f artifacts=%s",
        float(metrics["accuracy"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["macro_f1"]),
        output_dir,
    )
    LOGGER.info("=" * 72)
    return report, full_raw
