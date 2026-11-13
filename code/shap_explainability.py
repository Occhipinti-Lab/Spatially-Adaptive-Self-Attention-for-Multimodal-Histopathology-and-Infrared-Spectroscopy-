"""Spectra-only SHAP with train-only background and ROI-level feature impacts."""

from __future__ import annotations

import json
import logging
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _balanced_indices(targets, maximum: int, seed: int):
    targets = np.asarray(targets, dtype=int)
    if maximum >= len(targets):
        return np.arange(len(targets))
    rng = np.random.default_rng(seed)
    classes = np.unique(targets)
    allocation = maximum // len(classes)
    remainder = maximum % len(classes)
    selected = []
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(targets == label)
        count = min(len(candidates), allocation + (position < remainder))
        selected.extend(rng.choice(candidates, count, replace=False).tolist())
    if len(selected) < maximum:
        available = np.setdiff1d(np.arange(len(targets)), selected)
        selected.extend(rng.choice(available, maximum - len(selected), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _normalise_output(raw_values, input_shape, class_indices, output_count):

    if hasattr(raw_values, "values"):
        raw_values = raw_values.values
    if isinstance(raw_values, (list, tuple)):
        values = np.stack([np.asarray(value) for value in raw_values], axis=-1)
    else:
        values = np.asarray(raw_values)
    expected = tuple(input_shape)
    if values.shape == expected:
        return values
    if values.shape == expected + (output_count,):
        gather_shape = (len(class_indices),) + (1,) * (len(expected) - 1) + (1,)
        indices = np.asarray(class_indices, dtype=int).reshape(gather_shape)
        return np.take_along_axis(values, indices, axis=-1)[..., 0]
    if values.shape == (output_count,) + expected:
        values = np.moveaxis(values, 0, -1)
        gather_shape = (len(class_indices),) + (1,) * (len(expected) - 1) + (1,)
        indices = np.asarray(class_indices, dtype=int).reshape(gather_shape)
        return np.take_along_axis(values, indices, axis=-1)[..., 0]
    raise ValueError(
        f"unsupported SHAP output shape {values.shape}; expected {expected} plus "
        f"an optional {output_count}-class axis"
    )


def _display_feature_name(name):

    name = str(name)
    if name.startswith("WN_"):
        try:
            return f"WN_{float(name[3:]):g}"
        except ValueError:
            pass
    return name


def _save_summary_plot(shap, impacts, feature_values, feature_names, top_features, kind, path):
    import matplotlib.pyplot as plt

    maximum = min(10, int(top_features), impacts.shape[1])
    display_names = [_display_feature_name(name) for name in feature_names]
    figure = plt.figure(figsize=(13.0, max(6.0, 0.72 * maximum + 1.8)))
    shap.summary_plot(
        impacts,
        features=feature_values,
        feature_names=display_names,
        max_display=maximum,
        plot_type=kind,
        show=False,
        color_bar=True,
        color_bar_label="Feature value",
        plot_size=None,
        rng=np.random.default_rng(0),
    )
    figure = plt.gcf()
    figure.set_facecolor("white")
    if figure.axes:
        impact_axis = figure.axes[0]
        impact_axis.set_facecolor("white")
        impact_axis.set_xlabel(
            "SHAP value (impact on model output)", fontsize=14, labelpad=8
        )
        impact_axis.tick_params(axis="x", labelsize=11)
        impact_axis.tick_params(axis="y", labelsize=13, pad=8)
        impact_axis.grid(axis="y", linestyle=(0, (1, 4)), linewidth=0.8, alpha=0.35)
        impact_axis.set_axisbelow(True)
    if len(figure.axes) > 1:
        colour_axis = figure.axes[-1]
        colour_axis.set_ylabel("Feature value", fontsize=13, labelpad=10)
        colour_axis.tick_params(labelsize=11)
    figure.subplots_adjust(left=0.23, right=0.90, bottom=0.13, top=0.98)
    figure.savefig(path, dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_spectra_shap(
    model,
    train_partition,
    explained_partition,
    feature_names,
    output_dir: Path,
    max_samples: int = 20,
    background_size: int = 32,
    nsamples: int = 100,
    top_features: int = 10,
    seed: int = 42,
):
    """Create top-feature violin and beeswarm plots for a spectra-only model.

    SHAP evaluates the complete spectral input and ranks spectral features.
    """
    import shap

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spectral_count = explained_partition.features.shape[-1]
    if spectral_count <= 0:
        raise ValueError("spectra SHAP requires spectral channels")
    if len(feature_names) != spectral_count:
        raise ValueError(
            f"received {len(feature_names)} names for {spectral_count} spectral features"
        )
    background_indices = _balanced_indices(
        train_partition.targets, min(background_size, len(train_partition.targets)), seed
    )
    explained_indices = _balanced_indices(
        explained_partition.targets,
        min(max_samples, len(explained_partition.targets)),
        seed + 1,
    )
    background = train_partition.features[background_indices].astype(np.float32, copy=False)
    explained = explained_partition.features[explained_indices].astype(np.float32, copy=False)
    LOGGER.info(
        "SHAP START | background=%d explained=%d spectral_features=%d nsamples=%d output=%s",
        len(background),
        len(explained),
        spectral_count,
        nsamples,
        output_dir,
    )
    LOGGER.info("SHAP phase 1/4: predicting explained patches")
    probabilities = np.asarray(model.predict(explained, verbose=0), dtype=float)
    predicted_classes = probabilities.argmax(axis=1)
    LOGGER.info("SHAP phase 2/4: constructing train-background GradientExplainer")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The structure of `inputs` doesn't match the expected structure.*",
            category=UserWarning,
        )
        explainer = shap.GradientExplainer(model, background)
        LOGGER.info("SHAP phase 3/4: estimating feature impacts (this is the longest SHAP step)")
        raw_values = explainer.shap_values(
            explained,
            nsamples=int(nsamples),
            rseed=int(seed),
        )
    sample_values = _normalise_output(
        raw_values,
        explained.shape,
        predicted_classes,
        probabilities.shape[1],
    )
    roi_mean_impacts = sample_values.mean(axis=(1, 2))
    roi_mean_features = explained.mean(axis=(1, 2))
    mean_absolute = np.mean(np.abs(roi_mean_impacts), axis=0)
    order = np.argsort(-mean_absolute, kind="stable")
    ranking = pd.DataFrame(
        {
            "rank": np.arange(1, spectral_count + 1),
            "feature": np.asarray(feature_names)[order],
            "mean_absolute_shap": mean_absolute[order],
            "mean_signed_shap": np.mean(roi_mean_impacts, axis=0)[order],
        }
    )
    ranking.to_csv(output_dir / "shap_feature_ranking.csv", index=False)
    LOGGER.info(
        "SHAP phase 4/4: saving rankings, patch impacts, layered-violin beeswarm, "
        "and conventional dot beeswarm"
    )
    long_rows = []
    for row, partition_index in enumerate(explained_indices):
        for feature_index, name in enumerate(feature_names):
            long_rows.append(
                {
                    "patch_uid": explained_partition.patch_ids[partition_index],
                    "patient_id": explained_partition.patient_ids[partition_index],
                    "target": int(explained_partition.targets[partition_index]),
                    "predicted_class": int(predicted_classes[row]),
                    "feature": name,
                    "roi_mean_feature_value": float(roi_mean_features[row, feature_index]),
                    "roi_mean_shap_impact": float(roi_mean_impacts[row, feature_index]),
                }
            )
    pd.DataFrame(long_rows).to_csv(output_dir / "shap_patch_feature_impacts.csv", index=False)
    _save_summary_plot(
        shap,
        roi_mean_impacts,
        roi_mean_features,
        feature_names,
        top_features,
        "violin",
        output_dir / "shap_top10_beeswarm.png",
    )

    shutil.copy2(
        output_dir / "shap_top10_beeswarm.png",
        output_dir / "shap_top10_violin.png",
    )
    shutil.copy2(
        output_dir / "shap_top10_beeswarm.pdf",
        output_dir / "shap_top10_violin.pdf",
    )
    _save_summary_plot(
        shap,
        roi_mean_impacts,
        roi_mean_features,
        feature_names,
        top_features,
        "dot",
        output_dir / "shap_top10_dot_beeswarm.png",
    )
    metadata = {
        "explainer": "shap.GradientExplainer",
        "background_scope": "training patches only",
        "background_samples": int(len(background)),
        "explained_samples": int(len(explained)),
        "monte_carlo_samples": int(nsamples),
        "ranking_statistic": "mean absolute ROI-mean SHAP impact",
        "top_features": int(min(10, top_features, spectral_count)),
        "predicted_class_explained_per_patch": True,
        "plot_encoding": {
            "x": "signed SHAP impact on the model output",
            "y": "top features ranked by mean absolute SHAP impact",
            "colour": "ROI-mean feature value (low to high)",
            "primary_beeswarm": (
                "layered violin/density distribution across explained held-out patches"
            ),
            "dot_beeswarm": "one jittered point per explained held-out patch",
        },
        "primary_plot": "shap_top10_beeswarm.png",
        "plot_files": [
            "shap_top10_beeswarm.png",
            "shap_top10_beeswarm.pdf",
            "shap_top10_dot_beeswarm.png",
            "shap_top10_dot_beeswarm.pdf",
            "shap_top10_violin.png",
            "shap_top10_violin.pdf",
        ],
    }
    (output_dir / "shap_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "SHAP COMPLETE | top_features=%s artifacts=%s",
        ranking.head(min(10, top_features, len(ranking)))["feature"].tolist(),
        output_dir,
    )
    return ranking
