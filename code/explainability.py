
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

LOGGER = logging.getLogger(__name__)
MAX_EXPLAINED_SAMPLES = 40


def _normalise_map(values):
    values = np.asarray(values, dtype=np.float32)
    values -= values.min()
    maximum = values.max()
    return values / maximum if maximum > 0 else values


def gradcam(model, sample, class_index=None, layer_name="fusion_feature_map"):
    """Grad-CAM for one patch."""
    sample = np.asarray(sample, dtype=np.float32)
    batch = tf.convert_to_tensor(sample[None, ...])
    gradient_model = tf.keras.Model(
        model.input, [model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        feature_map, probabilities = gradient_model(batch, training=False)
        if class_index is None:
            class_index = int(tf.argmax(probabilities[0]).numpy())
        score = probabilities[0, int(class_index)]
    gradients = tape.gradient(score, feature_map)
    if gradients is None:
        raise RuntimeError(f"no gradient reached {layer_name}")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.nn.relu(tf.reduce_sum(feature_map * weights[:, None, None, :], axis=-1))
    heatmap = tf.image.resize(heatmap[..., None], sample.shape[:2], method="bilinear")[0, ..., 0]
    return _normalise_map(heatmap.numpy()), int(class_index), probabilities[0].numpy()


def integrated_gradients(model, sample, class_index=None, steps=32):

    sample = np.asarray(sample, dtype=np.float32)
    if steps < 2:
        raise ValueError("integrated-gradient steps must be at least two")
    baseline = np.zeros_like(sample)
    delta = sample - baseline
    if class_index is None:
        class_index = int(np.argmax(model(sample[None], training=False).numpy()[0]))
    gradients = []
    for alpha in np.linspace(0.0, 1.0, int(steps) + 1):
        point = tf.convert_to_tensor((baseline + alpha * delta)[None])
        with tf.GradientTape() as tape:
            tape.watch(point)
            score = model(point, training=False)[0, int(class_index)]
        gradient = tape.gradient(score, point)
        if gradient is None:
            raise RuntimeError("no input gradient was available")
        gradients.append(gradient[0].numpy())
    # Trapezoidal path integration is less biased than a right-endpoint sum.
    average = np.mean((np.asarray(gradients[:-1]) + np.asarray(gradients[1:])) / 2.0, axis=0)
    attribution = delta * average
    return attribution


def _base_image(sample, modality):
    if modality in {"image", "both"}:
        return np.clip(sample[..., :3], 0.0, 1.0)
    values = sample
    return _normalise_map(np.mean(values, axis=-1))


def _modality_slices(modality: str, channel_count: int):
    feature_count = int(channel_count)
    if modality == "image":
        return {"rgb": slice(0, 3)}
    if modality == "spectra":
        return {"spectra": slice(0, feature_count)}
    if modality == "both":
        if feature_count <= 3:
            raise ValueError("multimodal explainability requires RGB plus spectral channels")
        return {"rgb": slice(0, 3), "spectra": slice(3, feature_count)}
    raise ValueError(f"unsupported modality {modality!r}")


def _predict_probabilities(model, features, batch_size: int = 64):
    features = np.asarray(features, dtype=np.float32)
    rows = []
    for start in range(0, len(features), int(batch_size)):
        values = model(
            tf.convert_to_tensor(features[start : start + int(batch_size)]),
            training=False,
        )
        if isinstance(values, dict):
            values = values["fused"]
        rows.append(np.asarray(values, dtype=np.float64))
    probabilities = np.concatenate(rows, axis=0)
    if probabilities.ndim != 2 or not np.all(np.isfinite(probabilities)):
        raise ValueError("explainability received invalid model probabilities")
    return probabilities


def _grid_starts(length: int, window: int, stride: int):
    window = min(int(window), int(length))
    last = max(0, int(length) - window)
    starts = list(range(0, last + 1, int(stride)))
    if not starts or starts[-1] != last:
        starts.append(last)
    return starts, window


def spatial_occlusion_sensitivity(
    model,
    sample,
    class_index: int,
    channel_slice: slice,
    window_size: int = 5,
    stride: int = 3,
    baseline_probability: float | None = None,
):

    sample = np.asarray(sample, dtype=np.float32)
    if baseline_probability is None:
        baseline_probability = float(_predict_probabilities(model, sample[None])[0, class_index])
    row_starts, row_window = _grid_starts(sample.shape[0], window_size, stride)
    col_starts, col_window = _grid_starts(sample.shape[1], window_size, stride)
    variants, regions = [], []
    for row in row_starts:
        for column in col_starts:
            row_end = row + row_window
            column_end = column + col_window
            variant = sample.copy()
            variant[row:row_end, column:column_end, channel_slice] = 0.0
            variants.append(variant)
            regions.append((row, row_end, column, column_end))
    if not variants:
        return np.zeros(sample.shape[:2], dtype=np.float32)
    probabilities = _predict_probabilities(model, np.stack(variants))[:, int(class_index)]
    accumulated = np.zeros(sample.shape[:2], dtype=np.float64)
    coverage = np.zeros(sample.shape[:2], dtype=np.float64)
    for probability, (row, row_end, column, column_end) in zip(probabilities, regions):
        effect = float(baseline_probability - probability)
        accumulated[row:row_end, column:column_end] += effect
        coverage[row:row_end, column:column_end] += 1
    result = np.zeros(sample.shape[:2], dtype=np.float32)
    np.divide(accumulated, coverage, out=result, where=coverage > 0)
    return result


def modality_occlusion_and_perturbation(
    model,
    sample,
    modality: str,
    class_index: int,
    repeats: int,
    perturbation_scale: float,
    rng,
):

    sample = np.asarray(sample, dtype=np.float32)
    slices = _modality_slices(modality, sample.shape[-1])
    full_probabilities = _predict_probabilities(model, sample[None])[0]
    full_probability = float(full_probabilities[int(class_index)])
    rows = []
    for name, channel_slice in slices.items():
        occluded = sample.copy()
        occluded[..., channel_slice] = 0.0
        occluded_probabilities = _predict_probabilities(model, occluded[None])[0]

        observed = sample[..., channel_slice].reshape(-1, sample[..., channel_slice].shape[-1])
        channel_scale = np.std(observed, axis=0) if len(observed) else np.ones(observed.shape[-1])
        floor = 0.05 if name == "rgb" else 0.10
        channel_scale = np.maximum(channel_scale, floor)
        perturbations = []
        for _ in range(int(repeats)):
            variant = sample.copy()
            block = variant[..., channel_slice]
            noise = rng.normal(
                0.0,
                float(perturbation_scale) * channel_scale,
                size=block.shape,
            ).astype(np.float32)
            block += noise
            if name == "rgb":
                np.clip(block, 0.0, 1.0, out=block)
            variant[..., channel_slice] = block
            perturbations.append(variant)
        perturbed_probabilities = _predict_probabilities(model, np.stack(perturbations))
        selected_probabilities = perturbed_probabilities[:, int(class_index)]
        rows.append(
            {
                "modality": name,
                "full_probability": full_probability,
                "occluded_probability": float(occluded_probabilities[int(class_index)]),
                "occlusion_probability_drop": float(
                    full_probability - occluded_probabilities[int(class_index)]
                ),
                "relative_occlusion_drop": float(
                    (full_probability - occluded_probabilities[int(class_index)])
                    / max(full_probability, 1e-7)
                ),
                "occluded_prediction": int(np.argmax(occluded_probabilities)),
                "perturbation_mean_absolute_probability_change": float(
                    np.mean(np.abs(selected_probabilities - full_probability))
                ),
                "perturbation_mean_probability_drop": float(
                    np.mean(full_probability - selected_probabilities)
                ),
                "perturbation_probability_std": float(np.std(selected_probabilities)),
                "perturbation_class_switch_rate": float(
                    np.mean(np.argmax(perturbed_probabilities, axis=1) != int(class_index))
                ),
            }
        )

    positive_occlusion = np.maximum(
        [row["occlusion_probability_drop"] for row in rows], 0.0
    )
    perturbation = np.maximum(
        [row["perturbation_mean_absolute_probability_change"] for row in rows], 0.0
    )
    occlusion_total = float(np.sum(positive_occlusion))
    perturbation_total = float(np.sum(perturbation))
    for index, row in enumerate(rows):
        row["positive_occlusion_contribution"] = float(positive_occlusion[index])
        row["relative_occlusion_contribution"] = (
            float(positive_occlusion[index] / occlusion_total) if occlusion_total > 0 else 0.0
        )
        row["relative_perturbation_sensitivity"] = (
            float(perturbation[index] / perturbation_total) if perturbation_total > 0 else 0.0
        )
    return rows


def _stratified_explanation_indices(targets, maximum: int):

    targets = np.asarray(targets, dtype=int)
    maximum = min(int(maximum), len(targets), MAX_EXPLAINED_SAMPLES)
    classes = np.unique(targets)
    selected = []
    quota, remainder = divmod(maximum, max(1, len(classes)))
    for position, label in enumerate(classes):
        candidates = np.flatnonzero(targets == label)
        count = min(len(candidates), quota + (position < remainder))
        if count:
            selected.extend(candidates[np.linspace(0, len(candidates) - 1, count, dtype=int)])
    if len(selected) < maximum:
        remaining = np.setdiff1d(np.arange(len(targets)), np.asarray(selected, dtype=int))
        count = maximum - len(selected)
        selected.extend(remaining[np.linspace(0, len(remaining) - 1, count, dtype=int)])
    return np.asarray(sorted(set(map(int, selected))), dtype=int)


def _add_heatmap(figure, axis, values, title, cmap="magma", signed=False):
    values = np.asarray(values)
    if signed:
        limit = max(float(np.max(np.abs(values))), 1e-7)
        image = axis.imshow(values, cmap=cmap, vmin=-limit, vmax=limit)
        label = "Signed probability drop"
    else:
        image = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=1.0)
        label = "Attribution intensity"
    axis.set_title(title)
    axis.axis("off")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(label, fontsize=8)


def _save_explanation_figure(
    sample,
    modality,
    heatmap,
    occlusion_maps,
    sensitivity_rows,
    target,
    predicted,
    confidence,
    path,
    class_names=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    modality_names = list(occlusion_maps)
    if modality not in {"image", "both"}:
        # Spectra-only interpretation is primarily provided by the dedicated
        # SHAP violin/beeswarm.
        base = _base_image(sample, modality)
        figure, axes = plt.subplots(1, 4, figsize=(18, 4.5))
        axes[0].imshow(base, cmap=None if base.ndim == 3 else "viridis")
        axes[0].set_title("Input patch")
        axes[0].axis("off")
        _add_heatmap(figure, axes[1], heatmap, "Grad-CAM")
        _add_heatmap(
            figure,
            axes[2],
            occlusion_maps[modality_names[0]],
            f"{modality_names[0].upper()} spatial occlusion",
            cmap="coolwarm",
            signed=True,
        )
        impact_axis = axes[3]
        labels = [row["modality"].upper() for row in sensitivity_rows]
        locations = np.arange(len(labels))
        impact_axis.bar(
            locations - 0.18,
            [row["occlusion_probability_drop"] for row in sensitivity_rows],
            0.36,
            label="Global occlusion Δp",
            color="#D55E00",
        )
        impact_axis.bar(
            locations + 0.18,
            [row["perturbation_mean_absolute_probability_change"] for row in sensitivity_rows],
            0.36,
            label="Perturbation mean |Δp|",
            color="#0072B2",
        )
        impact_axis.axhline(0.0, color="black", linewidth=0.8)
        impact_axis.set_xticks(locations, labels)
        impact_axis.set_ylabel("Predicted-class probability change")
        impact_axis.set_title("Modality contribution and sensitivity")
        impact_axis.grid(axis="y", alpha=0.25)
        impact_axis.legend(fontsize=8)
        figure.suptitle(
            f"True class {target} | predicted class {predicted} | confidence {confidence:.4f}",
            fontsize=13,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.94))
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        return


    figure = plt.figure(figsize=(20, 11), facecolor="white")
    grid = figure.add_gridspec(
        2,
        6,
        height_ratios=(1.0, 1.02),
        hspace=0.34,
        wspace=0.72,
    )
    patch_axis = figure.add_subplot(grid[0, 0:2])
    cam_axis = figure.add_subplot(grid[0, 2:4])
    overlay_axis = figure.add_subplot(grid[0, 4:6])
    occlusion_axis = figure.add_subplot(grid[1, 0:3])
    perturbation_axis = figure.add_subplot(grid[1, 3:6])

    rgb = np.clip(sample[..., :3], 0.0, 1.0).copy()
    patch_axis.imshow(rgb, interpolation="nearest")
    patch_axis.set_title("(A) Co-registered H&E", fontsize=15, pad=12)
    patch_axis.axis("off")

    cam_artist = cam_axis.imshow(
        heatmap,
        cmap="jet",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    cam_axis.set_title("(B) Grad-CAM", fontsize=15, pad=12)
    cam_axis.axis("off")
    cam_bar = figure.colorbar(cam_artist, ax=cam_axis, fraction=0.046, pad=0.04)
    cam_bar.set_label("Normalised activation", fontsize=12, labelpad=10)
    cam_bar.set_ticks([0.0, 0.5, 1.0])

    overlay_axis.imshow(rgb, interpolation="nearest")
    overlay_artist = overlay_axis.imshow(
        heatmap,
        cmap="jet",
        alpha=0.48,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    overlay_axis.set_title("(C) Overlay", fontsize=15, pad=12)
    overlay_axis.axis("off")
    overlay_bar = figure.colorbar(overlay_artist, ax=overlay_axis, fraction=0.046, pad=0.04)
    overlay_bar.set_label("Normalised activation", fontsize=12, labelpad=10)
    overlay_bar.set_ticks([0.0, 0.5, 1.0])

    display_names = {"rgb": "Image", "spectra": "Spectra"}
    labels = [display_names.get(row["modality"], row["modality"].title()) for row in sensitivity_rows]
    colors = ["#5B7FDE" if row["modality"] == "rgb" else "#E7355F" for row in sensitivity_rows]
    locations = np.arange(len(labels), dtype=float)

    def add_percentage_bars(axis, values, title):
        values = np.asarray(values, dtype=float)
        bars = axis.bar(locations, values, width=0.56, color=colors)
        axis.set_xticks(locations, labels)
        axis.set_ylim(0.0, 1.15)
        axis.set_ylabel("Relative contribution / sensitivity", fontsize=12)
        axis.set_title(title, fontsize=15, pad=12)
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.35)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2.0,
                max(float(value), 0.0) + 0.025,
                f"{100.0 * value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    add_percentage_bars(
        occlusion_axis,
        [row["relative_occlusion_contribution"] for row in sensitivity_rows],
        "(D) Occlusion importance",
    )
    add_percentage_bars(
        perturbation_axis,
        [row["relative_perturbation_sensitivity"] for row in sensitivity_rows],
        "(E) Perturbation sensitivity",
    )

    def class_label(index):
        if class_names is not None and 0 <= int(index) < len(class_names):
            return str(class_names[int(index)])
        return str(int(index))

    figure.suptitle(
        f"{modality.title()} | True: {class_label(target)} | "
        f"Pred: {class_label(predicted)} | p={confidence:.3f}",
        fontsize=16,
        y=0.985,
    )
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def run_explainability(
    model,
    partition,
    output_dir: Path,
    modality: str,
    feature_names,
    max_samples: int = 40,
    integrated_gradient_steps: int = 32,
    occlusion_window: int = 5,
    occlusion_stride: int = 3,
    perturbation_repeats: int = 16,
    perturbation_scale: float = 0.10,
    seed: int = 42,
    class_names=None,
):
    """Explain up to 40 deterministic class-stratified held-out patches."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if int(max_samples) > MAX_EXPLAINED_SAMPLES:
        LOGGER.warning(
            "Requested %d explanations; applying the protocol maximum of %d",
            max_samples,
            MAX_EXPLAINED_SAMPLES,
        )
    selected = _stratified_explanation_indices(partition.targets, max_samples)
    LOGGER.info(
        "EXPLAINABILITY START | samples=%d modality=%s IG_steps=%d occlusion=%dx%d/stride%d perturbations=%d scale=%.3f output=%s",
        len(selected),
        modality,
        integrated_gradient_steps,
        occlusion_window,
        occlusion_window,
        occlusion_stride,
        perturbation_repeats,
        perturbation_scale,
        output_dir,
    )
    if modality == "both":
        names = ["RGB_R", "RGB_G", "RGB_B"] + list(feature_names)
    elif modality == "image":
        names = ["RGB_R", "RGB_G", "RGB_B"]
    else:
        names = list(feature_names)
    rows, channel_rows, sensitivity_rows = [], [], []
    modality_slices = _modality_slices(modality, partition.features.shape[-1])
    rng = np.random.default_rng(seed)
    for position, index in enumerate(selected, start=1):
        LOGGER.info(
            "[Explanation %d/%d] patch=%s patient=%s",
            position,
            len(selected),
            partition.patch_ids[index],
            partition.patient_ids[index],
        )
        sample = partition.features[index]
        heatmap, predicted, probabilities = gradcam(model, sample)
        confidence = float(probabilities[predicted])
        attributions = integrated_gradients(
            model, sample, predicted, steps=integrated_gradient_steps
        )
        channel_signed = np.mean(attributions, axis=(0, 1))
        channel_absolute = np.mean(np.abs(attributions), axis=(0, 1))
        for name, signed, absolute in zip(names, channel_signed, channel_absolute):
            channel_rows.append(
                {
                    "patch_uid": partition.patch_ids[index],
                    "patient_id": partition.patient_ids[index],
                    "feature": name,
                    "signed_attribution": float(signed),
                    "absolute_attribution": float(absolute),
                }
            )
        rows.append(
            {
                "patch_uid": partition.patch_ids[index],
                "patient_id": partition.patient_ids[index],
                "filename": partition.filenames[index],
                "target": int(partition.targets[index]),
                "prediction": predicted,
                "confidence": confidence,
            }
        )
        sample_sensitivity = modality_occlusion_and_perturbation(
            model,
            sample,
            modality,
            predicted,
            perturbation_repeats,
            perturbation_scale,
            rng,
        )
        for sensitivity in sample_sensitivity:
            sensitivity_rows.append(
                {
                    "patch_uid": partition.patch_ids[index],
                    "patient_id": partition.patient_ids[index],
                    "filename": partition.filenames[index],
                    "target": int(partition.targets[index]),
                    "prediction": predicted,
                    **sensitivity,
                }
            )
        occlusion_maps = {}
        for modality_name, channel_slice in modality_slices.items():
            LOGGER.info(
                "[Explanation %d/%d] computing %s spatial occlusion map",
                position,
                len(selected),
                modality_name,
            )
            occlusion_maps[modality_name] = spatial_occlusion_sensitivity(
                model,
                sample,
                predicted,
                channel_slice,
                window_size=occlusion_window,
                stride=occlusion_stride,
                baseline_probability=confidence,
            )
        _save_explanation_figure(
            sample,
            modality,
            heatmap,
            occlusion_maps,
            sample_sensitivity,
            int(partition.targets[index]),
            predicted,
            confidence,
            output_dir / f"patch_{partition.patch_ids[index]}.png",
            class_names=class_names,
        )
        attribution_payload = {
            "gradcam": heatmap,
            "integrated_gradients": attributions,
            "probabilities": probabilities,
        }
        attribution_payload.update(
            {f"{name}_spatial_occlusion": values for name, values in occlusion_maps.items()}
        )
        np.savez_compressed(
            output_dir / f"patch_{partition.patch_ids[index]}_attribution.npz",
            **attribution_payload,
        )
        LOGGER.info(
            "[Explanation %d/%d] complete | predicted=%d confidence=%.4f",
            position,
            len(selected),
            predicted,
            confidence,
        )
    pd.DataFrame(rows).to_csv(output_dir / "explained_samples.csv", index=False)
    channel_frame = pd.DataFrame(channel_rows)
    channel_frame.to_csv(output_dir / "sample_channel_attributions.csv", index=False)
    summary = (
        channel_frame.groupby("feature", as_index=False)
        .agg(
            mean_signed_attribution=("signed_attribution", "mean"),
            mean_absolute_attribution=("absolute_attribution", "mean"),
        )
        .sort_values("mean_absolute_attribution", ascending=False)
    )
    summary.to_csv(output_dir / "channel_attribution_summary.csv", index=False)
    sensitivity_frame = pd.DataFrame(sensitivity_rows)
    sensitivity_frame.to_csv(output_dir / "modality_sensitivity.csv", index=False)
    sensitivity_summary = (
        sensitivity_frame.groupby("modality", as_index=False)
        .agg(
            samples=("patch_uid", "count"),
            mean_occlusion_probability_drop=("occlusion_probability_drop", "mean"),
            median_occlusion_probability_drop=("occlusion_probability_drop", "median"),
            mean_relative_occlusion_drop=("relative_occlusion_drop", "mean"),
            mean_relative_occlusion_contribution=(
                "relative_occlusion_contribution",
                "mean",
            ),
            mean_perturbation_absolute_change=(
                "perturbation_mean_absolute_probability_change",
                "mean",
            ),
            mean_relative_perturbation_sensitivity=(
                "relative_perturbation_sensitivity",
                "mean",
            ),
            mean_perturbation_class_switch_rate=("perturbation_class_switch_rate", "mean"),
        )
    )
    sensitivity_summary.to_csv(output_dir / "modality_sensitivity_summary.csv", index=False)
    protocol = {
        "sample_selection": "deterministic class-stratified coverage; independent of predictions",
        "maximum_samples_per_seed": MAX_EXPLAINED_SAMPLES,
        "explained_samples": int(len(selected)),
        "gradcam_layer": "fusion_feature_map",
        "occlusion_reference": "zeroed modality",
        "occlusion_window": int(occlusion_window),
        "occlusion_stride": int(occlusion_stride),
        "perturbation_repeats": int(perturbation_repeats),
        "perturbation_scale": float(perturbation_scale),
        "perturbation_definition": (
            "Gaussian noise at perturbation_scale times within-patch per-channel standard "
            "deviation; minimum scale 0.05 RGB and 0.10 spectra"
        ),
        "explained_target": "the model-predicted class for each patch",
        "figure_layout": (
            "image/both: (A) co-registered H&E, (B) Grad-CAM, (C) overlay, "
            "(D) relative positive occlusion contribution, (E) relative perturbation sensitivity"
        ),
        "relative_occlusion_definition": (
            "positive predicted-class probability drops normalised across active modalities; "
            "raw signed drops remain in modality_sensitivity.csv"
        ),
        "relative_perturbation_definition": (
            "mean absolute predicted-class probability changes normalised across active modalities"
        ),
    }
    (output_dir / "explainability_protocol.json").write_text(
        json.dumps(protocol, indent=2), encoding="utf-8"
    )
    LOGGER.info("EXPLAINABILITY COMPLETE | artifacts=%s", output_dir)
    return summary
