from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from file_utils import ensure_dir
from model_inputs import ModelInput, get_sample_count, slice_model_samples
from pipeline_config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def _target_score(model: tf.keras.Model, X: ModelInput, y_true: np.ndarray) -> np.ndarray:
    preds = model.predict(X, verbose=0)
    if preds.ndim == 1:
        preds = np.column_stack([1.0 - preds, preds])
    elif preds.ndim == 2 and preds.shape[1] == 1:
        preds = np.column_stack([1.0 - preds[:, 0], preds[:, 0]])
    return preds[np.arange(preds.shape[0]), y_true.astype(int)]


def multimodal_occlusion_and_perturbation(
    model: tf.keras.Model,
    X: ModelInput,
    y_true: np.ndarray,
    config: PipelineConfig,
    output_dir: str,
    max_samples: int = 24,
    occlusion_window: int = 3,
    spectral_noise_sigma: float = 0.10,
    perturbation_repeats: int = 5,
) -> None:
    if config.training_modality != 'both' or config.fusion_strategy != 'early':
        LOGGER.info('Multimodal occlusion/perturbation skipped because the model input does not contain both modalities.')
        return
    sample_count = min(max_samples, get_sample_count(X))
    if sample_count == 0:
        return

    ensure_dir(output_dir)
    indices = np.arange(sample_count)
    X_samples = slice_model_samples(X, indices)
    y_samples = y_true[indices].astype(np.int32)
    baseline_scores = _target_score(model, X_samples, y_samples)

    image_drop_records = []
    spectra_drop_records = []

    for local_index in range(sample_count):
        fused_patch = X_samples[local_index:local_index + 1].copy()
        height, width = fused_patch.shape[1:3]

        window_drops = []
        for row in range(0, height, occlusion_window):
            for col in range(0, width, occlusion_window):
                row_end = min(row + occlusion_window, height)
                col_end = min(col + occlusion_window, width)
                occluded_input = fused_patch.copy()
                occluded_input[:, row:row_end, col:col_end, :config.num_image_channels] = 0.0
                occluded_score = _target_score(model, occluded_input, np.asarray([y_samples[local_index]]))[0]
                window_drops.append(max(0.0, float(baseline_scores[local_index] - occluded_score)))
        image_drop_records.append(float(np.mean(window_drops)) if window_drops else 0.0)

        perturbation_drops = []
        for _ in range(perturbation_repeats):
            perturbed_input = fused_patch.copy()
            noise = np.random.normal(
                0.0,
                spectral_noise_sigma,
                size=perturbed_input[:, :, :, config.num_image_channels:].shape,
            ).astype(np.float32)
            perturbed_input[:, :, :, config.num_image_channels:] = np.clip(
                perturbed_input[:, :, :, config.num_image_channels:] + noise,
                0.0,
                1.0,
            )
            perturbed_score = _target_score(model, perturbed_input, np.asarray([y_samples[local_index]]))[0]
            perturbation_drops.append(max(0.0, float(baseline_scores[local_index] - perturbed_score)))
        spectra_drop_records.append(float(np.mean(perturbation_drops)) if perturbation_drops else 0.0)

    image_mean = float(np.mean(image_drop_records)) if image_drop_records else 0.0
    spectra_mean = float(np.mean(spectra_drop_records)) if spectra_drop_records else 0.0
    total = image_mean + spectra_mean
    image_pct = (image_mean / total * 100.0) if total > 0 else 0.0
    spectra_pct = (spectra_mean / total * 100.0) if total > 0 else 0.0

    summary_df = pd.DataFrame(
        {
            'sample_index': np.arange(sample_count),
            'baseline_score': baseline_scores,
            'image_occlusion_drop': image_drop_records,
            'spectra_perturbation_drop': spectra_drop_records,
        }
    )
    summary_df.to_csv(os.path.join(output_dir, 'multimodal_quantification_per_sample.csv'), index=False)

    pd.DataFrame(
        {
            'metric': ['image_mean_drop', 'spectra_mean_drop', 'image_pct', 'spectra_pct'],
            'value': [image_mean, spectra_mean, image_pct, spectra_pct],
        }
    ).to_csv(os.path.join(output_dir, 'multimodal_quantification_summary.csv'), index=False)

    plt.figure(figsize=(6, 4))
    plt.bar(['Image occlusion', 'Spectral perturbation'], [image_mean, spectra_mean])
    plt.ylabel('Mean target-score drop')
    plt.title('Multimodal contribution quantification')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'multimodal_quantification_bar.png'), dpi=200, bbox_inches='tight')
    plt.close()
