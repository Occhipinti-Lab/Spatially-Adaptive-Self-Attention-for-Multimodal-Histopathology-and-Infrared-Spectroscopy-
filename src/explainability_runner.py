from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import tensorflow as tf

from file_utils import ensure_dir
from gradcam_explainability import generate_gradcam_heatmaps
from modality_importance import multimodal_occlusion_and_perturbation
from model_inputs import ModelInput
from pipeline_config import PipelineConfig
from shap_explainability import spectra_shap_summary


def run_explainability_suite(
    model: tf.keras.Model,
    X_test: ModelInput,
    y_test: np.ndarray,
    config: PipelineConfig,
    feature_names: Sequence[str],
    output_dir: str,
) -> None:
    ensure_dir(output_dir)
    if config.gradcam_enabled:
        generate_gradcam_heatmaps(
            model=model,
            X=X_test,
            y_true=y_test,
            config=config,
            output_dir=os.path.join(output_dir, 'gradcam'),
            max_samples=config.explain_num_samples,
        )
    if config.shap_enabled and (config.training_modality == 'spectra' or config.fusion_strategy == 'spectra_only'):
        spectra_shap_summary(
            model=model,
            X=X_test,
            y_true=y_test,
            feature_names=feature_names,
            output_dir=os.path.join(output_dir, 'spectra_shap'),
        )
    if config.multimodal_quantification_enabled:
        multimodal_occlusion_and_perturbation(
            model=model,
            X=X_test,
            y_true=y_test,
            config=config,
            output_dir=os.path.join(output_dir, 'multimodal_quantification'),
            max_samples=config.explain_num_samples,
        )
