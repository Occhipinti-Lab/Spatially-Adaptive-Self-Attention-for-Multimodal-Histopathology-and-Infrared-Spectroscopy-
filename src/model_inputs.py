from __future__ import annotations

from typing import Tuple

import numpy as np


ModelInput = np.ndarray
ModelInputShape = Tuple[int, int, int]


def get_sample_count(X: ModelInput) -> int:
    return int(X.shape[0])


def slice_model_samples(X: ModelInput, indices: np.ndarray) -> ModelInput:
    return X[indices]


def prepare_model_inputs(
    X: np.ndarray,
    num_image_channels: int,
    num_spectra_channels: int,
    training_modality: str,
    fusion_strategy: str,
    patch_size: int,
) -> Tuple[ModelInput, ModelInputShape]:
    if training_modality == 'image':
        return X[..., :num_image_channels], (patch_size, patch_size, num_image_channels)
    if training_modality == 'spectra':
        return X[..., num_image_channels:], (patch_size, patch_size, num_spectra_channels)
    if training_modality != 'both':
        raise ValueError(f'Unsupported training_modality: {training_modality}')

    if fusion_strategy == 'early':
        return X, (patch_size, patch_size, num_image_channels + num_spectra_channels)
    if fusion_strategy == 'image_only':
        return X[..., :num_image_channels], (patch_size, patch_size, num_image_channels)
    if fusion_strategy == 'spectra_only':
        return X[..., num_image_channels:], (patch_size, patch_size, num_spectra_channels)
    raise ValueError(f'Unsupported fusion_strategy: {fusion_strategy}')
