from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from pipeline_config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def _split_patch_modalities(patch: np.ndarray, config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray]:
    image_part = patch[..., :config.num_image_channels]
    spectra_part = patch[..., config.num_image_channels:]
    return image_part, spectra_part


def _recombine_patch_modalities(image_part: np.ndarray, spectra_part: np.ndarray) -> np.ndarray:
    return np.concatenate((image_part, spectra_part), axis=-1).astype(np.float32)


def _flip_horizontal(patch: np.ndarray, config: PipelineConfig) -> np.ndarray:
    image_part, spectra_part = _split_patch_modalities(patch, config)
    return _recombine_patch_modalities(image_part[:, ::-1, :], spectra_part[:, ::-1, :])


def _flip_vertical(patch: np.ndarray, config: PipelineConfig) -> np.ndarray:
    image_part, spectra_part = _split_patch_modalities(patch, config)
    return _recombine_patch_modalities(image_part[::-1, :, :], spectra_part[::-1, :, :])


def _rotate_90(patch: np.ndarray, config: PipelineConfig) -> np.ndarray:
    image_part, spectra_part = _split_patch_modalities(patch, config)
    return _recombine_patch_modalities(np.rot90(image_part, 1), np.rot90(spectra_part, 1))


def _rotate_180(patch: np.ndarray, config: PipelineConfig) -> np.ndarray:
    image_part, spectra_part = _split_patch_modalities(patch, config)
    return _recombine_patch_modalities(np.rot90(image_part, 2), np.rot90(spectra_part, 2))


def _rotate_270(patch: np.ndarray, config: PipelineConfig) -> np.ndarray:
    image_part, spectra_part = _split_patch_modalities(patch, config)
    return _recombine_patch_modalities(np.rot90(image_part, 3), np.rot90(spectra_part, 3))


def _adjust_brightness_contrast(
    patch: np.ndarray,
    config: PipelineConfig,
    brightness_range: Tuple[float, float] = (0.8, 1.2),
    contrast_range: Tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    image_part, spectra_part = _split_patch_modalities(patch, config)
    image_uint8 = (np.clip(image_part, 0, 1) * 255.0).astype(np.uint8)
    brightness_factor = float(np.random.uniform(*brightness_range))
    contrast_factor = float(np.random.uniform(*contrast_range))
    adjusted_uint8 = cv2.convertScaleAbs(image_uint8, alpha=contrast_factor, beta=(brightness_factor - 1.0) * 127.0)
    adjusted = adjusted_uint8.astype(np.float32) / 255.0
    return _recombine_patch_modalities(adjusted, spectra_part)


AUGMENTATION_FUNCTIONS = (
    _flip_horizontal,
    _flip_vertical,
    _rotate_90,
    _rotate_180,
    _rotate_270,
    _adjust_brightness_contrast,
)


def downsample_to_min_class(X: np.ndarray, y: np.ndarray, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return X, y
    rng = np.random.default_rng(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    min_count = int(np.min(counts))
    selected_indices = []
    for cls in unique_classes:
        class_indices = np.where(y == cls)[0]
        selected_indices.append(rng.choice(class_indices, size=min_count, replace=False))
    merged = np.concatenate(selected_indices)
    rng.shuffle(merged)
    return X[merged], y[merged]


def upsample_by_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0:
        return X, y
    rng = np.random.default_rng(seed)
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = int(np.max(counts))

    augmented_X = []
    augmented_y = []
    for cls, count in zip(unique_classes, counts):
        class_mask = y == cls
        class_samples = X[class_mask]
        augmented_X.append(class_samples)
        augmented_y.append(np.full(class_samples.shape[0], cls, dtype=np.int32))

        needed = max_count - int(count)
        if needed <= 0:
            continue

        chosen = rng.choice(class_samples.shape[0], size=needed, replace=True)
        synthetic = []
        for source_idx in chosen:
            patch = class_samples[int(source_idx)]
            augmentation = AUGMENTATION_FUNCTIONS[int(rng.integers(0, len(AUGMENTATION_FUNCTIONS)))]
            synthetic.append(augmentation(patch, config))
        synthetic_array = np.asarray(synthetic, dtype=np.float32)
        augmented_X.append(synthetic_array)
        augmented_y.append(np.full(synthetic_array.shape[0], cls, dtype=np.int32))

    X_balanced = np.concatenate(augmented_X, axis=0)
    y_balanced = np.concatenate(augmented_y, axis=0)
    permutation = rng.permutation(X_balanced.shape[0])
    return X_balanced[permutation], y_balanced[permutation]


def augment_training_set(
    X: np.ndarray,
    y: np.ndarray,
    config: PipelineConfig,
    seed: Optional[int] = None,
    augment_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if X.shape[0] == 0 or augment_factor <= 0:
        return X, y
    rng = np.random.default_rng(seed)
    augmented_batches = [X]
    augmented_labels = [y]
    for _ in range(augment_factor):
        aug_batch = np.empty_like(X, dtype=np.float32)
        for index, patch in enumerate(X):
            augmentation = AUGMENTATION_FUNCTIONS[int(rng.integers(0, len(AUGMENTATION_FUNCTIONS)))]
            aug_batch[index] = augmentation(patch, config)
        augmented_batches.append(aug_batch)
        augmented_labels.append(y.copy())
    X_aug = np.concatenate(augmented_batches, axis=0)
    y_aug = np.concatenate(augmented_labels, axis=0)
    permutation = rng.permutation(X_aug.shape[0])
    return X_aug[permutation], y_aug[permutation]


def apply_balancing(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: PipelineConfig,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if not config.balance_classes or X_train.shape[0] == 0:
        LOGGER.info('Skipping class balancing.')
        return X_train, y_train
    if config.balancing_strategy == 'downsample':
        return downsample_to_min_class(X_train, y_train, seed=seed)
    if config.balancing_strategy == 'upsample':
        return upsample_by_augmentation(X_train, y_train, config=config, seed=seed)
    if config.balancing_strategy == 'augment':
        return augment_training_set(X_train, y_train, config=config, seed=seed, augment_factor=1)
    raise ValueError(f'Unsupported balancing strategy: {config.balancing_strategy}')
