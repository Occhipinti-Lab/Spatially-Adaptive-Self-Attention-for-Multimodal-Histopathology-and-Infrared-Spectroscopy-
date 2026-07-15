from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from dataset_split import safe_train_val_test_split_indices
from file_utils import ensure_dir
from patch_extraction import extract_non_overlapping_pure_patches_per_file
from pipeline_config import PipelineConfig
from spectral_file_loader import load_and_process_file
from spectral_transform import (
    fit_spectral_transform_bundle,
    save_spectral_transform_bundle,
    transform_patches_with_bundle,
)


LOGGER = logging.getLogger(__name__)


def _concat_or_empty(array_list: List[np.ndarray], dtype: np.dtype, shape_tail: tuple[int, ...]) -> np.ndarray:
    if array_list:
        return np.concatenate(array_list, axis=0)
    return np.empty((0, *shape_tail), dtype=dtype)


def _concat_label_or_empty(array_list: List[np.ndarray]) -> np.ndarray:
    if array_list:
        return np.concatenate(array_list, axis=0).astype(np.int32)
    return np.empty((0,), dtype=np.int32)


def _append_split(
    target_patches: List[np.ndarray],
    target_labels: List[np.ndarray],
    target_masks: List[np.ndarray],
    X_file: np.ndarray,
    y_file: np.ndarray,
    masks_file: np.ndarray,
    indices: np.ndarray,
) -> None:
    target_patches.append(X_file[indices])
    target_labels.append(y_file[indices])
    target_masks.append(masks_file[indices])


def create_patch_pool_and_split(config: PipelineConfig, seed: Optional[int] = None) -> Dict[str, object]:
    rng_seed = seed if seed is not None else config.seed_list[0]
    csv_files = sorted(
        [
            os.path.join(config.data_dir, filename)
            for filename in os.listdir(config.data_dir)
            if filename.lower().endswith('.csv')
        ]
    )
    if not csv_files:
        raise FileNotFoundError(f'No CSV files were found in {config.data_dir}')

    train_patches_raw_list: List[np.ndarray] = []
    train_labels_list: List[np.ndarray] = []
    train_masks_list: List[np.ndarray] = []
    val_patches_raw_list: List[np.ndarray] = []
    val_labels_list: List[np.ndarray] = []
    val_masks_list: List[np.ndarray] = []
    test_patches_raw_list: List[np.ndarray] = []
    test_labels_list: List[np.ndarray] = []
    test_masks_list: List[np.ndarray] = []

    filtered_wavenumbers_reference: Optional[np.ndarray] = None
    raw_feature_names_reference: Optional[List[str]] = None

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            filtered_spectral, filtered_wavenumbers, labels, coords, image, raw_feature_names = load_and_process_file(
                filepath,
                config,
            )
            valid_mask = labels != -1
            filtered_spectral = filtered_spectral[valid_mask]
            labels = labels[valid_mask]
            coords = coords[valid_mask]

            if filtered_wavenumbers_reference is None:
                filtered_wavenumbers_reference = filtered_wavenumbers.copy()
                raw_feature_names_reference = list(raw_feature_names)
            elif not np.allclose(filtered_wavenumbers_reference, filtered_wavenumbers):
                raise ValueError(f'Filtered wavenumber grid mismatch detected in {filename}.')

            patches, patch_labels, _, _, patch_masks = extract_non_overlapping_pure_patches_per_file(
                processed_spectral_data=filtered_spectral,
                labels=labels,
                coords=coords,
                image=image,
                filename=filename,
                feature_names=raw_feature_names,
                config=config,
            )

            if not patches:
                LOGGER.warning('No patches were extracted from %s. Skipping file.', filename)
                continue

            X_file = np.asarray(patches, dtype=np.float32)
            y_file = np.asarray(patch_labels, dtype=np.int32)
            masks_file = np.asarray(patch_masks, dtype=bool)

            if config.reserved_files and filename in config.reserved_files:
                test_patches_raw_list.append(X_file)
                test_labels_list.append(y_file)
                test_masks_list.append(masks_file)
                LOGGER.info('Reserved file %s sent entirely to the test set with %d patches.', filename, y_file.shape[0])
                continue

            train_idx, val_idx, test_idx = safe_train_val_test_split_indices(y_file, rng_seed)
            _append_split(train_patches_raw_list, train_labels_list, train_masks_list, X_file, y_file, masks_file, train_idx)
            _append_split(val_patches_raw_list, val_labels_list, val_masks_list, X_file, y_file, masks_file, val_idx)
            _append_split(test_patches_raw_list, test_labels_list, test_masks_list, X_file, y_file, masks_file, test_idx)
            LOGGER.info(
                'Split %s into train=%d, validation=%d, test=%d patches.',
                filename,
                train_idx.shape[0],
                val_idx.shape[0],
                test_idx.shape[0],
            )
        except Exception as exc:
            LOGGER.exception('Failed to process %s: %s', filename, exc)

    if filtered_wavenumbers_reference is None or raw_feature_names_reference is None:
        raise ValueError('No files were successfully processed.')

    raw_channels = config.num_image_channels + filtered_wavenumbers_reference.shape[0]
    raw_patch_shape = (config.patch_size, config.patch_size, raw_channels)
    mask_shape = (config.patch_size, config.patch_size)
    train_patches_raw = _concat_or_empty(train_patches_raw_list, np.float32, raw_patch_shape)
    train_labels = _concat_label_or_empty(train_labels_list)
    train_masks = _concat_or_empty(train_masks_list, bool, mask_shape).astype(bool)
    val_patches_raw = _concat_or_empty(val_patches_raw_list, np.float32, raw_patch_shape)
    val_labels = _concat_label_or_empty(val_labels_list)
    val_masks = _concat_or_empty(val_masks_list, bool, mask_shape).astype(bool)
    test_patches_raw = _concat_or_empty(test_patches_raw_list, np.float32, raw_patch_shape)
    test_labels = _concat_label_or_empty(test_labels_list)
    test_masks = _concat_or_empty(test_masks_list, bool, mask_shape).astype(bool)

    if train_patches_raw.shape[0] == 0:
        raise ValueError('Training set is empty after patch extraction and splitting.')

    bundle = fit_spectral_transform_bundle(
        train_patches_raw=train_patches_raw,
        train_masks=train_masks,
        filtered_wavenumbers=filtered_wavenumbers_reference,
        raw_feature_names=raw_feature_names_reference,
        config=config,
    )

    transformed_channels = config.num_image_channels + len(bundle.output_feature_names)
    transformed_patch_shape = (0, config.patch_size, config.patch_size, transformed_channels)
    X_train = transform_patches_with_bundle(train_patches_raw, train_masks, bundle, config)
    X_val = (
        transform_patches_with_bundle(val_patches_raw, val_masks, bundle, config)
        if val_patches_raw.shape[0] > 0
        else np.empty(transformed_patch_shape, dtype=np.float32)
    )
    X_test = (
        transform_patches_with_bundle(test_patches_raw, test_masks, bundle, config)
        if test_patches_raw.shape[0] > 0
        else np.empty(transformed_patch_shape, dtype=np.float32)
    )

    preprocessing_dir = os.path.join(config.output_root_dir, 'preprocessing')
    ensure_dir(preprocessing_dir)
    if config.save_preprocessing_bundle:
        save_spectral_transform_bundle(bundle, os.path.join(preprocessing_dir, 'spectral_transform_bundle.pkl'))

    LOGGER.info(
        'Combined transformed dataset sizes -> train=%d, validation=%d, test=%d',
        X_train.shape[0],
        X_val.shape[0],
        X_test.shape[0],
    )

    return {
        'train': (X_train, train_labels),
        'validation': (X_val, val_labels),
        'test': (X_test, test_labels),
        'feature_names': bundle.output_feature_names,
        'spectral_bundle': bundle,
        'filtered_wavenumbers': filtered_wavenumbers_reference,
    }
