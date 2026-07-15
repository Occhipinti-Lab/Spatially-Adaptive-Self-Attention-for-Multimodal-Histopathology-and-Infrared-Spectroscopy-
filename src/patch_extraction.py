
from __future__ import annotations

import logging
import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from pipeline_config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def save_patch_to_csv(
    patch: np.ndarray,
    valid_mask: np.ndarray,
    label: int,
    patch_index: int,
    output_dir: str,
    feature_names: Sequence[str],
    start_row: int,
    start_col: int,
    config: PipelineConfig,
) -> str:
    os.makedirs(output_dir, exist_ok=True)

    patch_height, patch_width, total_channels = patch.shape
    if total_channels < config.num_image_channels:
        raise ValueError('Patch does not contain the required image channels.')

    flat_patch = patch.reshape(patch_height * patch_width, total_channels)
    flat_mask = valid_mask.reshape(-1).astype(np.int32)
    local_y, local_x = np.meshgrid(range(patch_height), range(patch_width), indexing='ij')

    coords_df = pd.DataFrame(
        {
            'Pixel_Y': local_y.flatten() + start_row,
            'Pixel_X': local_x.flatten() + start_col,
            'Start_Row': start_row,
            'Start_Col': start_col,
            'ValidMask': flat_mask,
        }
    )

    image_headers = ['RGB_R', 'RGB_G', 'RGB_B']
    image_part = flat_patch[:, :config.num_image_channels]
    spectra_part = flat_patch[:, config.num_image_channels:]

    if config.training_modality == 'image':
        payload_df = pd.DataFrame(image_part, columns=image_headers)
    elif config.training_modality == 'spectra':
        payload_df = pd.DataFrame(spectra_part, columns=list(feature_names))
    elif config.training_modality == 'both':
        payload_df = pd.DataFrame(
            np.concatenate([image_part, spectra_part], axis=1),
            columns=image_headers + list(feature_names),
        )
    else:
        raise ValueError(f'Unsupported training_modality: {config.training_modality}')

    output_filename = f'patch_{patch_index}_row_{start_row}_col_{start_col}_label_{label}.csv'
    output_path = os.path.join(output_dir, output_filename)
    pd.concat([coords_df, payload_df], axis=1).to_csv(output_path, index=False)
    return output_path


def extract_non_overlapping_pure_patches_per_file(
    processed_spectral_data: np.ndarray,
    labels: np.ndarray,
    coords: np.ndarray,
    image: np.ndarray,
    filename: str,
    feature_names: Sequence[str],
    config: PipelineConfig,
) -> Tuple[List[np.ndarray], List[int], Dict[str, int], List[Tuple[int, int]], List[np.ndarray]]:
    LOGGER.info('Extracting coverage-preserving non-overlapping patches from %s', filename)

    if processed_spectral_data.shape[0] != labels.shape[0] or coords.shape[0] != labels.shape[0]:
        raise ValueError('processed_spectral_data, labels, and coords must have the same number of rows.')
    if coords.shape[0] == 0:
        return [], [], {}, [], []

    num_spectral_channels = int(processed_spectral_data.shape[1])
    base_output_dir = config.patch_save_dir if config.patch_save_dir else 'extracted_patches'
    file_basename = os.path.splitext(os.path.basename(filename))[0]
    patch_output_dir = os.path.join(base_output_dir, file_basename)
    os.makedirs(patch_output_dir, exist_ok=True)

    patch_buckets: Dict[Tuple[int, int, int], Dict[str, np.ndarray | int]] = {}

    for idx in range(coords.shape[0]):
        row = int(coords[idx, 0])
        col = int(coords[idx, 1])
        label = int(labels[idx])
        if label < 0:
            continue

        tile_row = row // config.patch_size
        tile_col = col // config.patch_size
        start_row = tile_row * config.patch_size
        start_col = tile_col * config.patch_size
        local_row = row - start_row
        local_col = col - start_col
        key = (tile_row, tile_col, label)

        if key not in patch_buckets:
            patch_buckets[key] = {
                'image': np.zeros((config.patch_size, config.patch_size, config.num_image_channels), dtype=np.float32),
                'spectra': np.zeros((config.patch_size, config.patch_size, num_spectral_channels), dtype=np.float32),
                'mask': np.zeros((config.patch_size, config.patch_size), dtype=bool),
                'label': label,
                'start_row': start_row,
                'start_col': start_col,
            }

        bucket = patch_buckets[key]
        mask = bucket['mask']
        assert isinstance(mask, np.ndarray)
        if bool(mask[local_row, local_col]):
            raise ValueError(
                f'Duplicate coordinate detected in {filename}: row={row}, col={col}, label={label}'
            )

        image_patch = bucket['image']
        spectra_patch = bucket['spectra']
        assert isinstance(image_patch, np.ndarray)
        assert isinstance(spectra_patch, np.ndarray)

        image_patch[local_row, local_col] = image[row, col]
        spectra_patch[local_row, local_col] = processed_spectral_data[idx]
        mask[local_row, local_col] = True

    all_patches: List[np.ndarray] = []
    all_labels: List[int] = []
    patch_stats: Dict[str, int] = {}
    patch_coords_list: List[Tuple[int, int]] = []
    patch_masks: List[np.ndarray] = []

    total_covered_pixels = 0
    for patch_index, key in enumerate(sorted(patch_buckets.keys())):
        bucket = patch_buckets[key]
        image_patch = bucket['image']
        spectra_patch = bucket['spectra']
        valid_mask = bucket['mask']
        label = int(bucket['label'])
        start_row = int(bucket['start_row'])
        start_col = int(bucket['start_col'])

        assert isinstance(image_patch, np.ndarray)
        assert isinstance(spectra_patch, np.ndarray)
        assert isinstance(valid_mask, np.ndarray)

        combined_patch = np.concatenate([image_patch, spectra_patch], axis=-1).astype(np.float32)
        save_patch_to_csv(
            patch=combined_patch,
            valid_mask=valid_mask,
            label=label,
            patch_index=patch_index,
            output_dir=patch_output_dir,
            feature_names=feature_names,
            start_row=start_row,
            start_col=start_col,
            config=config,
        )

        total_covered_pixels += int(valid_mask.sum())
        all_patches.append(combined_patch)
        all_labels.append(label)
        patch_coords_list.append((start_row, start_col))
        patch_masks.append(valid_mask.astype(bool))

    expected_pixels = int(np.sum(labels != -1))
    if total_covered_pixels != expected_pixels:
        raise ValueError(
            f'Patch coverage error for {filename}: covered {total_covered_pixels} valid pixels but expected {expected_pixels}.'
        )

    patch_stats['num_patches'] = len(all_patches)
    patch_stats['patch_output_dir'] = patch_output_dir
    patch_stats['covered_pixels'] = total_covered_pixels

    if all_labels:
        unique_labels, counts = np.unique(np.asarray(all_labels, dtype=np.int32), return_counts=True)
        for label, count in zip(unique_labels, counts):
            patch_stats[config.reverse_label_map.get(int(label), f'Class_{int(label)}')] = int(count)

    LOGGER.info(
        'Extracted %d patches from %s, covered %d/%d valid pixels exactly once, saved to %s',
        len(all_patches),
        filename,
        total_covered_pixels,
        expected_pixels,
        patch_output_dir,
    )
    return all_patches, all_labels, patch_stats, patch_coords_list, patch_masks
