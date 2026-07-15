from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from pipeline_config import PipelineConfig

try:
    import umap  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError('The package `umap-learn` is required. Install it with `pip install umap-learn`.') from exc


LOGGER = logging.getLogger(__name__)


@dataclass
class RangeReducer:
    start: float
    end: float
    feature_indices: np.ndarray
    reducer: object
    n_components: int


@dataclass
class SpectralTransformBundle:
    filtered_wavenumbers: np.ndarray
    range_reducers: List[RangeReducer]
    scaler: Optional[MinMaxScaler]
    output_feature_names: List[str]
    input_feature_names: List[str]


def _allocate_exact_umap_components(
    filtered_wavenumbers: np.ndarray,
    config: PipelineConfig,
    n_train_pixels: int,
) -> List[Tuple[Tuple[int, int], np.ndarray, int]]:
    non_empty_ranges: List[Tuple[Tuple[int, int], np.ndarray, int]] = []
    max_total_components = 0
    for start, end in config.final_wavenumber_ranges:
        feature_indices = np.where((filtered_wavenumbers >= start) & (filtered_wavenumbers <= end))[0]
        if feature_indices.size == 0:
            continue
        max_for_range = int(min(feature_indices.size, max(1, n_train_pixels - 1)))
        non_empty_ranges.append(((start, end), feature_indices, max_for_range))
        max_total_components += max_for_range

    if not non_empty_ranges:
        raise ValueError('No non-empty spectral ranges were found for exact UMAP allocation.')
    if config.n_components < len(non_empty_ranges):
        raise ValueError(
            f'Requested n_components={config.n_components} but there are {len(non_empty_ranges)} non-empty ranges. '
            'At least one component per range is required.'
        )
    if config.n_components > max_total_components:
        raise ValueError(
            f'Cannot allocate exactly {config.n_components} UMAP components. '
            f'The maximum feasible total given the available bands and training pixels is {max_total_components}.'
        )

    band_counts = np.asarray([feature_indices.size for _, feature_indices, _ in non_empty_ranges], dtype=np.float64)
    raw_allocations = (band_counts / band_counts.sum()) * config.n_components
    allocations = np.floor(raw_allocations).astype(int)
    allocations = np.maximum(allocations, 1)

    max_per_range = np.asarray([max_components for _, _, max_components in non_empty_ranges], dtype=np.int32)
    allocations = np.minimum(allocations, max_per_range)

    current_total = int(allocations.sum())
    remainders = raw_allocations - np.floor(raw_allocations)

    while current_total < config.n_components:
        candidates = np.argsort(-remainders)
        progressed = False
        for index in candidates:
            if allocations[index] < max_per_range[index]:
                allocations[index] += 1
                current_total += 1
                progressed = True
                if current_total == config.n_components:
                    break
        if not progressed:
            raise ValueError('Unable to reach the exact requested UMAP component count.')

    while current_total > config.n_components:
        candidates = np.argsort(remainders)
        progressed = False
        for index in candidates:
            if allocations[index] > 1:
                allocations[index] -= 1
                current_total -= 1
                progressed = True
                if current_total == config.n_components:
                    break
        if not progressed:
            raise ValueError('Unable to reduce allocations to the exact requested UMAP component count.')

    if int(allocations.sum()) != config.n_components:
        raise ValueError(
            f'Exact UMAP allocation failed: requested {config.n_components} but allocated {int(allocations.sum())}.'
        )

    allocation_summary = []
    for (range_bounds, feature_indices, _), allocated in zip(non_empty_ranges, allocations):
        allocation_summary.append((range_bounds, feature_indices, int(allocated)))
    return allocation_summary


def _flatten_valid_spectral_pixels(patches: np.ndarray, masks: np.ndarray, config: PipelineConfig) -> np.ndarray:
    spectra = patches[..., config.num_image_channels:]
    flattened_spectra = spectra.reshape(-1, spectra.shape[-1])
    flattened_masks = masks.reshape(-1)
    valid_pixels = flattened_spectra[flattened_masks]
    if valid_pixels.shape[0] == 0:
        raise ValueError('No valid spectral pixels were found in the training patches.')
    return valid_pixels.astype(np.float32)


def fit_spectral_transform_bundle(
    train_patches_raw: np.ndarray,
    train_masks: np.ndarray,
    filtered_wavenumbers: np.ndarray,
    raw_feature_names: List[str],
    config: PipelineConfig,
) -> SpectralTransformBundle:
    train_valid_pixels = _flatten_valid_spectral_pixels(train_patches_raw, train_masks, config)
    if train_valid_pixels.shape[0] < 3:
        raise ValueError('At least three valid training pixels are required to fit exact UMAP safely.')
    LOGGER.info('Fitting spectral transformers on %d valid training pixels only.', train_valid_pixels.shape[0])

    if not config.use_umap:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(train_valid_pixels)
        return SpectralTransformBundle(
            filtered_wavenumbers=filtered_wavenumbers.copy(),
            range_reducers=[],
            scaler=scaler,
            output_feature_names=list(raw_feature_names),
            input_feature_names=list(raw_feature_names),
        )

    exact_allocation = _allocate_exact_umap_components(filtered_wavenumbers, config, train_valid_pixels.shape[0])

    reduced_training_parts = []
    range_reducers: List[RangeReducer] = []
    output_feature_names: List[str] = []
    feature_counter = 0

    for (start, end), feature_indices, n_components in exact_allocation:
        range_data = train_valid_pixels[:, feature_indices]
        n_neighbors = min(config.umap_n_neighbors, max(2, range_data.shape[0] - 1))
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=config.umap_min_dist,
            metric=config.umap_metric,
            random_state=42,
            n_jobs=1,
        )
        reduced_range = reducer.fit_transform(range_data).astype(np.float32)
        reduced_training_parts.append(reduced_range)
        range_reducers.append(
            RangeReducer(
                start=float(start),
                end=float(end),
                feature_indices=feature_indices.astype(np.int32),
                reducer=reducer,
                n_components=int(n_components),
            )
        )
        for local_index in range(n_components):
            output_feature_names.append(f'UMAP_{int(start)}_{int(end)}_comp_{feature_counter + local_index}')
        feature_counter += n_components
        LOGGER.info(
            'Fitted exact UMAP range %s-%s with %d input bands -> %d components.',
            start,
            end,
            feature_indices.size,
            n_components,
        )

    reduced_training = np.concatenate(reduced_training_parts, axis=1).astype(np.float32)
    if reduced_training.shape[1] != config.n_components:
        raise ValueError(
            f'Exact UMAP failed after fitting: requested {config.n_components} components but got {reduced_training.shape[1]}.'
        )

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(reduced_training)
    LOGGER.info('Exact UMAP allocation succeeded: requested=%d, achieved=%d', config.n_components, reduced_training.shape[1])
    return SpectralTransformBundle(
        filtered_wavenumbers=filtered_wavenumbers.copy(),
        range_reducers=range_reducers,
        scaler=scaler,
        output_feature_names=output_feature_names,
        input_feature_names=list(raw_feature_names),
    )


def transform_patches_with_bundle(
    patches_raw: np.ndarray,
    patch_masks: np.ndarray,
    bundle: SpectralTransformBundle,
    config: PipelineConfig,
) -> np.ndarray:
    image_part = patches_raw[..., :config.num_image_channels].astype(np.float32)
    raw_spectra = patches_raw[..., config.num_image_channels:].astype(np.float32)
    flattened_spectra = raw_spectra.reshape(-1, raw_spectra.shape[-1])
    flattened_masks = patch_masks.reshape(-1)
    transformed_flat = np.zeros((flattened_spectra.shape[0], len(bundle.output_feature_names)), dtype=np.float32)

    if np.any(flattened_masks):
        valid_pixels = flattened_spectra[flattened_masks]
        if bundle.range_reducers:
            reduced_parts = []
            for range_reducer in bundle.range_reducers:
                range_input = valid_pixels[:, range_reducer.feature_indices]
                reduced_parts.append(range_reducer.reducer.transform(range_input).astype(np.float32))
            reduced_pixels = np.concatenate(reduced_parts, axis=1).astype(np.float32)
        else:
            reduced_pixels = valid_pixels.astype(np.float32)

        if reduced_pixels.shape[1] != len(bundle.output_feature_names):
            raise ValueError(
                f'Transformed spectral dimension mismatch: expected {len(bundle.output_feature_names)} got {reduced_pixels.shape[1]}.'
            )

        if bundle.scaler is not None:
            reduced_pixels = bundle.scaler.transform(reduced_pixels).astype(np.float32)
        reduced_pixels = np.nan_to_num(reduced_pixels, nan=0.0, posinf=0.0, neginf=0.0)
        transformed_flat[flattened_masks] = reduced_pixels

    transformed_spectra = transformed_flat.reshape(
        patches_raw.shape[0],
        config.patch_size,
        config.patch_size,
        len(bundle.output_feature_names),
    )
    return np.concatenate([image_part, transformed_spectra], axis=-1).astype(np.float32)


def save_spectral_transform_bundle(bundle: SpectralTransformBundle, output_path: str) -> None:
    with open(output_path, 'wb') as handle:
        pickle.dump(bundle, handle)
