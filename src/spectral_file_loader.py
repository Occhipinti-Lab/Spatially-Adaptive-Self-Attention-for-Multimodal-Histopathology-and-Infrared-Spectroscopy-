
from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import savgol_filter

from pipeline_config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def find_corresponding_image(filepath: str, image_dir: str) -> str:
    filename = os.path.basename(filepath)
    stem = os.path.splitext(filename)[0]
    for extension in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
        candidate = os.path.join(image_dir, f'{stem}{extension}')
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f'Corresponding image file for full filename stem `{stem}` was not found in `{image_dir}`.'
    )


def load_image_as_float32(image_path: str) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
    return image / 255.0


def extract_spectral_columns(df: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    spectral_columns = [column for column in df.columns if column.startswith('Wavenumber_')]
    if not spectral_columns:
        raise ValueError("No spectral columns starting with 'Wavenumber_' were found.")
    wavenumbers = np.asarray([float(column.split('_', 1)[1]) for column in spectral_columns], dtype=np.float64)
    return spectral_columns, wavenumbers


def build_range_mask(wavenumbers: np.ndarray, ranges: List[Tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(wavenumbers.shape[0], dtype=bool)
    for start, end in ranges:
        mask |= (wavenumbers >= start) & (wavenumbers <= end)
    return mask


def map_labels_to_numeric(labels_raw: np.ndarray, config: PipelineConfig) -> np.ndarray:
    numeric_labels = np.asarray([config.label_map.get(str(label), -1) for label in labels_raw], dtype=np.int32)
    return numeric_labels


def preprocess_spectra(
    spectral_data_raw: np.ndarray,
    wavenumbers: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    initial_mask = build_range_mask(wavenumbers, config.initial_wavenumber_ranges)
    initial_spectral = spectral_data_raw[:, initial_mask]
    initial_wavenumbers = wavenumbers[initial_mask]
    LOGGER.info('After initial filtering: %d bands.', initial_spectral.shape[1])

    if initial_spectral.shape[1] == 0:
        raise ValueError('No spectral bands remain after initial wavenumber filtering.')
    if initial_spectral.shape[1] < config.savgol_window:
        raise ValueError(
            f'Savitzky-Golay window ({config.savgol_window}) is larger than available spectral length '
            f'({initial_spectral.shape[1]}).'
        )

    derivative_data = np.apply_along_axis(
        lambda row: savgol_filter(
            row,
            window_length=config.savgol_window,
            polyorder=config.savgol_polyorder,
            deriv=config.savgol_deriv,
        ),
        axis=1,
        arr=initial_spectral,
    )

    final_mask = build_range_mask(initial_wavenumbers, config.final_wavenumber_ranges)
    filtered_spectral = derivative_data[:, final_mask].astype(np.float32)
    filtered_wavenumbers = initial_wavenumbers[final_mask].astype(np.float64)

    LOGGER.info('After final filtering: %d bands.', filtered_spectral.shape[1])
    if filtered_spectral.shape[1] == 0:
        raise ValueError('No spectral bands remain after final wavenumber filtering.')

    feature_names = [f'Wavenumber_{wave:.4f}' for wave in filtered_wavenumbers]
    return filtered_spectral, filtered_wavenumbers, feature_names


def load_and_process_file(
    filepath: str,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    LOGGER.info('Processing file: %s', filepath)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'CSV file not found: {filepath}')

    df = pd.read_csv(filepath)
    spectral_columns, wavenumbers = extract_spectral_columns(df)

    if 'Pixel_Y' not in df.columns or 'Pixel_X' not in df.columns or 'Label' not in df.columns:
        raise ValueError('The CSV file must contain Pixel_Y, Pixel_X, and Label columns.')

    spectral_data_raw = df[spectral_columns].to_numpy(dtype=np.float64)
    coords = df[['Pixel_Y', 'Pixel_X']].to_numpy(dtype=np.int32)
    labels = map_labels_to_numeric(df['Label'].to_numpy(), config)

    image_path = find_corresponding_image(filepath, config.image_dir)
    image = load_image_as_float32(image_path)

    height, width = image.shape[:2]
    inside_bounds = (
        (coords[:, 0] >= 0) & (coords[:, 0] < height) &
        (coords[:, 1] >= 0) & (coords[:, 1] < width)
    )
    if not np.all(inside_bounds):
        dropped = int((~inside_bounds).sum())
        LOGGER.warning('Dropping %d rows from %s because their coordinates fall outside the image bounds.', dropped, filepath)
        spectral_data_raw = spectral_data_raw[inside_bounds]
        coords = coords[inside_bounds]
        labels = labels[inside_bounds]

    filtered_spectral, filtered_wavenumbers, feature_names = preprocess_spectra(
        spectral_data_raw=spectral_data_raw,
        wavenumbers=wavenumbers,
        config=config,
    )

    LOGGER.info(
        'Loaded %s with %d valid rows, image shape %s, and %d filtered spectral bands.',
        os.path.basename(filepath),
        coords.shape[0],
        tuple(image.shape),
        filtered_spectral.shape[1],
    )
    return filtered_spectral, filtered_wavenumbers, labels, coords, image, feature_names
