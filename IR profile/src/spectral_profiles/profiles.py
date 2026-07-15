from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from spectral_profiles.io import SpectralDataset


def build_mean_profiles(dataset: SpectralDataset) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for label, group in dataset.frame.groupby(dataset.label_column, sort=False):
        values = group[dataset.spectral_columns].to_numpy(dtype=float)
        means = np.nanmean(values, axis=0)
        valid_counts = np.isfinite(values).sum(axis=0)

        for column, wavenumber, mean_value, valid_count in zip(
            dataset.spectral_columns,
            dataset.wavenumbers,
            means,
            valid_counts,
            strict=True,
        ):
            rows.append(
                {
                    "source_file": dataset.source_file,
                    "source_path": str(dataset.source_path),
                    "label": str(label),
                    "spectral_column": column,
                    "wavenumber": wavenumber,
                    "mean_intensity": mean_value,
                    "valid_pixel_count": int(valid_count),
                    "group_pixel_count": int(len(group)),
                }
            )

    return pd.DataFrame(rows)


def build_sampled_pixel_profiles(
    dataset: SpectralDataset,
    samples_per_file: int = 24,
    seed: int = 42,
    balance_by_label: bool = True,
) -> pd.DataFrame:
    if samples_per_file <= 0:
        return pd.DataFrame()

    rng = np.random.default_rng(seed)
    sampled_indices = _sample_row_indices(dataset, samples_per_file, rng, balance_by_label)
    sampled = dataset.frame.loc[sampled_indices].copy()

    rows: list[dict[str, object]] = []
    for sample_number, (row_index, row) in enumerate(sampled.iterrows(), start=1):
        label = str(row[dataset.label_column])
        sample_id = f"{dataset.source_path.stem}_pixel_{sample_number:03d}"

        for column, wavenumber in zip(dataset.spectral_columns, dataset.wavenumbers, strict=True):
            rows.append(
                {
                    "source_file": dataset.source_file,
                    "source_path": str(dataset.source_path),
                    "sample_id": sample_id,
                    "row_index": int(row_index),
                    "label": label,
                    "spectral_column": column,
                    "wavenumber": wavenumber,
                    "intensity": row[column],
                }
            )

    return pd.DataFrame(rows)


def _sample_row_indices(
    dataset: SpectralDataset,
    samples_per_file: int,
    rng: np.random.Generator,
    balance_by_label: bool,
) -> list[int]:
    valid_rows = dataset.frame[dataset.spectral_columns].notna().any(axis=1)
    frame = dataset.frame.loc[valid_rows]

    if frame.empty:
        return []

    if not balance_by_label:
        sample_count = min(samples_per_file, len(frame))
        return rng.choice(frame.index.to_numpy(), size=sample_count, replace=False).tolist()

    labels = frame[dataset.label_column].astype(str)
    label_values = labels.drop_duplicates().tolist()
    per_label = max(1, ceil(samples_per_file / len(label_values)))

    selected: list[int] = []
    for label in label_values:
        label_indices = frame.index[labels == label].to_numpy()
        sample_count = min(per_label, len(label_indices))
        selected.extend(rng.choice(label_indices, size=sample_count, replace=False).tolist())

    if len(selected) > samples_per_file:
        selected = rng.choice(np.array(selected), size=samples_per_file, replace=False).tolist()

    selected.sort()
    return [int(index) for index in selected]

