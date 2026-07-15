from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def safe_train_val_test_split_indices(y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(y.shape[0])
    if y.shape[0] < 3:
        return indices, np.asarray([], dtype=np.int32), np.asarray([], dtype=np.int32)

    can_stratify_outer = len(np.unique(y)) >= 2 and np.all(np.bincount(y) >= 2)
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=y if can_stratify_outer else None,
        shuffle=True,
    )

    y_train_val = y[train_val_idx]
    bincount_inner = np.bincount(y_train_val) if y_train_val.size > 0 else np.asarray([])
    can_stratify_inner = (
        len(np.unique(y_train_val)) >= 2
        and bincount_inner.size > 0
        and np.all(bincount_inner[bincount_inner > 0] >= 2)
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed,
        stratify=y_train_val if can_stratify_inner else None,
        shuffle=True,
    )
    return np.asarray(train_idx), np.asarray(val_idx), np.asarray(test_idx)


def filter_patches_by_mode(
    X: np.ndarray,
    y: np.ndarray,
    coords: Optional[np.ndarray],
    file_indices: Optional[np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if X.shape[0] == 0:
        return X, y, coords, file_indices

    mode_map = {
        'mal_ep_vs_mal_st': ([0, 1], {0: 0, 1: 1}),
        'nonmal_ep_vs_nonmal_st': ([2, 3], {2: 0, 3: 1}),
        'mal_ep_vs_nonmal_ep': ([0, 2], {0: 0, 2: 1}),
        'mal_st_vs_nonmal_st': ([1, 3], {1: 0, 3: 1}),
        'mal_ep_vs_mal_st_vs_nonmal_ep_vs_nonmal_st': ([0, 1, 2, 3], {0: 0, 1: 0, 2: 1, 3: 1}),
        'all': ([0, 1, 2, 3], {0: 0, 1: 1, 2: 2, 3: 3}),
    }
    if mode not in mode_map:
        raise ValueError(f'Invalid mode: {mode}')

    relevant_labels, remap = mode_map[mode]
    mask = np.isin(y, relevant_labels)
    filtered_X = X[mask]
    filtered_y = np.asarray([remap[int(label)] for label in y[mask]], dtype=np.int32)
    filtered_coords = coords[mask] if coords is not None else None
    filtered_file_indices = file_indices[mask] if file_indices is not None else None
    return filtered_X, filtered_y, filtered_coords, filtered_file_indices
