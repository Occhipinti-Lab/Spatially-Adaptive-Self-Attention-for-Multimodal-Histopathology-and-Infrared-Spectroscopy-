from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from pipeline_config import DISPLAY_NAMES


LOGGER = logging.getLogger(__name__)


def get_mode_class_name_map(mode: str) -> Dict[int, str]:
    if mode == 'all':
        return dict(DISPLAY_NAMES)
    mode_names = {
        'mal_ep_vs_mal_st': {0: 'cancerous epithelium', 1: 'cancerous stroma'},
        'nonmal_ep_vs_nonmal_st': {0: 'non-cancerous epithelium', 1: 'non-cancerous stroma'},
        'mal_ep_vs_nonmal_ep': {0: 'cancerous epithelium', 1: 'non-cancerous epithelium'},
        'mal_st_vs_nonmal_st': {0: 'cancerous stroma', 1: 'non-cancerous stroma'},
        'mal_ep_vs_mal_st_vs_nonmal_ep_vs_nonmal_st': {0: 'cancerous', 1: 'non-cancerous'},
    }
    return mode_names.get(mode, {0: 'Class_0', 1: 'Class_1'})


def compute_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    unique_classes, counts = np.unique(y_train, return_counts=True)
    n_samples = int(len(y_train))
    n_classes = int(len(unique_classes))
    if n_classes == 0:
        return {}
    if n_classes == 1:
        return {int(unique_classes[0]): 1.0}

    class_weights = {
        int(cls): float(n_samples / (n_classes * count))
        for cls, count in zip(unique_classes, counts)
    }
    LOGGER.info('Computed class weights: %s', class_weights)
    return class_weights
