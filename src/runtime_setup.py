from __future__ import annotations

import logging
import os
import random

import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        LOGGER.warning('No GPU found. Training will run on CPU.')
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        LOGGER.info('Using GPUs: %s', gpus)
    except RuntimeError as exc:
        LOGGER.error('GPU configuration failed: %s', exc)


def set_global_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    LOGGER.info('Global seed set to %d', seed_value)
