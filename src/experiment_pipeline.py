from __future__ import annotations

import logging
import os
from typing import Sequence

import numpy as np

from data_balancing import apply_balancing
from dataset_builder import create_patch_pool_and_split
from dataset_split import filter_patches_by_mode
from evaluation import evaluate_model
from experiment_labels import compute_class_weights, get_mode_class_name_map
from explainability_runner import run_explainability_suite
from model_inputs import prepare_model_inputs
from model_training import train_model
from pipeline_config import PipelineConfig
from runtime_setup import set_global_seed


LOGGER = logging.getLogger(__name__)


def _prepare_dataset_for_model(
    X: np.ndarray,
    config: PipelineConfig,
    num_spectra_channels: int,
):
    return prepare_model_inputs(
        X=X,
        num_image_channels=config.num_image_channels,
        num_spectra_channels=num_spectra_channels,
        training_modality=config.training_modality,
        fusion_strategy=config.fusion_strategy,
        patch_size=config.patch_size,
    )


def _model_class_count(y_train: np.ndarray) -> int:
    unique_class_count = len(np.unique(y_train)) if y_train.size > 0 else 0
    model_num_classes = 1 if unique_class_count == 2 else unique_class_count
    if model_num_classes == 0:
        raise ValueError('No classes are available for training.')
    return model_num_classes


def _log_config(summary_lines: Sequence[str]) -> None:
    for line in summary_lines:
        LOGGER.info(line)


def run_pipeline(config: PipelineConfig) -> None:
    os.makedirs(config.output_root_dir, exist_ok=True)
    _log_config(config.summary_lines())

    LOGGER.info('Creating leakage-safe patch pool and patch-level split once for all runs.')
    split_data = create_patch_pool_and_split(config=config, seed=config.seed_list[0])
    X_train, y_train = split_data['train']
    X_val, y_val = split_data['validation']
    X_test, y_test = split_data['test']
    feature_names = split_data['feature_names']

    if X_train.size == 0:
        raise ValueError('Training set is empty after dataset creation.')

    num_spectra_channels = int(X_train.shape[-1] - config.num_image_channels)
    LOGGER.info('Detected spectral channels after train-only preprocessing: %d', num_spectra_channels)

    X_train, y_train, _, _ = filter_patches_by_mode(X_train, y_train, None, None, config.mode)
    X_val, y_val, _, _ = filter_patches_by_mode(X_val, y_val, None, None, config.mode)
    X_test, y_test, _, _ = filter_patches_by_mode(X_test, y_test, None, None, config.mode)

    X_train, y_train = apply_balancing(X_train, y_train, config=config, seed=config.seed_list[0])

    LOGGER.info(
        'Final dataset sizes after filtering/balancing -> train=%d, validation=%d, test=%d',
        y_train.shape[0],
        y_val.shape[0],
        y_test.shape[0],
    )

    for run_index, seed in enumerate(config.seed_list[:config.num_runs], start=1):
        LOGGER.info('===== Run %d/%d | Seed %d =====', run_index, config.num_runs, seed)
        set_global_seed(seed)

        X_train_run, model_input_shape = _prepare_dataset_for_model(X_train, config, num_spectra_channels)
        X_val_run, _ = _prepare_dataset_for_model(X_val, config, num_spectra_channels)
        X_test_run, _ = _prepare_dataset_for_model(X_test, config, num_spectra_channels)

        class_weights = compute_class_weights(y_train)
        model_num_classes = _model_class_count(y_train)

        seed_dir = os.path.join(config.output_root_dir, f'run_seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)

        model, _, run_config_str, _ = train_model(
            X_train=X_train_run,
            y_train=y_train,
            X_val=X_val_run,
            y_val=y_val,
            input_shape=model_input_shape,
            num_classes=model_num_classes,
            config=config,
            seed_dir=seed_dir,
            class_weights=class_weights,
        )

        if y_test.shape[0] > 0:
            eval_output_dir = os.path.join(seed_dir, 'test_results')
            _, test_accuracy = evaluate_model(
                model=model,
                X_test=X_test_run,
                y_test=y_test,
                class_names_map=get_mode_class_name_map(config.mode),
                run_config_str=run_config_str,
                output_prefix=eval_output_dir,
            )
            LOGGER.info('Run %d test accuracy: %.4f', run_index, test_accuracy)
            if config.run_explainability:
                run_explainability_suite(
                    model=model,
                    X_test=X_test_run,
                    y_test=y_test,
                    config=config,
                    feature_names=feature_names,
                    output_dir=os.path.join(seed_dir, 'explainability'),
                )
        else:
            LOGGER.warning('Test set is empty. Skipping evaluation and explainability for run %d.', run_index)
