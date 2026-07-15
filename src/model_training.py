from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

from file_utils import ensure_dir
from model_architecture import (
    build_model,
    build_transfer_model,
    count_trainable_and_frozen_params,
    unfreeze_top_backbone_layers,
)
from model_inputs import ModelInput, ModelInputShape
from pipeline_config import PipelineConfig
from training_plots import plot_training_history


LOGGER = logging.getLogger(__name__)


def _compile_model(model: tf.keras.Model, learning_rate: float, num_classes: int) -> None:
    loss_fn = 'binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy'
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])


def _has_validation_samples(X_val: ModelInput) -> bool:
    return X_val.shape[0] > 0


def _fit_model(
    model: tf.keras.Model,
    X_train: ModelInput,
    y_train: np.ndarray,
    X_val: ModelInput,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    class_weights: Optional[Dict[int, float]],
    log_dir: str,
) -> pd.DataFrame:
    has_validation = _has_validation_samples(X_val)
    monitor_name = 'val_loss' if has_validation else 'loss'
    callbacks = [
        ReduceLROnPlateau(monitor=monitor_name, factor=0.5, patience=8, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor=monitor_name, patience=20, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir, histogram_freq=0),
    ]
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val) if has_validation else None,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights,
    )
    return pd.DataFrame(history.history)


def _build_model_for_training(
    input_shape: ModelInputShape,
    config: PipelineConfig,
    num_classes: int,
) -> tf.keras.Model:
    if config.training_strategy == 'scratch':
        model = build_model(
            input_shape=input_shape,
            num_classes=num_classes,
            fusion_strategy=config.fusion_strategy,
            l2_reg=1e-3,
        )
        _compile_model(model, learning_rate=config.learning_rate, num_classes=num_classes)
        return model

    model = build_transfer_model(input_shape=input_shape, config=config, l2_reg=1e-3)
    _compile_model(model, learning_rate=config.transfer_learning_rate, num_classes=num_classes)
    return model


def _run_config_name(config: PipelineConfig) -> str:
    return (
        f'mode_{config.mode}_ncomp{config.n_components}_patch{config.patch_size}_'
        f'modality_{config.training_modality}_fusion_{config.fusion_strategy}_strategy_{config.training_strategy}'
    )


def train_model(
    X_train: ModelInput,
    y_train: np.ndarray,
    X_val: ModelInput,
    y_val: np.ndarray,
    input_shape: ModelInputShape,
    num_classes: int,
    config: PipelineConfig,
    seed_dir: str,
    class_weights: Optional[Dict[int, float]] = None,
) -> Tuple[tf.keras.Model, pd.DataFrame, str, str]:
    ensure_dir(seed_dir)
    model = _build_model_for_training(input_shape=input_shape, config=config, num_classes=num_classes)
    model.summary(print_fn=LOGGER.info)

    trainable_params, frozen_params = count_trainable_and_frozen_params(model)
    LOGGER.info('Model parameter counts -> trainable=%d | frozen=%d', trainable_params, frozen_params)

    run_config_str = _run_config_name(config)
    output_prefix = os.path.join(seed_dir, 'model_outputs', f'run_{run_config_str}')
    ensure_dir(os.path.dirname(output_prefix))

    all_histories: List[pd.DataFrame] = []

    if config.training_strategy == 'scratch':
        log_dir = os.path.join(seed_dir, 'logs', 'scratch', datetime.now().strftime('%Y%m%d-%H%M%S'))
        ensure_dir(log_dir)
        history_df = _fit_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            epochs=config.epochs,
            batch_size=config.batch_size,
            class_weights=class_weights,
            log_dir=log_dir,
        )
        all_histories.append(history_df)
    else:
        if config.feature_extraction_epochs > 0:
            log_dir = os.path.join(seed_dir, 'logs', 'feature_extraction', datetime.now().strftime('%Y%m%d-%H%M%S'))
            ensure_dir(log_dir)
            history_feature = _fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=config.feature_extraction_epochs,
                batch_size=config.batch_size,
                class_weights=class_weights,
                log_dir=log_dir,
            )
            history_feature['phase'] = 'feature_extraction'
            all_histories.append(history_feature)

        if config.staged_fine_tune and config.fine_tune_epochs > 0:
            unfrozen_count = unfreeze_top_backbone_layers(model, config.unfreeze_top_layers)
            LOGGER.info('Unfroze %d backbone layers for fine-tuning.', unfrozen_count)
            _compile_model(model, learning_rate=config.fine_tune_learning_rate, num_classes=num_classes)
            trainable_params, frozen_params = count_trainable_and_frozen_params(model)
            LOGGER.info('After unfreezing -> trainable=%d | frozen=%d', trainable_params, frozen_params)
            log_dir = os.path.join(seed_dir, 'logs', 'fine_tune', datetime.now().strftime('%Y%m%d-%H%M%S'))
            ensure_dir(log_dir)
            history_ft = _fit_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=config.fine_tune_epochs,
                batch_size=config.batch_size,
                class_weights=class_weights,
                log_dir=log_dir,
            )
            history_ft['phase'] = 'fine_tune'
            all_histories.append(history_ft)

    combined_history = pd.concat(all_histories, axis=0, ignore_index=True) if all_histories else pd.DataFrame()
    model.save_weights(f'{output_prefix}_final_weights.weights.h5')
    if config.save_full_model:
        model.save(f'{output_prefix}_final_model.keras')
    if not combined_history.empty:
        combined_history.to_csv(f'{output_prefix}_training_history.csv', index_label='epoch')
        plot_training_history(combined_history, run_config_str, f'{output_prefix}_training_plots.png')
    return model, combined_history, run_config_str, output_prefix
