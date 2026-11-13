
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

if __package__:
    from .config import ExperimentConfig
    from .data import DatasetBundle, Partition
    from .model import (
        MODEL_ARCHITECTURE,
        SparseSmoothedCategoricalCrossentropy,
        build_model,
        inference_model,
    )
else:  # Flat portable deployment: all modules beside main.py.
    from config import ExperimentConfig
    from data import DatasetBundle, Partition
    from model import (
        MODEL_ARCHITECTURE,
        SparseSmoothedCategoricalCrossentropy,
        build_model,
        inference_model,
    )

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    training_model: tf.keras.Model
    inference_model: tf.keras.Model
    history: dict
    run_dir: Path


class ContinuousProgressLogger(tf.keras.callbacks.Callback):


    def __init__(self, epochs: int, steps_per_epoch: int, log_every_batches: int):
        super().__init__()
        self.epochs = int(epochs)
        self.steps_per_epoch = int(steps_per_epoch)
        self.log_every_batches = int(log_every_batches)
        self.epoch_index = 0
        self.epoch_started = 0.0

    @staticmethod
    def _metrics(logs):
        result = []
        for name, value in (logs or {}).items():
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                result.append(f"{name}={numeric:.5f}")
        return " ".join(result)

    def on_train_begin(self, logs=None):
        LOGGER.info(
            "TRAINING START | epochs=%d steps_per_epoch=%d expected_optimizer_iterations=%d",
            self.epochs,
            self.steps_per_epoch,
            self.epochs * self.steps_per_epoch,
        )

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_index = int(epoch)
        self.epoch_started = time.perf_counter()
        LOGGER.info("EPOCH %d/%d START", epoch + 1, self.epochs)

    def on_train_batch_end(self, batch, logs=None):
        batch_number = int(batch) + 1
        should_log = (
            batch_number == 1
            or batch_number == self.steps_per_epoch
            or batch_number % self.log_every_batches == 0
        )
        if not should_log:
            return
        iteration = int(tf.keras.backend.get_value(self.model.optimizer.iterations))
        try:
            learning_rate = float(
                tf.keras.backend.get_value(self.model.optimizer.learning_rate)
            )
        except (AttributeError, TypeError, ValueError):
            learning_rate = float("nan")
        LOGGER.info(
            "TRAIN PROGRESS | epoch=%d/%d batch=%d/%d optimizer_iteration=%d lr=%s %s",
            self.epoch_index + 1,
            self.epochs,
            batch_number,
            self.steps_per_epoch,
            iteration,
            f"{learning_rate:.3e}" if np.isfinite(learning_rate) else "scheduled",
            self._metrics(logs),
        )

    def on_epoch_end(self, epoch, logs=None):
        LOGGER.info(
            "EPOCH %d/%d COMPLETE | elapsed=%.1fs %s",
            epoch + 1,
            self.epochs,
            time.perf_counter() - self.epoch_started,
            self._metrics(logs),
        )

    def on_train_end(self, logs=None):
        iteration = int(tf.keras.backend.get_value(self.model.optimizer.iterations))
        LOGGER.info("TRAINING COMPLETE | optimizer_iterations=%d", iteration)


def set_seed(seed: int, deterministic: bool = True):
    tf.keras.utils.set_random_seed(int(seed))
    if deterministic:
        try:
            tf.config.experimental.enable_op_determinism()
        except (AttributeError, RuntimeError):
            pass


def class_weights(targets, num_classes: int):
    counts = np.bincount(np.asarray(targets, dtype=int), minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"training targets omit a class: {counts.tolist()}")
    return len(targets) / (num_classes * counts.astype(float))


def _targets(targets, modality):
    if modality != "both":
        return targets
    return {"fused": targets, "rgb_aux": targets, "spectra_aux": targets}


def _sample_weights(targets, weights, modality):
    values = weights[np.asarray(targets, dtype=int)]
    if modality != "both":
        return values
    return {"fused": values, "rgb_aux": values, "spectra_aux": values}


def _compile(model, cfg: ExperimentConfig):
    loss = SparseSmoothedCategoricalCrossentropy(
        cfg.num_classes, label_smoothing=cfg.label_smoothing
    )
    try:
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            global_clipnorm=1.0,
        )
    except AttributeError:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.learning_rate, global_clipnorm=1.0
        )
    if cfg.modality == "both":
        model.compile(
            optimizer=optimizer,
            loss={"fused": loss, "rgb_aux": loss, "spectra_aux": loss},
            loss_weights={
                "fused": 1.0,
                "rgb_aux": cfg.rgb_aux_weight,
                "spectra_aux": cfg.spectra_aux_weight,
            },
            metrics={"fused": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]},
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )


def validate_modality_contract(
    bundle: DatasetBundle,
    cfg: ExperimentConfig,
    model: tf.keras.Model,
):
    """Fail closed if configuration, materialized channels, and graph disagree."""
    rgb_channels = 3 if cfg.modality in {"image", "both"} else 0
    spectral_channels = bundle.spectral_channels if cfg.modality in {"spectra", "both"} else 0
    if cfg.modality in {"spectra", "both"} and spectral_channels <= 0:
        raise RuntimeError(
            f"modality contract failed: {cfg.modality} requires positive spectral channels"
        )
    expected_channels = rgb_channels + spectral_channels
    partition_shapes = {
        "train": list(bundle.train.features.shape),
        "validation": list(bundle.validation.features.shape),
        "test": list(bundle.test.features.shape),
    }
    actual_channels = {
        name: shape[-1] for name, shape in partition_shapes.items()
    }
    if any(value != expected_channels for value in actual_channels.values()):
        raise RuntimeError(
            "modality contract failed: expected "
            f"{expected_channels} channels for {cfg.modality}, got {actual_channels}"
        )
    layer_names = {layer.name for layer in model.layers}
    required = {
        "image": {"image_channels", "image_projection", "fusion_feature_map", "fused"},
        "spectra": {"spectra_channels", "spectra_projection", "fusion_feature_map", "fused"},
        "both": {
            "rgb_channels",
            "spectral_channels",
            "rgb_projection",
            "spectra_projection",
            "gated_early_fusion",
            "rgb_aux",
            "spectra_aux",
            "fusion_feature_map",
            "fused",
        },
    }[cfg.modality]
    missing = sorted(required - layer_names)
    if missing:
        raise RuntimeError(
            f"modality contract failed: {cfg.modality} graph is missing layers {missing}"
        )
    forbidden = {
        "image": {"spectral_channels", "spectra_projection", "gated_early_fusion"},
        "spectra": {"rgb_channels", "rgb_projection", "gated_early_fusion"},
        "both": set(),
    }[cfg.modality]
    unexpected = sorted(forbidden & layer_names)
    if unexpected:
        raise RuntimeError(
            f"modality contract failed: {cfg.modality} graph contains layers {unexpected}"
        )
    contract = {
        "verified": True,
        "active_modality": cfg.modality,
        "experiment_role": cfg.experiment_role,
        "ablation_variant": cfg.ablation_variant,
        "parent_requested_modality": cfg.parent_requested_modality,
        "rgb_channels": rgb_channels,
        "spectral_channels": spectral_channels,
        "expected_total_channels": expected_channels,
        "partition_shapes": partition_shapes,
        "required_graph_layers_verified": sorted(required),
        "model_output_names": list(model.output_names),
    }
    return contract


def train(bundle: DatasetBundle, cfg: ExperimentConfig, seed: int, run_dir: Path):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    set_seed(seed, cfg.deterministic_ops)
    model = build_model(
        bundle.train.features.shape[1:],
        cfg.num_classes,
        cfg.modality,
        bundle.spectral_channels,
        latent_width=cfg.latent_width,
        encoder_widths=cfg.encoder_widths,
        attention_heads=cfg.attention_heads,
        dropout=cfg.dropout,
        head_dropout=cfg.head_dropout,
        modality_dropout=cfg.modality_dropout,
        optical_density_jitter=cfg.optical_density_jitter,
        l2=cfg.l2,
        augment=True,
    )
    modality_contract = validate_modality_contract(bundle, cfg, model)
    (run_dir / "modality_contract.json").write_text(
        json.dumps(modality_contract, indent=2), encoding="utf-8"
    )
    LOGGER.warning(
        "MODALITY CONTRACT VERIFIED | ACTIVE_MODALITY=%s role=%s variant=%s "
        "RGB_channels=%d spectral_channels=%d total_channels=%d",
        cfg.modality,
        cfg.experiment_role,
        cfg.ablation_variant or "not_applicable",
        modality_contract["rgb_channels"],
        modality_contract["spectral_channels"],
        modality_contract["expected_total_channels"],
    )
    _compile(model, cfg)
    LOGGER.info("=" * 72)
    LOGGER.info(
        "MODEL BUILT | name=%s ACTIVE_MODALITY=%s seed=%d input_shape=%s",
        model.name,
        cfg.modality,
        seed,
        bundle.train.features.shape[1:],
    )
    model.summary(print_fn=lambda line: LOGGER.info("MODEL | %s", line))
    trainable_parameters = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    non_trainable_parameters = int(
        sum(np.prod(weight.shape) for weight in model.non_trainable_weights)
    )
    LOGGER.info(
        "PARAMETERS | total=%d trainable=%d non_trainable=%d",
        model.count_params(),
        trainable_parameters,
        non_trainable_parameters,
    )
    LOGGER.info("=" * 72)
    weights = (
        class_weights(bundle.train.targets, cfg.num_classes)
        if cfg.balance_classes
        else np.ones(cfg.num_classes, dtype=float)
    )
    monitor = "val_fused_loss" if cfg.modality == "both" else "val_loss"
    checkpoint = run_dir / "best.weights.h5"
    steps_per_epoch = int(np.ceil(len(bundle.train.targets) / cfg.batch_size))
    LOGGER.info(
        "TRAINING DATA | train_samples=%d validation_samples=%d batch_size=%d class_weights=%s monitor=%s",
        len(bundle.train.targets),
        len(bundle.validation.targets),
        cfg.batch_size,
        np.round(weights, 5).tolist(),
        monitor,
    )
    callbacks = [
        ContinuousProgressLogger(cfg.epochs, steps_per_epoch, cfg.log_every_batches),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint,
            monitor=monitor,
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=monitor, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="min",
            patience=cfg.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    fit_kwargs = {
        "x": bundle.train.features,
        "y": _targets(bundle.train.targets, cfg.modality),
        "sample_weight": _sample_weights(bundle.train.targets, weights, cfg.modality),
        "validation_data": (
            bundle.validation.features,
            _targets(bundle.validation.targets, cfg.modality),
        ),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "shuffle": True,
        "callbacks": callbacks,
        "verbose": 1,
    }
    history = model.fit(**fit_kwargs)
    # EarlyStopping restores the best epoch; loading is an additional invariant
    # when training ends through a non-early-stop path.
    if checkpoint.exists():
        LOGGER.info("Restoring best checkpoint: %s", checkpoint)
        model.load_weights(checkpoint)
    fused_model = inference_model(model)
    LOGGER.info("Saving training and fused inference models to %s", run_dir)
    model.save(run_dir / "training_model.keras")
    fused_model.save(run_dir / "fused_inference_model.keras")
    pd.DataFrame(history.history).to_csv(run_dir / "history.csv", index_label="epoch")
    recipe = {
        "model_architecture": MODEL_ARCHITECTURE,
        "active_modality": cfg.modality,
        "experiment_role": cfg.experiment_role,
        "ablation_variant": cfg.ablation_variant,
        "parent_requested_modality": cfg.parent_requested_modality,
        "modality_contract": str(run_dir / "modality_contract.json"),
        "split_strategy": cfg.split_strategy,
        "split_seed": seed,
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "manifest": str(bundle.manifest_path),
        "test_set_used_during_fit": False,
        "input_shape": list(bundle.train.features.shape[1:]),
        "feature_order": "[R,G,B,spectra...]" if cfg.modality == "both" else cfg.modality,
        "spectral_channels": bundle.spectral_channels,
        "auxiliary_loss_weights": {
            "rgb": cfg.rgb_aux_weight if cfg.modality == "both" else 0,
            "spectra": cfg.spectra_aux_weight if cfg.modality == "both" else 0,
        },
        "modality_dropout": cfg.modality_dropout if cfg.modality == "both" else 0,
        "class_weights": weights.tolist(),
        "epochs_completed": len(history.epoch),
    }
    (run_dir / "training_recipe.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    LOGGER.info(
        "Training artifacts saved | epochs_completed=%d history=%s",
        len(history.epoch),
        run_dir / "history.csv",
    )
    return TrainingResult(model, fused_model, history.history, run_dir)
