"""End-to-end experiment orchestration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

if __package__:
    from .config import ExperimentConfig, image_directory_candidates
    from .audit import run_duplicate_audit
    from .data import build_dataset
    from .evaluation import TemperatureScaler, evaluate_partition
    from .explainability import run_explainability
    from .shap_explainability import run_spectra_shap
    from .training import train
else:  # Flat portable deployment: all modules beside main.py.
    from config import ExperimentConfig, image_directory_candidates
    from audit import run_duplicate_audit
    from data import build_dataset
    from evaluation import TemperatureScaler, evaluate_partition
    from explainability import run_explainability
    from shap_explainability import run_spectra_shap
    from training import train

LOGGER = logging.getLogger(__name__)


def configure_accelerator():
    devices = tf.config.list_physical_devices("GPU")
    LOGGER.info(
        "Compute devices | GPUs=%s",
        [device.name for device in devices] if devices else "none (CPU execution)",
    )
    for device in devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass


def run_experiments(cfg: ExperimentConfig):
    cfg.validate()
    if not cfg.data_dir.is_dir():
        raise FileNotFoundError(f"spectral data directory does not exist: {cfg.data_dir}")
    if cfg.modality in {"image", "both"} and not cfg.image_dir.is_dir():
        searched = "\n  - ".join(
            str(path) for path in image_directory_candidates(cfg.data_dir)
        )
        raise FileNotFoundError(
            "No co-registered image directory was found. Automatic discovery searched:\n"
            f"  - {searched}\n"
            "Place images in DATA_DIR/images, beside DATA_DIR in an images folder, "
            "or pass --image-dir explicitly."
        )
    representation = "rgb" if cfg.modality == "image" else "full_spectra"
    run_root = (
        cfg.output_dir
        / cfg.mode
        / cfg.split_strategy
        / cfg.modality
        / representation
        / f"patch_{cfg.patch_size}"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    cfg.save(run_root / "experiment_config.json")
    LOGGER.info("=" * 72)
    LOGGER.info(
        "EXPERIMENT START | role=%s variant=%s parent_requested_modality=%s seeds=%s "
        "split=%s ACTIVE_MODALITY=%s patch_size=%d",
        cfg.experiment_role,
        cfg.ablation_variant or "not_applicable",
        cfg.parent_requested_modality or cfg.modality,
        cfg.split_seeds,
        cfg.split_strategy,
        cfg.modality,
        cfg.patch_size,
    )
    LOGGER.info("Input data=%s | images=%s | output=%s", cfg.data_dir, cfg.image_dir, run_root)
    LOGGER.info("Phase: dataset duplicate audit")
    run_duplicate_audit(
        cfg,
        run_root / "dataset_duplicate_audit.json",
        strict=cfg.strict_duplicate_audit,
    )
    configure_accelerator()
    if cfg.split_strategy == "patch":
        LOGGER.warning(
            "PATCH SPLIT SELECTED: patients/source files may cross partitions. "
            "Do not interpret its test score as unseen-patient generalization."
        )
    rows = []
    for seed_index, seed in enumerate(cfg.split_seeds, start=1):
        seed_dir = run_root / f"seed_{seed}"
        LOGGER.info("=" * 72)
        LOGGER.info("SEED %d/%d START | seed=%d", seed_index, len(cfg.split_seeds), seed)
        LOGGER.info("Phase 1/4: split construction and patch extraction")
        bundle = build_dataset(cfg, seed, seed_dir / "data")
        LOGGER.info("Phase 2/4: model construction and training")
        result = train(bundle, cfg, seed, seed_dir / "model")

        LOGGER.info("Phase 3/4: validation evaluation and calibration")
        validation_report, validation_raw = evaluate_partition(
            result.inference_model,
            bundle.validation,
            seed_dir / "validation",
            cfg.batch_size,
            cfg.bootstrap_iterations,
            seed,
            cfg.modality,
            class_names=cfg.target_names,
            partition_name="validation",
            split_seed=seed,
            experiment_role=cfg.experiment_role,
            ablation_variant=cfg.ablation_variant,
        )
        calibration = TemperatureScaler().fit(validation_raw, bundle.validation.targets)
        LOGGER.info("Validation temperature calibration fitted: T=%.5f", calibration.temperature)
        (seed_dir / "calibration.json").write_text(
            json.dumps(
                {
                    "temperature": calibration.temperature,
                    "fit_scope": "validation only",
                    "test_used": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        row = {
            "seed": seed,
            "split_strategy": cfg.split_strategy,
            "modality": cfg.modality,
            "spectral_channels": bundle.spectral_channels,
            "train_patches": len(bundle.train.targets),
            "validation_patches": len(bundle.validation.targets),
            "test_patches": len(bundle.test.targets),
            **{f"validation_{key}": value for key, value in validation_report["metrics"].items()
               if np.isscalar(value)},
        }
        if cfg.final_test:
            LOGGER.warning("FINAL TEST ENABLED: architecture and hyperparameters must already be frozen")
            test_report, _ = evaluate_partition(
                result.inference_model,
                bundle.test,
                seed_dir / "test",
                cfg.batch_size,
                cfg.bootstrap_iterations,
                seed + 10000,
                cfg.modality,
                temperature=calibration,
                class_names=cfg.target_names,
                partition_name="test",
                split_seed=seed,
                experiment_role=cfg.experiment_role,
                ablation_variant=cfg.ablation_variant,
            )
            row.update(
                {f"test_{key}": value for key, value in test_report["metrics"].items()
                 if np.isscalar(value)}
            )
        if cfg.run_explainability:
            LOGGER.info("Phase 4/4: held-out explainability")
            explained_partition = bundle.test if cfg.final_test else bundle.validation
            explained_name = "test" if cfg.final_test else "validation"
            run_explainability(
                result.inference_model,
                explained_partition,
                seed_dir / explained_name / "explainability",
                cfg.modality,
                bundle.feature_names,
                max_samples=cfg.explain_samples,
                integrated_gradient_steps=cfg.integrated_gradient_steps,
                occlusion_window=cfg.occlusion_window,
                occlusion_stride=cfg.occlusion_stride,
                perturbation_repeats=cfg.perturbation_repeats,
                perturbation_scale=cfg.perturbation_scale,
                seed=seed,
                class_names=cfg.target_names,
            )
            if cfg.modality == "spectra" and cfg.run_shap:
                run_spectra_shap(
                    result.inference_model,
                    bundle.train,
                    explained_partition,
                    bundle.feature_names,
                    seed_dir / explained_name / "explainability" / "shap",
                    max_samples=cfg.explain_samples,
                    background_size=cfg.shap_background_size,
                    nsamples=cfg.shap_nsamples,
                    top_features=cfg.shap_top_features,
                    seed=seed,
                )
        else:
            LOGGER.info("Phase 4/4: explainability skipped by configuration")
        rows.append(row)
        LOGGER.info("SEED %d/%d COMPLETE | seed=%d output=%s", seed_index, len(cfg.split_seeds), seed, seed_dir)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_root / "seed_summary.csv", index=False)
    metric_columns = [
        column
        for column in summary.columns
        if column.startswith(("validation_", "test_"))
        and pd.api.types.is_numeric_dtype(summary[column])
    ]
    aggregate = {
        column: {
            "mean": (
                float(summary[column].mean())
                if np.isfinite(summary[column].mean())
                else None
            ),
            "standard_deviation": (
                float(summary[column].std(ddof=1))
                if len(summary) > 1 and np.isfinite(summary[column].std(ddof=1))
                else 0.0 if len(summary) == 1 else None
            ),
            "seeds": int(summary[column].notna().sum()),
        }
        for column in metric_columns
    }
    (run_root / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )
    LOGGER.info(
        "EXPERIMENT COMPLETE | seeds_completed=%d summary=%s aggregate=%s",
        len(rows),
        run_root / "seed_summary.csv",
        run_root / "aggregate_summary.json",
    )
    LOGGER.info("=" * 72)
    return summary
