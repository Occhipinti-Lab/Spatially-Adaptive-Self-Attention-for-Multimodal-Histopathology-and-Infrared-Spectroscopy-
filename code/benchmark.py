"""Matched-seed modality ablation suite."""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

if __package__:
    from .config import ExperimentConfig
else:
    from config import ExperimentConfig

LOGGER = logging.getLogger(__name__)


def ablation_variants(cfg: ExperimentConfig):
    return list(dict.fromkeys([cfg.modality, "image", "spectra", "both"]))


def run_ablation_suite(cfg: ExperimentConfig):
    if __package__:
        from .experiment import run_experiments
    else:
        from experiment import run_experiments

    variants = ablation_variants(cfg)
    suite_root = Path(cfg.output_dir) / cfg.mode / cfg.split_strategy / "ablation_suite"
    suite_root.mkdir(parents=True, exist_ok=True)
    controller = {
        "experiment_role": "ablation_suite_controller",
        "requested_modality": cfg.modality,
        "variants_in_execution_order": variants,
        "split_seeds": list(cfg.split_seeds),
        "final_test": bool(cfg.final_test),
        "requested_config": cfg.to_dict(),
    }
    (suite_root / "ablation_suite_controller.json").write_text(
        json.dumps(controller, indent=2), encoding="utf-8"
    )
    frames = []
    for variant_index, modality in enumerate(variants, start=1):
        LOGGER.info("ABLATION VARIANT %d/%d | modality=%s", variant_index, len(variants), modality)
        variant = replace(
            cfg,
            modality=modality,
            run_ablation_suite=False,
            experiment_role="ablation_variant",
            parent_requested_modality=cfg.modality,
            ablation_variant=modality,
        )
        frame = run_experiments(variant).copy()
        frame.insert(0, "variant", modality)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(suite_root / "ablation_suite_seed_results.csv", index=False)
    numeric = [
        column for column in combined
        if column.startswith(("validation_", "test_"))
        and pd.api.types.is_numeric_dtype(combined[column])
    ]
    aggregate = combined.groupby("variant", as_index=False)[numeric].agg(["mean", "std"])
    aggregate.to_csv(suite_root / "ablation_suite_aggregate.csv")
    return combined
