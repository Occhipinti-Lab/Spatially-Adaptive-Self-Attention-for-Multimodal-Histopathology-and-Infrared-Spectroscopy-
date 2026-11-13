from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

if __package__:
    from .config import parse_config
else:  # Flat portable deployment: all modules beside main.py.
    from config import parse_config

LOGGER = logging.getLogger(__name__)


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = parse_config(argv)
    config_module = sys.modules[parse_config.__module__]
    LOGGER.info(
        "CONFIGURATION SOURCE | file=%s",
        Path(config_module.__file__).resolve(),
    )
    LOGGER.info(
        "EFFECTIVE CONFIGURATION | modality=%s split=%s seeds=%s epochs=%d final_test=%s ablation_suite=%s",
        config.modality,
        config.split_strategy,
        config.split_seeds,
        config.epochs,
        config.final_test,
        config.run_ablation_suite,
    )
    # Keep --help and validation lightweight; TensorFlow is imported only when
    # an actual experiment begins.
    if __package__:
        from .experiment import run_experiments
    else:
        from experiment import run_experiments

    if config.run_ablation_suite:
        if __package__:
            from .benchmark import run_ablation_suite
        else:
            from benchmark import run_ablation_suite

        LOGGER.warning(
            "",
            config.modality,
            config.modality,
        )
        return run_ablation_suite(config)
    return run_experiments(config)


if __name__ == "__main__":
    main()
