from __future__ import annotations

from experiment_pipeline import run_pipeline
from pipeline_config import get_config
from runtime_setup import configure_gpu, configure_logging


def main() -> None:
    configure_logging()
    configure_gpu()
    config = get_config()
    run_pipeline(config)


if __name__ == '__main__':
    main()
