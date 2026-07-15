from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectral_profiles import (  # noqa: E402
    build_mean_profiles,
    build_sampled_pixel_profiles,
    find_csv_files,
    load_spectral_csv,
    plot_mean_profiles,
    plot_sampled_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full spectral profile workflow.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="CSV file or directory. Defaults to ./data, then ../data if present.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Output directory.",
    )
    parser.add_argument("--samples-per-file", default=24, type=int, help="Pixels to sample per file.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed.")
    parser.add_argument("--label-column", default="Label", help="Column containing class labels.")
    parser.add_argument(
        "--spectral-prefix",
        default="Wavenumber_",
        help="Prefix used by spectral columns.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search for CSV files recursively.")
    parser.add_argument(
        "--no-balance-by-label",
        action="store_true",
        help="Sample from the whole file instead of balancing across labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir or _default_data_dir()

    csv_files = find_csv_files(data_dir, recursive=args.recursive)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {data_dir}")

    mean_tables = []
    sampled_tables = []

    for index, csv_file in enumerate(csv_files):
        dataset = load_spectral_csv(csv_file, args.label_column, args.spectral_prefix)
        mean_tables.append(build_mean_profiles(dataset))
        sampled_tables.append(
            build_sampled_pixel_profiles(
                dataset,
                samples_per_file=args.samples_per_file,
                seed=args.seed + index,
                balance_by_label=not args.no_balance_by_label,
            )
        )

    mean_profiles = pd.concat(mean_tables, ignore_index=True)
    sampled_profiles = pd.concat(sampled_tables, ignore_index=True)

    table_dir = args.output_dir / "tables"
    mean_plot_dir = args.output_dir / "plots" / "mean_profiles"
    sample_plot_dir = args.output_dir / "plots" / "sampled_pixels"
    table_dir.mkdir(parents=True, exist_ok=True)

    mean_table_path = table_dir / "mean_spectral_profiles.csv"
    sample_table_path = table_dir / "sampled_pixel_profiles.csv"
    mean_profiles.to_csv(mean_table_path, index=False)
    sampled_profiles.to_csv(sample_table_path, index=False)

    mean_plot_paths = plot_mean_profiles(mean_profiles, mean_plot_dir)
    sample_plot_paths = plot_sampled_profiles(sampled_profiles, sample_plot_dir)

    metadata = {
        "data_dir": str(data_dir),
        "output_dir": str(args.output_dir),
        "csv_files": [str(path) for path in csv_files],
        "file_count": len(csv_files),
        "samples_per_file": args.samples_per_file,
        "seed": args.seed,
        "label_column": args.label_column,
        "spectral_prefix": args.spectral_prefix,
        "balance_by_label": not args.no_balance_by_label,
        "outputs": {
            "mean_profiles": str(mean_table_path),
            "sampled_profiles": str(sample_table_path),
            "mean_plots": [str(path) for path in mean_plot_paths],
            "sampled_plots": [str(path) for path in sample_plot_paths],
        },
    }

    metadata_path = args.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Processed {len(csv_files)} CSV file(s).")
    print(f"Wrote {mean_table_path}")
    print(f"Wrote {sample_table_path}")
    print(f"Wrote {len(mean_plot_paths)} mean profile plot(s) to {mean_plot_dir}")
    print(f"Wrote {len(sample_plot_paths)} sampled pixel plot(s) to {sample_plot_dir}")
    print(f"Wrote {metadata_path}")


def _default_data_dir() -> Path:
    candidates = [PROJECT_ROOT / "data", PROJECT_ROOT.parent / "data"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / "data"


if __name__ == "__main__":
    main()

