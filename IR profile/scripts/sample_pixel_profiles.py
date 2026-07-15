from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spectral_profiles import (
    build_sampled_pixel_profiles,
    find_csv_files,
    load_spectral_csv,
    plot_sampled_profiles,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample individual pixel spectral profiles.")
    parser.add_argument("--data-dir", default=Path("data"), type=Path, help="CSV file or directory.")
    parser.add_argument("--output-dir", default=Path("results"), type=Path, help="Output directory.")
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
    csv_files = find_csv_files(args.data_dir, recursive=args.recursive)
    if not csv_files:
        raise SystemExit(f"No CSV files found in {args.data_dir}")

    profiles = []
    for index, csv_file in enumerate(csv_files):
        dataset = load_spectral_csv(csv_file, args.label_column, args.spectral_prefix)
        profiles.append(
            build_sampled_pixel_profiles(
                dataset,
                samples_per_file=args.samples_per_file,
                seed=args.seed + index,
                balance_by_label=not args.no_balance_by_label,
            )
        )

    sampled_profiles = pd.concat(profiles, ignore_index=True)
    table_dir = args.output_dir / "tables"
    plot_dir = args.output_dir / "plots" / "sampled_pixels"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_dir / "sampled_pixel_profiles.csv"
    sampled_profiles.to_csv(table_path, index=False)
    plot_paths = plot_sampled_profiles(sampled_profiles, plot_dir)

    print(f"Processed {len(csv_files)} CSV file(s).")
    print(f"Wrote {table_path}")
    print(f"Wrote {len(plot_paths)} sampled pixel plot(s) to {plot_dir}")


if __name__ == "__main__":
    main()

