from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .annotation import AnnotationSummary, label_spectral_csv, label_spectral_directory
from .spectral_export import discover_agilent_files, extract_spectral_file, extract_spectral_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectral-pathology",
        description="Extract Agilent FTIR spectra and label pixels from pathology annotation images.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract .dmt or .seq spectra to CSV.")
    extract_parser.add_argument("source", help="Agilent file or directory containing Agilent files.")
    extract_parser.add_argument("--output-dir", default="data/spectra", help="Directory for spectra CSV files.")
    extract_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Search only the top level when the source is a directory.",
    )

    label_parser = subparsers.add_parser("label", help="Assign pathology labels to spectra CSV files.")
    label_parser.add_argument("spectra", help="Spectra CSV file or directory of spectra CSV files.")
    label_parser.add_argument("--annotation-image", help="Annotation image for a single spectra CSV.")
    label_parser.add_argument("--annotations-dir", help="Directory containing annotation images.")
    label_parser.add_argument("--output-dir", default="data/labeled", help="Directory for labeled CSV files.")
    label_parser.add_argument(
        "--color-tolerance",
        type=int,
        default=0,
        help="Maximum per-channel RGB difference allowed when matching annotation colors.",
    )

    run_parser = subparsers.add_parser("run", help="Extract spectra, then apply pathology labels.")
    run_parser.add_argument("source", help="Agilent file or directory containing Agilent files.")
    run_parser.add_argument("--annotations-dir", required=True, help="Directory containing annotation images.")
    run_parser.add_argument("--spectra-dir", default="data/spectra", help="Directory for spectra CSV files.")
    run_parser.add_argument("--output-dir", default="data/labeled", help="Directory for labeled CSV files.")
    run_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Search only the top level when the source is a directory.",
    )
    run_parser.add_argument(
        "--color-tolerance",
        type=int,
        default=0,
        help="Maximum per-channel RGB difference allowed when matching annotation colors.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "extract":
            return _run_extract(args)
        if args.command == "label":
            return _run_label(args)
        if args.command == "run":
            return _run_pipeline(args)
    except Exception as error:
        parser.exit(1, f"error: {error}\n")

    parser.exit(1, "error: unknown command\n")
    return 1


def _run_extract(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if source.is_file():
        summaries = [extract_spectral_file(source, args.output_dir)]
    else:
        files = discover_agilent_files(source, recursive=not args.no_recursive)
        summaries = extract_spectral_files(files, args.output_dir)

    for summary in summaries:
        print(
            f"extracted {summary.source_file} -> {summary.output_csv} "
            f"({summary.height}x{summary.width}, {summary.spectral_points} spectra points)"
        )
    return 0


def _run_label(args: argparse.Namespace) -> int:
    spectra_path = Path(args.spectra)

    if spectra_path.is_file():
        if not args.annotation_image:
            raise ValueError("--annotation-image is required when labeling a single CSV file.")
        output_dir = Path(args.output_dir)
        output_csv = output_dir / f"{spectra_path.stem}_labeled.csv"
        summary = label_spectral_csv(
            data_csv_path=spectra_path,
            annotation_image_path=args.annotation_image,
            output_csv_path=output_csv,
            color_tolerance=args.color_tolerance,
        )
        _print_annotation_summary(summary)
        return 0

    if not args.annotations_dir:
        raise ValueError("--annotations-dir is required when labeling a directory.")

    batch = label_spectral_directory(
        data_dir=spectra_path,
        annotation_dir=args.annotations_dir,
        output_dir=args.output_dir,
        color_tolerance=args.color_tolerance,
    )
    for summary in batch.processed:
        _print_annotation_summary(summary)
    for missing_csv in batch.missing_annotations:
        print(f"missing annotation image for {missing_csv}")
    return 0


def _run_pipeline(args: argparse.Namespace) -> int:
    source = Path(args.source)
    if source.is_file():
        extract_spectral_file(source, args.spectra_dir)
    else:
        files = discover_agilent_files(source, recursive=not args.no_recursive)
        extract_spectral_files(files, args.spectra_dir)

    batch = label_spectral_directory(
        data_dir=args.spectra_dir,
        annotation_dir=args.annotations_dir,
        output_dir=args.output_dir,
        color_tolerance=args.color_tolerance,
    )
    for summary in batch.processed:
        _print_annotation_summary(summary)
    for missing_csv in batch.missing_annotations:
        print(f"missing annotation image for {missing_csv}")
    return 0


def _print_annotation_summary(summary: AnnotationSummary) -> None:
    print(
        f"labeled {summary.input_csv} -> {summary.output_csv} "
        f"({summary.labeled_rows}/{summary.input_rows} rows)"
    )


if __name__ == "__main__":
    raise SystemExit(main())
