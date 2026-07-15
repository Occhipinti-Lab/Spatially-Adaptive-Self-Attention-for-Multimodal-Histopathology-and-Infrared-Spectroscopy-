from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .agilent_reader import AgilentIRFile


SUPPORTED_EXTENSIONS = {".dmt", ".seq"}


@dataclass(frozen=True)
class SpectralExportSummary:
    source_file: Path | None
    output_csv: Path
    height: int
    width: int
    spectral_points: int


def discover_agilent_files(source: str | Path, recursive: bool = True) -> list[Path]:
    """Return readable Agilent FTIR files from a file or directory path."""
    source_path = Path(source)

    if source_path.is_file():
        if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported spectral file type: {source_path.suffix}")
        if not AgilentIRFile.isreadable(source_path):
            raise ValueError(f"Agilent file is not readable: {source_path}")
        return [source_path]

    if not source_path.is_dir():
        raise FileNotFoundError(f"Spectral source does not exist: {source_path}")

    return [Path(path) for path in AgilentIRFile.find_files(source_path, recursive=recursive)]


def export_spectral_cube_to_csv(
    intensities: np.ndarray,
    wavenumbers: np.ndarray,
    output_csv_path: str | Path,
) -> SpectralExportSummary:
    """Write a spectral cube to CSV with one row per image pixel."""
    if intensities.ndim != 3:
        raise ValueError("Expected intensities with shape (height, width, spectral_points).")

    height, width, spectral_points = intensities.shape
    if spectral_points != len(wavenumbers):
        raise ValueError(
            "The number of spectral intensity channels does not match the wavenumber axis."
        )

    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["Pixel_Y", "Pixel_X"]
    header.extend(f"Wavenumber_{float(wavenumber):.2f}" for wavenumber in wavenumbers)

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)

        for y in range(height):
            for x in range(width):
                writer.writerow([y, x, *intensities[y, x, :].tolist()])

    return SpectralExportSummary(
        source_file=None,
        output_csv=output_path,
        height=height,
        width=width,
        spectral_points=spectral_points,
    )


def extract_spectral_file(
    source_file: str | Path,
    output_dir: str | Path,
    output_name: str | None = None,
) -> SpectralExportSummary:
    """Extract one Agilent .dmt or .seq file to a pixel-indexed CSV file."""
    source_path = Path(source_file)
    if not source_path.is_file():
        raise FileNotFoundError(f"Spectral file does not exist: {source_path}")
    if not AgilentIRFile.isreadable(source_path):
        raise ValueError(f"Agilent file is not readable: {source_path}")

    reader = AgilentIRFile()
    reader.read(source_path)

    output_directory = Path(output_dir)
    csv_name = output_name or f"{source_path.stem}_spectra.csv"
    output_csv_path = output_directory / csv_name

    summary = export_spectral_cube_to_csv(
        intensities=reader.intensities,
        wavenumbers=reader.wavenumbers,
        output_csv_path=output_csv_path,
    )

    return SpectralExportSummary(
        source_file=source_path,
        output_csv=summary.output_csv,
        height=summary.height,
        width=summary.width,
        spectral_points=summary.spectral_points,
    )


def extract_spectral_files(
    source_files: Iterable[str | Path],
    output_dir: str | Path,
) -> list[SpectralExportSummary]:
    """Extract multiple Agilent files to CSV files in one output directory."""
    summaries: list[SpectralExportSummary] = []
    used_names: set[str] = set()

    for source_file in source_files:
        source_path = Path(source_file)
        output_name = f"{source_path.stem}_spectra.csv"
        if output_name in used_names:
            index = 2
            while f"{source_path.stem}_spectra_{index}.csv" in used_names:
                index += 1
            output_name = f"{source_path.stem}_spectra_{index}.csv"
        used_names.add(output_name)

        summaries.append(
            extract_spectral_file(
                source_file=source_path,
                output_dir=output_dir,
                output_name=output_name,
            )
        )

    return summaries
