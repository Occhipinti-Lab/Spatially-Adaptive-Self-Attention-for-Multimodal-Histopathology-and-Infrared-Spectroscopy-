from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SpectralDataset:
    source_path: Path
    source_file: str
    frame: pd.DataFrame
    spectral_columns: list[str]
    wavenumbers: np.ndarray
    label_column: str

    @property
    def spectra(self) -> np.ndarray:
        return self.frame[self.spectral_columns].to_numpy(dtype=float)

    @property
    def labels(self) -> pd.Series:
        return self.frame[self.label_column].astype(str)


def find_csv_files(data_dir: Path, recursive: bool = False) -> list[Path]:
    data_dir = data_dir.expanduser().resolve()
    if data_dir.is_file():
        if data_dir.suffix.lower() != ".csv":
            raise ValueError(f"Expected a CSV file, got: {data_dir}")
        return [data_dir]

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def load_spectral_csv(
    csv_path: Path,
    label_column: str = "Label",
    spectral_prefix: str = "Wavenumber_",
) -> SpectralDataset:
    csv_path = csv_path.expanduser().resolve()
    frame = pd.read_csv(csv_path)

    if label_column not in frame.columns:
        raise ValueError(f"{csv_path.name} does not contain label column '{label_column}'.")

    spectral_columns, wavenumbers = extract_spectral_axis(frame, spectral_prefix)
    if not spectral_columns:
        raise ValueError(
            f"{csv_path.name} does not contain columns beginning with '{spectral_prefix}'."
        )

    frame = frame.copy()
    frame[spectral_columns] = frame[spectral_columns].apply(pd.to_numeric, errors="coerce")

    return SpectralDataset(
        source_path=csv_path,
        source_file=csv_path.name,
        frame=frame,
        spectral_columns=spectral_columns,
        wavenumbers=wavenumbers,
        label_column=label_column,
    )


def extract_spectral_axis(
    frame: pd.DataFrame,
    spectral_prefix: str,
) -> tuple[list[str], np.ndarray]:
    parsed_columns: list[tuple[float, str]] = []

    for column in frame.columns:
        if not column.startswith(spectral_prefix):
            continue
        raw_value = column.removeprefix(spectral_prefix)
        try:
            parsed_columns.append((float(raw_value), column))
        except ValueError as exc:
            raise ValueError(f"Could not parse wavenumber from column '{column}'.") from exc

    parsed_columns.sort(key=lambda item: item[0])
    wavenumbers = np.array([item[0] for item in parsed_columns], dtype=float)
    spectral_columns = [item[1] for item in parsed_columns]
    return spectral_columns, wavenumbers

