from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from PIL import Image


DEFAULT_LABEL_COLORS: dict[tuple[int, int, int], str] = {
    (255, 0, 0): "malignant_epithelium",
    (103, 193, 66): "non_malignant_epithelium",
    (243, 143, 53): "non_malignant_stroma",
    (163, 55, 156): "malignant_stroma",
}

DEFAULT_IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg")
SPECTRA_SUFFIXES = ("_spectra", "_spectral_data", "_3d_spectral_data")


@dataclass(frozen=True)
class AnnotationSummary:
    input_csv: Path
    annotation_image: Path
    output_csv: Path
    input_rows: int
    labeled_rows: int


@dataclass(frozen=True)
class BatchAnnotationSummary:
    processed: list[AnnotationSummary]
    missing_annotations: list[Path]


def _color_code(rgb: tuple[int, int, int] | np.ndarray) -> int:
    red, green, blue = [int(value) for value in rgb]
    return (red << 16) + (green << 8) + blue


def _sample_key(path: Path) -> str:
    stem = path.stem
    for suffix in SPECTRA_SUFFIXES:
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)].lower()
    return stem.lower()


def find_annotation_images(
    annotation_dir: str | Path,
    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
) -> dict[str, Path]:
    """Index annotation images by their base filename."""
    directory = Path(annotation_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Annotation directory does not exist: {directory}")

    extensions = {extension.lower() for extension in image_extensions}
    images: dict[str, Path] = {}
    for image_path in directory.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in extensions:
            images[image_path.stem.lower()] = image_path
    return images


def match_annotation_image(csv_path: str | Path, annotation_images: Mapping[str, Path]) -> Path | None:
    """Find the annotation image corresponding to a spectra CSV file."""
    path = Path(csv_path)
    direct_key = path.stem.lower()
    sample_key = _sample_key(path)
    return annotation_images.get(direct_key) or annotation_images.get(sample_key)


def label_spectral_csv(
    data_csv_path: str | Path,
    annotation_image_path: str | Path,
    output_csv_path: str | Path,
    label_colors: Mapping[tuple[int, int, int], str] = DEFAULT_LABEL_COLORS,
    color_tolerance: int = 0,
) -> AnnotationSummary:
    """Assign pathology labels to spectral rows using Pixel_X and Pixel_Y coordinates."""
    if color_tolerance < 0:
        raise ValueError("color_tolerance must be zero or greater.")
    if not label_colors:
        raise ValueError("At least one label color is required.")

    data_path = Path(data_csv_path)
    image_path = Path(annotation_image_path)
    output_path = Path(output_csv_path)

    data = pd.read_csv(data_path)
    missing_columns = {"Pixel_X", "Pixel_Y"} - set(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required coordinate column(s): {missing}")

    coordinates = data[["Pixel_Y", "Pixel_X"]].apply(pd.to_numeric, errors="raise")
    coordinate_values = coordinates.to_numpy(dtype=float)
    if not np.isfinite(coordinate_values).all():
        raise ValueError("Pixel_X and Pixel_Y must contain finite numeric coordinates.")
    if not np.equal(coordinate_values, np.rint(coordinate_values)).all():
        raise ValueError("Pixel_X and Pixel_Y must contain integer pixel coordinates.")

    y_values = coordinate_values[:, 0].astype(np.int64)
    x_values = coordinate_values[:, 1].astype(np.int64)

    annotation = np.asarray(Image.open(image_path).convert("RGB"))
    height, width = annotation.shape[:2]
    within_bounds = (
        (y_values >= 0)
        & (y_values < height)
        & (x_values >= 0)
        & (x_values < width)
    )

    labels = np.full(len(data), None, dtype=object)
    valid_indices = np.flatnonzero(within_bounds)

    if len(valid_indices):
        pixels = annotation[y_values[valid_indices], x_values[valid_indices], :]
        labels[valid_indices] = _labels_for_pixels(
            pixels=pixels,
            label_colors=label_colors,
            color_tolerance=color_tolerance,
        )

    matched = pd.notna(labels)
    labeled_data = data.loc[matched].copy()
    labeled_data["Label"] = labels[matched]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labeled_data.to_csv(output_path, index=False)

    return AnnotationSummary(
        input_csv=data_path,
        annotation_image=image_path,
        output_csv=output_path,
        input_rows=len(data),
        labeled_rows=len(labeled_data),
    )


def _labels_for_pixels(
    pixels: np.ndarray,
    label_colors: Mapping[tuple[int, int, int], str],
    color_tolerance: int,
) -> np.ndarray:
    if color_tolerance == 0:
        lookup = {_color_code(color): label for color, label in label_colors.items()}
        pixel_values = pixels.astype(np.int64)
        codes = (pixel_values[:, 0] << 16) + (pixel_values[:, 1] << 8) + pixel_values[:, 2]
        return pd.Series(codes).map(lookup).to_numpy(dtype=object)

    palette = np.asarray(list(label_colors.keys()), dtype=np.int16)
    label_names = np.asarray(list(label_colors.values()), dtype=object)
    differences = np.abs(pixels.astype(np.int16)[:, None, :] - palette[None, :, :])
    distances = np.max(differences, axis=2)
    best_indices = np.argmin(distances, axis=1)
    best_distances = distances[np.arange(len(pixels)), best_indices]

    labels = np.full(len(pixels), None, dtype=object)
    within_tolerance = best_distances <= color_tolerance
    labels[within_tolerance] = label_names[best_indices[within_tolerance]]
    return labels


def label_spectral_directory(
    data_dir: str | Path,
    annotation_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.csv",
    image_extensions: Sequence[str] = DEFAULT_IMAGE_EXTENSIONS,
    color_tolerance: int = 0,
) -> BatchAnnotationSummary:
    """Label every spectra CSV in a directory when a matching annotation image exists."""
    data_directory = Path(data_dir)
    if not data_directory.is_dir():
        raise FileNotFoundError(f"Spectra directory does not exist: {data_directory}")

    output_directory = Path(output_dir)
    annotation_images = find_annotation_images(annotation_dir, image_extensions=image_extensions)

    processed: list[AnnotationSummary] = []
    missing_annotations: list[Path] = []

    for csv_path in sorted(data_directory.glob(pattern)):
        annotation_image = match_annotation_image(csv_path, annotation_images)
        if annotation_image is None:
            missing_annotations.append(csv_path)
            continue

        output_csv = output_directory / f"{_sample_key(csv_path)}_labeled_spectra.csv"
        processed.append(
            label_spectral_csv(
                data_csv_path=csv_path,
                annotation_image_path=annotation_image,
                output_csv_path=output_csv,
                color_tolerance=color_tolerance,
            )
        )

    return BatchAnnotationSummary(
        processed=processed,
        missing_annotations=missing_annotations,
    )
