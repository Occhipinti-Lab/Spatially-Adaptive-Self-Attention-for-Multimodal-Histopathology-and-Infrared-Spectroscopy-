# -*- coding: utf-8 -*-
"""Generate the full preprocessing figure plate.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


LABEL_MAP = {
    "malignant_epithelium": 0,
    "malignant_stroma": 1,
    "non_malignant_epithelium": 2,
    "non_malignant_stroma": 3,
}
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}

CLASS_COLORS = {
    0: "#6A5ACD",
    1: "#008080",
    2: "#CC7722",
    3: "#708090",
}

FIRST_SELECTION_RANGES = (
    (1000.0, 1319.0),
    (1481.0, 1769.0),
    (2986.0, 3569.0),
)

SECOND_SELECTION_RANGES = (
    (1019.0, 1300.0),
    (1500.0, 1750.0),
    (3005.0, 3550.0),
)

ROW_LABELS = ("Raw spectra", "Stage 1 Selection", "Stage 2 Selection")
PANEL_LETTERS = tuple("abcdefghijkl")
BYTES_PER_MB = 1024 * 1024
PDF_DPI_STEPS = (220, 180, 150, 120, 96)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the full preprocessing figure plate from spectral CSV files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing CSV files with Label and Wavenumber_* columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("preprocessing_plot"),
        help="Directory where full_preprocessing_figure PNG/PDF files will be saved.",
    )
    parser.add_argument(
        "--font-family",
        default="Arial",
        help="Plot font family. Defaults to Arial.",
    )
    parser.add_argument(
        "--max-pdf-mb",
        type=float,
        default=5.0,
        help="Maximum allowed size for the generated PDF.",
    )
    parser.add_argument(
        "--png-dpi",
        type=int,
        default=160,
        help="PNG export DPI. The default creates an image close to 2048 x 1365 px.",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=19,
        help="Savitzky-Golay filter window length. Must be odd.",
    )
    parser.add_argument(
        "--savgol-polyorder",
        type=int,
        default=4,
        help="Savitzky-Golay polynomial order.",
    )
    return parser.parse_args()


def configure_matplotlib(font_family: str) -> None:
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    if font_family not in available_fonts:
        print(
            f"WARNING: '{font_family}' was not found by Matplotlib. "
            "Matplotlib will use the first available fallback font."
        )

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 160,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
            "axes.linewidth": 0.8,
            "axes.unicode_minus": False,
        }
    )


def class_display_name(class_id: int) -> str:
    return REVERSE_LABEL_MAP[class_id].replace("_", " ").title()


def read_spectral_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    if "Label" not in df.columns:
        raise ValueError(f"{path} is missing the required 'Label' column.")

    spectral_columns: list[tuple[float, str]] = []
    for column in df.columns:
        if not column.startswith("Wavenumber_"):
            continue
        try:
            wavenumber = float(column.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Could not parse wavenumber from '{column}'.") from exc
        spectral_columns.append((wavenumber, column))

    if not spectral_columns:
        raise ValueError(f"{path} has no Wavenumber_* columns.")

    spectral_columns.sort(key=lambda item: item[0])
    wavenumbers = np.asarray([item[0] for item in spectral_columns], dtype=float)
    spectra = df[[item[1] for item in spectral_columns]].to_numpy(dtype=float)
    labels = df["Label"].astype(str).str.strip().to_numpy()
    return wavenumbers, spectra, labels


def load_all_spectra(data_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    csv_paths = sorted(data_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Add CSV files there or pass --data-dir."
        )

    combined: dict[int, list[np.ndarray]] = {class_id: [] for class_id in REVERSE_LABEL_MAP}
    master_wavenumbers: np.ndarray | None = None

    print("\nLoading CSV files...\n")
    for csv_path in csv_paths:
        print(f"Reading: {csv_path}")
        wavenumbers, spectra, labels = read_spectral_csv(csv_path)

        if master_wavenumbers is None:
            master_wavenumbers = wavenumbers
        elif len(wavenumbers) != len(master_wavenumbers) or not np.allclose(
            wavenumbers, master_wavenumbers
        ):
            raise ValueError(
                f"{csv_path} has a different wavenumber grid from the first CSV file."
            )

        numeric_labels = np.asarray([LABEL_MAP.get(label, -1) for label in labels])
        valid_rows = numeric_labels >= 0
        if not np.all(valid_rows):
            unknown_labels = sorted(set(labels[~valid_rows]))
            print(f"WARNING: skipping unmapped labels in {csv_path.name}: {unknown_labels}")

        spectra = spectra[valid_rows]
        numeric_labels = numeric_labels[valid_rows]
        for class_id in REVERSE_LABEL_MAP:
            class_spectra = spectra[numeric_labels == class_id]
            if class_spectra.shape[0] > 0:
                combined[class_id].append(class_spectra)

    assert master_wavenumbers is not None
    combined_arrays: dict[int, np.ndarray] = {}
    for class_id, class_arrays in combined.items():
        if class_arrays:
            combined_arrays[class_id] = np.vstack(class_arrays)
        else:
            combined_arrays[class_id] = np.empty((0, len(master_wavenumbers)))

    return master_wavenumbers, combined_arrays


def extract_ranges(
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    ranges: Iterable[tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    mask = np.zeros_like(wavenumbers, dtype=bool)
    for start, end in ranges:
        mask |= (wavenumbers >= start) & (wavenumbers <= end)

    if not np.any(mask):
        raise ValueError("The requested spectral ranges do not match any wavenumbers.")
    return wavenumbers[mask], spectra[:, mask]


def validate_savgol_settings(window_length: int, polyorder: int, n_features: int) -> None:
    if window_length % 2 == 0:
        raise ValueError("--savgol-window must be odd.")
    if polyorder >= window_length:
        raise ValueError("--savgol-polyorder must be smaller than --savgol-window.")
    if window_length > n_features:
        raise ValueError(
            f"--savgol-window ({window_length}) is larger than the selected spectra length "
            f"({n_features})."
        )


def apply_savgol_derivative(
    spectra_by_class: dict[int, np.ndarray],
    window_length: int,
    polyorder: int,
) -> dict[int, np.ndarray]:
    first_nonempty = next(
        (spectra for spectra in spectra_by_class.values() if spectra.shape[0] > 0),
        None,
    )
    if first_nonempty is not None:
        validate_savgol_settings(window_length, polyorder, first_nonempty.shape[1])

    derivative: dict[int, np.ndarray] = {}
    for class_id, spectra in spectra_by_class.items():
        if spectra.shape[0] == 0:
            derivative[class_id] = spectra
            continue
        derivative[class_id] = savgol_filter(
            spectra,
            window_length=window_length,
            polyorder=polyorder,
            deriv=1,
            axis=1,
        )
    return derivative


def make_fixed_ticks(wavenumbers: np.ndarray) -> np.ndarray:
    return np.linspace(float(np.min(wavenumbers)), float(np.max(wavenumbers)), 4)


def style_panel(
    ax: plt.Axes,
    row_index: int,
    column_index: int,
    wavenumbers: np.ndarray,
) -> None:
    row_limits = (
        (-0.11, 0.73),
        (-0.065, 0.305),
        (-0.010, 0.0085),
    )
    row_ticks = (
        (0.0, 0.2, 0.4, 0.6),
        (0.0, 0.1, 0.2, 0.3),
        (-0.005, 0.0, 0.005),
    )

    ax.set_xlim(float(np.min(wavenumbers)), float(np.max(wavenumbers)))
    ax.set_ylim(*row_limits[row_index])
    ax.set_yticks(row_ticks[row_index])

    ticks = make_fixed_ticks(wavenumbers)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(round(tick))) for tick in ticks])

    if column_index == 0:
        ax.set_ylabel("Intensity")
    else:
        ax.set_ylabel("")

    if row_index == 2:
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    else:
        ax.set_xlabel("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", length=3.5, width=0.8, pad=3)


def plot_spectra(
    ax: plt.Axes,
    wavenumbers: np.ndarray,
    spectra: np.ndarray,
    color: str,
) -> None:
    if spectra.shape[0] == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        return

    ax.plot(
        wavenumbers,
        spectra.T,
        color=color,
        alpha=0.16,
        linewidth=0.35,
        rasterized=True,
        zorder=2,
    )


def add_selection_bands(ax: plt.Axes, ranges: Iterable[tuple[float, float]]) -> None:
    x_min, x_max = ax.get_xlim()
    for start, end in ranges:
        band_start = max(start, x_min)
        band_end = min(end, x_max)
        if band_start < band_end:
            ax.axvspan(band_start, band_end, color="#eeeeee", alpha=0.8, zorder=0)


def plot_full_preprocessing_figure(
    raw_wavenumbers: np.ndarray,
    raw_spectra: dict[int, np.ndarray],
    stage1_wavenumbers: np.ndarray,
    stage1_spectra: dict[int, np.ndarray],
    stage2_wavenumbers: np.ndarray,
    stage2_spectra: dict[int, np.ndarray],
) -> plt.Figure:
    stage_wavenumbers = (raw_wavenumbers, stage1_wavenumbers, stage2_wavenumbers)
    stage_spectra = (raw_spectra, stage1_spectra, stage2_spectra)

    fig, axes = plt.subplots(3, 4, figsize=(12.8, 8.53))
    fig.subplots_adjust(
        left=0.095,
        right=0.99,
        top=0.965,
        bottom=0.075,
        wspace=0.30,
        hspace=0.43,
    )

    for row_index in range(3):
        for column_index, class_id in enumerate(sorted(REVERSE_LABEL_MAP)):
            ax = axes[row_index, column_index]
            panel_index = row_index * 4 + column_index
            panel_letter = PANEL_LETTERS[panel_index]

            style_panel(ax, row_index, column_index, stage_wavenumbers[row_index])
            if row_index == 2:
                add_selection_bands(ax, SECOND_SELECTION_RANGES)

            plot_spectra(
                ax,
                stage_wavenumbers[row_index],
                stage_spectra[row_index][class_id],
                CLASS_COLORS[class_id],
            )
            ax.set_title(
                f"({panel_letter}) {class_display_name(class_id)}",
                pad=7,
                fontweight="normal",
            )

    for row_index, row_label in enumerate(ROW_LABELS):
        row_box = axes[row_index, 0].get_position()
        y_center = row_box.y0 + row_box.height / 2
        fig.text(
            0.024,
            y_center,
            row_label,
            rotation=90,
            va="center",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    return fig


def save_light_pdf(
    fig: plt.Figure,
    pdf_path: Path,
    max_pdf_mb: float,
    dpi_steps: Iterable[int] = PDF_DPI_STEPS,
) -> None:
    max_bytes = int(max_pdf_mb * BYTES_PER_MB)
    last_size = 0
    last_dpi = 0

    for dpi in dpi_steps:
        fig.savefig(pdf_path, format="pdf", dpi=dpi, bbox_inches="tight", facecolor="white")
        last_size = pdf_path.stat().st_size
        last_dpi = dpi
        if last_size <= max_bytes:
            print(f"Saved {pdf_path.name} ({last_size / BYTES_PER_MB:.2f} MB, {dpi} dpi)")
            return

    raise RuntimeError(
        f"{pdf_path} is {last_size / BYTES_PER_MB:.2f} MB at {last_dpi} dpi, "
        f"which is above the {max_pdf_mb:.1f} MB limit."
    )


def save_full_figure(
    fig: plt.Figure,
    output_dir: Path,
    png_dpi: int,
    max_pdf_mb: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "full_preprocessing_figure.png"
    pdf_path = output_dir / "full_preprocessing_figure.pdf"

    fig.savefig(png_path, format="png", dpi=png_dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved {png_path}")
    save_light_pdf(fig, pdf_path, max_pdf_mb=max_pdf_mb)


def build_preprocessing_stages(
    data_dir: Path,
    savgol_window: int,
    savgol_polyorder: int,
) -> tuple[
    np.ndarray,
    dict[int, np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
    np.ndarray,
    dict[int, np.ndarray],
]:
    raw_wavenumbers, raw_spectra = load_all_spectra(data_dir)

    stage1_spectra: dict[int, np.ndarray] = {}
    stage1_wavenumbers: np.ndarray | None = None
    for class_id, spectra in raw_spectra.items():
        stage1_wavenumbers, stage1_spectra[class_id] = extract_ranges(
            raw_wavenumbers, spectra, FIRST_SELECTION_RANGES
        )

    assert stage1_wavenumbers is not None
    derivative_spectra = apply_savgol_derivative(
        stage1_spectra,
        window_length=savgol_window,
        polyorder=savgol_polyorder,
    )

    stage2_spectra: dict[int, np.ndarray] = {}
    stage2_wavenumbers: np.ndarray | None = None
    for class_id, spectra in derivative_spectra.items():
        stage2_wavenumbers, stage2_spectra[class_id] = extract_ranges(
            stage1_wavenumbers, spectra, SECOND_SELECTION_RANGES
        )

    assert stage2_wavenumbers is not None
    return (
        raw_wavenumbers,
        raw_spectra,
        stage1_wavenumbers,
        stage1_spectra,
        stage2_wavenumbers,
        stage2_spectra,
    )


def run_pipeline(args: argparse.Namespace) -> None:
    if args.max_pdf_mb <= 0:
        raise ValueError("--max-pdf-mb must be greater than zero.")
    if args.png_dpi <= 0:
        raise ValueError("--png-dpi must be greater than zero.")

    configure_matplotlib(args.font_family)
    stages = build_preprocessing_stages(
        args.data_dir,
        savgol_window=args.savgol_window,
        savgol_polyorder=args.savgol_polyorder,
    )
    fig = plot_full_preprocessing_figure(*stages)
    save_full_figure(
        fig,
        output_dir=args.output_dir,
        png_dpi=args.png_dpi,
        max_pdf_mb=args.max_pdf_mb,
    )
    plt.close(fig)
    print(f"\nAll plots saved in: {args.output_dir}")


def main() -> None:
    run_pipeline(parse_args())


if __name__ == "__main__":
    main()
