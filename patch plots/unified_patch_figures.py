import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.size"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 16
plt.rcParams["ytick.labelsize"] = 16
plt.rcParams["legend.fontsize"] = 16
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# ============================================================
# DEFAULT SETTINGS
# ============================================================
PATCH_SIZE = 16
MIN_PIXELS = 1
N_SHOW = 4

SPECTRUM_MODE = "mean_valid_pixels"
PATCH_DISPLAY_MODE = "masked"
DEFAULT_OUTPUT_DIR = "unified_patch_figures"
DEFAULT_IMAGE_CANDIDATES = (
    "images/A7.jpg",
    "A7.jpg",
    "../images/A7.jpg",
)
DEFAULT_CSV_CANDIDATES = (
    "data/A7_filtered.csv",
    "A7_filtered.csv",
    "../data/A7_filtered.csv",
)

FINAL_RANGES = [(1019, 1300), (1500, 1750), (3005, 3550)]

LABEL_MAP = {
    "malignant_epithelium": 0,
    "malignant_stroma": 1,
    "non_malignant_epithelium": 2,
    "non_malignant_stroma": 3,
}
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}

PALETTE = [
    "#5E3C99",
    "#008B8B",
    "#4C78A8",
    "#D17C1B",
    "#C2185B",
    "#556B2F",
    "#1E88E5",
    "#00ACC1",
    "#B8860B",
    "#8D6E63",
]

# ============================================================
# TIGHTER LAYOUT SETTINGS
# ============================================================
OUTER_MARGIN_X = 0.007
OUTER_MARGIN_Y = 0.007
INNER_GAP_X = 0.005
INNER_GAP_Y = 0.005

HE_WIDTH_FRACTION = 0.48
BASE_FIGURE_HEIGHT_INCHES = 11.5
EXPORT_PAD_INCHES = 0.00

PatchRecord = Dict[str, object]
LayoutBox = Tuple[float, float, float, float]
BBox = Tuple[int, int, int, int]


def existing_path_or_none(path: Path) -> Optional[Path]:
    return path.resolve() if path.exists() else None


def resolve_input_path(
    explicit_path: Optional[str],
    candidates: Sequence[str],
    description: str,
) -> Path:
    if explicit_path:
        resolved = existing_path_or_none(Path(explicit_path).expanduser())
        if resolved is not None:
            return resolved
        raise FileNotFoundError(f"{description} not found: {explicit_path}")

    script_dir = Path(__file__).resolve().parent
    search_roots = (Path.cwd(), script_dir, script_dir.parent)
    checked_paths: List[Path] = []

    for root in search_roots:
        for candidate in candidates:
            candidate_path = root / candidate
            checked_paths.append(candidate_path)
            resolved = existing_path_or_none(candidate_path)
            if resolved is not None:
                return resolved

    checked = "\n  - ".join(str(path) for path in checked_paths)
    raise FileNotFoundError(
        f"{description} was not found. Pass it explicitly with the CLI.\n"
        f"Checked:\n  - {checked}"
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate H&E patch overview and compact co-registration figures "
            "from an image plus a spectral CSV."
        )
    )
    parser.add_argument(
        "--image",
        help="Path to the H&E image, for example A7.jpg.",
    )
    parser.add_argument(
        "--csv",
        help="Path to the spectral CSV, for example A7_filtered.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated PNG/PDF outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=PATCH_SIZE,
        help=f"Patch size in pixels. Default: {PATCH_SIZE}",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=MIN_PIXELS,
        help=f"Minimum valid pixels required for a patch. Default: {MIN_PIXELS}",
    )
    parser.add_argument(
        "--n-show",
        type=int,
        default=N_SHOW,
        help=f"Number of spatially spread patches to show. Default: {N_SHOW}",
    )
    parser.add_argument(
        "--spectrum-mode",
        choices=("seed_pixel", "mean_valid_pixels"),
        default=SPECTRUM_MODE,
        help=f"Spectrum plotted for each patch. Default: {SPECTRUM_MODE}",
    )
    parser.add_argument(
        "--patch-display-mode",
        choices=("masked", "context"),
        default=PATCH_DISPLAY_MODE,
        help=f"Patch crop display mode. Default: {PATCH_DISPLAY_MODE}",
    )
    return parser.parse_args(argv)

# ============================================================
# FILE AND DATA HELPERS
# ============================================================
def save_figure_both(
    fig: plt.Figure,
    base_path_no_ext: str,
    dpi: int = 600,
) -> Tuple[str, str]:
    png_path = f"{base_path_no_ext}.png"
    pdf_path = f"{base_path_no_ext}.pdf"

    fig.savefig(
        png_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
        facecolor="white",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=EXPORT_PAD_INCHES,
        facecolor="white",
    )
    return png_path, pdf_path


def load_image(path: str) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image)


def load_csv_and_prepare_features(
    csv_path: str,
    image_shape: Tuple[int, ...],
    label_map: Dict[str, int],
    final_ranges: Sequence[Tuple[float, float]],
) -> Tuple[
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    df = pd.read_csv(csv_path)

    required_cols = {"Pixel_X", "Pixel_Y", "Label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    image_height, image_width = image_shape[:2]

    pixel_x = np.rint(df["Pixel_X"].to_numpy()).astype(int)
    pixel_y = np.rint(df["Pixel_Y"].to_numpy()).astype(int)

    valid_xy = (
        (pixel_x >= 0)
        & (pixel_x < image_width)
        & (pixel_y >= 0)
        & (pixel_y < image_height)
    )

    df = df.loc[valid_xy].copy()
    pixel_x = pixel_x[valid_xy]
    pixel_y = pixel_y[valid_xy]

    mapped_labels = df["Label"].map(label_map)
    if mapped_labels.isna().any():
        unknown_labels = (
            df.loc[mapped_labels.isna(), "Label"].dropna().unique().tolist()
        )
        raise ValueError(f"Unknown labels found in CSV: {unknown_labels}")

    pixel_labels = mapped_labels.to_numpy(dtype=int)

    spectral_columns: List[str] = []
    spectral_wavenumbers: List[float] = []

    for column in df.columns:
        if not column.startswith("Wavenumber_"):
            continue

        try:
            wavenumber = float(column.split("_", 1)[1])
        except ValueError:
            continue

        if any(low <= wavenumber <= high for low, high in final_ranges):
            spectral_columns.append(column)
            spectral_wavenumbers.append(wavenumber)

    if not spectral_columns:
        raise ValueError("No Wavenumber_* columns were found in FINAL_RANGES.")

    order = np.argsort(spectral_wavenumbers)
    sorted_wavenumbers = np.asarray(spectral_wavenumbers, dtype=float)[order]
    sorted_columns = [spectral_columns[index] for index in order]
    spectral_matrix = df[sorted_columns].to_numpy(dtype=np.float32)

    return (
        df,
        pixel_x,
        pixel_y,
        pixel_labels,
        sorted_wavenumbers,
        spectral_matrix,
    )


def build_dense_grids(
    image_shape: Tuple[int, ...],
    pixel_x: np.ndarray,
    pixel_y: np.ndarray,
    pixel_labels: np.ndarray,
    spectral_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    image_height, image_width = image_shape[:2]
    spectral_depth = spectral_matrix.shape[1]

    spectra_grid = np.zeros(
        (image_height, image_width, spectral_depth),
        dtype=np.float32,
    )
    label_grid = np.full(
        (image_height, image_width),
        fill_value=-1,
        dtype=np.int32,
    )
    present_grid = np.zeros(
        (image_height, image_width),
        dtype=np.uint8,
    )

    for x_coord, y_coord, label, spectrum in zip(
        pixel_x,
        pixel_y,
        pixel_labels,
        spectral_matrix,
    ):
        spectra_grid[y_coord, x_coord, :] = spectrum
        label_grid[y_coord, x_coord] = label
        present_grid[y_coord, x_coord] = 1

    return spectra_grid, label_grid, present_grid


# ============================================================
# PATCH EXTRACTION (unchanged)
# ============================================================
def extract_exact_patches(
    image_rgb: np.ndarray,
    spectra_grid: np.ndarray,
    label_grid: np.ndarray,
    present_grid: np.ndarray,
    reverse_label_map: Dict[int, str],
    patch_size: int = 32,
    min_pixels: int = 1,
) -> List[PatchRecord]:
    image_height, image_width = label_grid.shape
    half_size = patch_size // 2

    used = np.zeros((image_height, image_width), dtype=np.uint8)
    patches: List[PatchRecord] = []

    for row in range(image_height):
        for column in range(image_width):
            seed_label = label_grid[row, column]

            if seed_label < 0:
                continue
            if present_grid[row, column] == 0:
                continue
            if used[row, column] == 1:
                continue

            x1 = max(0, column - half_size)
            x2 = min(image_width - 1, column + half_size)
            y1 = max(0, row - half_size)
            y2 = min(image_height - 1, row + half_size)

            crop_rgb = image_rgb[y1 : y2 + 1, x1 : x2 + 1].copy()
            crop_height = y2 - y1 + 1
            crop_width = x2 - x1 + 1

            local_mask = np.zeros((crop_height, crop_width), dtype=np.uint8)
            valid_global_coords: List[Tuple[int, int]] = []
            valid_local_coords: List[Tuple[int, int]] = []

            for global_y in range(y1, y2 + 1):
                for global_x in range(x1, x2 + 1):
                    is_valid = (
                        label_grid[global_y, global_x] == seed_label
                        and present_grid[global_y, global_x] == 1
                        and used[global_y, global_x] == 0
                    )
                    if not is_valid:
                        continue

                    local_y = global_y - y1
                    local_x = global_x - x1
                    local_mask[local_y, local_x] = 1
                    valid_global_coords.append((global_x, global_y))
                    valid_local_coords.append((local_x, local_y))

            if len(valid_global_coords) < min_pixels:
                continue

            for global_x, global_y in valid_global_coords:
                used[global_y, global_x] = 1

            masked_patch_rgb = np.zeros_like(crop_rgb)
            masked_patch_rgb[local_mask == 1] = crop_rgb[local_mask == 1]

            valid_spectra = np.asarray(
                [
                    spectra_grid[global_y, global_x, :]
                    for global_x, global_y in valid_global_coords
                ],
                dtype=np.float32,
            )

            seed_spectrum = spectra_grid[row, column, :].copy()
            mean_spectrum = valid_spectra.mean(axis=0)

            patch_record: PatchRecord = {
                "seed_x": column,
                "seed_y": row,
                "seed_local_x": column - x1,
                "seed_local_y": row - y1,
                "label": int(seed_label),
                "label_name": reverse_label_map[int(seed_label)],
                "bbox": (x1, y1, x2, y2),
                "crop_rgb": crop_rgb,
                "masked_patch_rgb": masked_patch_rgb,
                "local_mask": local_mask,
                "valid_global_coords": valid_global_coords,
                "valid_local_coords": valid_local_coords,
                "valid_pixel_count": len(valid_global_coords),
                "seed_spectrum": seed_spectrum,
                "mean_spectrum": mean_spectrum,
            }
            patches.append(patch_record)

    return patches


def select_evenly_spread_patches(
    patch_list: Sequence[PatchRecord],
    n_show: int,
) -> List[PatchRecord]:
    if not patch_list:
        return []

    x_coords = np.asarray(
        [float(patch["seed_x"]) for patch in patch_list],
        dtype=float,
    )
    y_coords = np.asarray(
        [float(patch["seed_y"]) for patch in patch_list],
        dtype=float,
    )

    centre_x = x_coords.mean()
    centre_y = y_coords.mean()

    angles = np.arctan2(y_coords - centre_y, x_coords - centre_x)
    order = np.argsort(angles)

    if len(order) <= n_show:
        return [patch_list[index] for index in order]

    sampled_indices = np.linspace(
        0,
        len(order) - 1,
        n_show,
        dtype=int,
    )
    return [patch_list[order[index]] for index in sampled_indices]


def get_patch_display_image(
    patch: PatchRecord,
    patch_display_mode: str,
) -> np.ndarray:
    if patch_display_mode == "masked":
        return np.asarray(patch["masked_patch_rgb"])
    if patch_display_mode == "context":
        return np.asarray(patch["crop_rgb"])
    raise ValueError(
        "PATCH_DISPLAY_MODE must be either 'masked' or 'context'."
    )


def get_patch_spectrum(
    patch: PatchRecord,
    spectrum_mode: str,
) -> Tuple[np.ndarray, str]:
    if spectrum_mode == "seed_pixel":
        return np.asarray(patch["seed_spectrum"]), "Seed-pixel spectrum"
    if spectrum_mode == "mean_valid_pixels":
        return np.asarray(patch["mean_spectrum"]), "Mean patch spectrum"
    raise ValueError(
        "SPECTRUM_MODE must be either 'seed_pixel' or 'mean_valid_pixels'."
    )


# ============================================================
# LAYOUT HELPERS - TIGHTER
# ============================================================
def build_four_patch_layout_boxes() -> Tuple[
    LayoutBox,
    List[LayoutBox],
]:
    usable_width = 1.0 - (2.0 * OUTER_MARGIN_X) - INNER_GAP_X
    usable_height = 1.0 - (2.0 * OUTER_MARGIN_Y) - (2.0 * INNER_GAP_Y)

    left_width = usable_width * HE_WIDTH_FRACTION
    right_width = usable_width - left_width

    row_height = usable_height / 3.0

    left_x = OUTER_MARGIN_X
    right_x = left_x + left_width + INNER_GAP_X

    bottom_y = OUTER_MARGIN_Y
    middle_y = bottom_y + row_height + INNER_GAP_Y
    top_y = middle_y + row_height + INNER_GAP_Y

    he_box = (
        left_x,
        middle_y,
        left_width,
        (2.0 * row_height) + INNER_GAP_Y,
    )

    card_boxes = [
        (right_x, top_y, right_width, row_height),
        (right_x, middle_y, right_width, row_height),
        (left_x, bottom_y, left_width, row_height),   # Patch 3 now uses full left width
        (right_x, bottom_y, right_width, row_height),
    ]

    return he_box, card_boxes


def calculate_figure_size(
    image_rgb: np.ndarray,
    he_box: LayoutBox,
) -> Tuple[float, float]:
    image_height, image_width = image_rgb.shape[:2]
    image_aspect = image_width / image_height

    _, _, he_width_fraction, he_height_fraction = he_box
    figure_aspect = image_aspect * (he_height_fraction / he_width_fraction)
    figure_height = BASE_FIGURE_HEIGHT_INCHES
    figure_width = figure_height * figure_aspect

    return figure_width, figure_height


def calculate_square_axes_size(
    fig: plt.Figure,
    desired_width_fraction: float,
    maximum_height_fraction: float,
) -> Tuple[float, float]:
    figure_width, figure_height = fig.get_size_inches()
    square_height_fraction = desired_width_fraction * figure_width / figure_height

    if square_height_fraction <= maximum_height_fraction:
        return desired_width_fraction, square_height_fraction

    square_width_fraction = maximum_height_fraction * figure_height / figure_width
    return square_width_fraction, maximum_height_fraction


def get_patch_bbox(patch: PatchRecord) -> BBox:
    x1, y1, x2, y2 = patch["bbox"]
    return int(x1), int(y1), int(x2), int(y2)


def get_patch_seed(patch: PatchRecord) -> Tuple[int, int]:
    return int(patch["seed_x"]), int(patch["seed_y"])


def bbox_dimensions(bbox: BBox) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return x2 - x1 + 1, y2 - y1 + 1


def bbox_center(bbox: BBox) -> Tuple[float, float]:
    x1, y1, _, _ = bbox
    width, height = bbox_dimensions(bbox)
    return x1 + (width / 2.0), y1 + (height / 2.0)


def add_patch_bbox(
    ax: plt.Axes,
    bbox: BBox,
    color: str,
    linewidth: float,
    zorder: int,
) -> None:
    x1, y1, _, _ = bbox
    width, height = bbox_dimensions(bbox)
    ax.add_patch(
        Rectangle(
            (x1, y1),
            width,
            height,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            zorder=zorder,
        )
    )


def add_seed_marker(
    ax: plt.Axes,
    x_coord: float,
    y_coord: float,
    color: str,
    size: float,
    linewidth: float,
    zorder: int,
) -> None:
    ax.scatter(
        x_coord,
        y_coord,
        s=size,
        c=[color],
        edgecolors="white",
        linewidths=linewidth,
        zorder=zorder,
    )


def style_spines(
    ax: plt.Axes,
    color: str,
    linewidth: float,
    visible: Optional[bool] = None,
) -> None:
    for spine in ax.spines.values():
        if visible is not None:
            spine.set_visible(visible)
        spine.set_linewidth(linewidth)
        spine.set_color(color)


def compute_common_card_content_geometry(
    fig: plt.Figure,
    card_boxes: Sequence[LayoutBox],
) -> Dict[str, float]:
    if not card_boxes:
        raise ValueError("At least one card box is required.")

    # Use the smallest card width to ensure uniform tight layout across all patches
    min_card_width = min(width for _, _, width, _ in card_boxes)
    min_card_height = min(height for _, _, _, height in card_boxes)

    # Tight layout - matching Patch 3 style
    content_bottom_fraction = 0.14
    content_top_fraction = 0.79
    content_height = (content_top_fraction - content_bottom_fraction) * min_card_height

    patch_left_margin = 0.022 * min_card_width
    spectrum_right_margin = 0.022 * min_card_width
    label_gutter = 0.135 * min_card_width

    # Bigger patch relative to card to reduce white space
    desired_patch_width = 0.37 * min_card_width
    maximum_patch_height = 0.88 * content_height

    patch_width, patch_height = calculate_square_axes_size(
        fig=fig,
        desired_width_fraction=desired_patch_width,
        maximum_height_fraction=maximum_patch_height,
    )

    available_spectrum_width = (
        min_card_width
        - patch_left_margin
        - patch_width
        - label_gutter
        - spectrum_right_margin
    )

    spectrum_width, spectrum_height = calculate_square_axes_size(
        fig=fig,
        desired_width_fraction=available_spectrum_width,
        maximum_height_fraction=0.88 * content_height,
    )

    group_width = patch_width + label_gutter + spectrum_width

    return {
        "content_bottom_fraction": content_bottom_fraction,
        "content_top_fraction": content_top_fraction,
        "patch_width": patch_width,
        "patch_height": patch_height,
        "spectrum_width": spectrum_width,
        "spectrum_height": spectrum_height,
        "label_gutter": label_gutter,
        "group_width": group_width,
    }


def compute_common_spectrum_limits(
    selected_patches: Sequence[PatchRecord],
    spectral_wavenumbers: np.ndarray,
    spectrum_mode: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    finite_x = spectral_wavenumbers[np.isfinite(spectral_wavenumbers)]
    if finite_x.size == 0:
        raise ValueError("No finite spectral wavenumbers are available.")

    all_finite_y: List[np.ndarray] = []

    for patch in selected_patches:
        spectrum, _ = get_patch_spectrum(patch, spectrum_mode)
        finite_mask = np.isfinite(spectral_wavenumbers) & np.isfinite(spectrum)
        if np.any(finite_mask):
            all_finite_y.append(spectrum[finite_mask])

    if not all_finite_y:
        raise ValueError("No finite spectral intensities are available.")

    combined_y = np.concatenate(all_finite_y)

    x_min = float(np.min(finite_x))
    x_max = float(np.max(finite_x))
    y_min = float(np.min(combined_y))
    y_max = float(np.max(combined_y))

    x_span = x_max - x_min
    y_span = y_max - y_min

    x_pad = 0.012 * x_span if x_span > 0 else 1.0
    y_pad = 0.060 * y_span if y_span > 0 else max(abs(y_max) * 0.05, 1.0)

    common_x_limits = (x_min - x_pad, x_max + x_pad)
    common_y_limits = (y_min - y_pad, y_max + y_pad)

    return common_x_limits, common_y_limits


def style_spectrum_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.18, linewidth=0.50, zorder=0)

    ax.tick_params(
        axis="x",
        which="major",
        labelsize=16,
        length=5.0,
        width=1.1,
        pad=5.0,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=16,
        length=5.0,
        width=1.1,
        pad=3.0,
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))

    style_spines(ax, color="#4A4A4A", linewidth=1.1)


def format_class_name(label_name: str, width: int = 30) -> str:
    readable = label_name.replace("_", " ")
    return textwrap.fill(readable, width=width)


def add_card_background(
    fig: plt.Figure,
    card_box: LayoutBox,
    color: str,
) -> None:
    left, bottom, width, height = card_box

    card = Rectangle(
        (left, bottom),
        width,
        height,
        transform=fig.transFigure,
        facecolor="white",
        edgecolor="#CFCFCF",
        linewidth=0.75,
        zorder=-5,
        clip_on=False,
    )
    fig.add_artist(card)

    accent_height = min(0.006, 0.022 * height)
    accent = Rectangle(
        (left, bottom + height - accent_height),
        width,
        accent_height,
        transform=fig.transFigure,
        facecolor=color,
        edgecolor="none",
        zorder=-4,
        clip_on=False,
    )
    fig.add_artist(accent)


def add_patch_spectrum_card(
    fig: plt.Figure,
    patch: PatchRecord,
    spectral_wavenumbers: np.ndarray,
    color: str,
    card_box: LayoutBox,
    patch_index: int,
    spectrum_mode: str,
    patch_display_mode: str,
    common_x_limits: Tuple[float, float],
    common_y_limits: Tuple[float, float],
    shared_geometry: Dict[str, float],
) -> None:
    left, bottom, width, height = card_box
    add_card_background(fig, card_box, color)

    seed_x, seed_y = get_patch_seed(patch)
    label_text = format_class_name(str(patch["label_name"]), width=30)

    fig.text(
        left + (0.025 * width),
        bottom + (0.975 * height),
        f"Patch {patch_index} | ({seed_x}, {seed_y})",
        ha="left",
        va="top",
        fontsize=20,
        fontweight=None,
        color="#202020",
        zorder=1,
    )

    fig.text(
        left + (0.025 * width),
        bottom + (0.875 * height),
        label_text,
        ha="left",
        va="top",
        fontsize=16,
        color="#3A3A3A",
        linespacing=0.96,
        zorder=1,
    )

    content_bottom = bottom + (shared_geometry["content_bottom_fraction"] * height)
    content_top = bottom + (shared_geometry["content_top_fraction"] * height)
    content_height = content_top - content_bottom

    patch_width = shared_geometry["patch_width"]
    patch_height = shared_geometry["patch_height"]
    spectrum_width = shared_geometry["spectrum_width"]
    spectrum_height = shared_geometry["spectrum_height"]
    label_gutter = shared_geometry["label_gutter"]
    group_width = shared_geometry["group_width"]

    if group_width > width:
        raise ValueError(
            "The shared card content width is larger than one of the card boxes."
        )

    group_left = left + ((width - group_width) / 2.0)
    patch_left = group_left
    patch_bottom = content_bottom + ((content_height - patch_height) / 2.0)
    spectrum_left = patch_left + patch_width + label_gutter
    spectrum_bottom = content_bottom + ((content_height - spectrum_height) / 2.0)

    ax_patch = fig.add_axes(
        [patch_left, patch_bottom, patch_width, patch_height],
        zorder=3,
    )
    patch_image = get_patch_display_image(patch, patch_display_mode)
    ax_patch.imshow(
        patch_image,
        interpolation="nearest",
        aspect="equal",
    )

    if patch_display_mode == "context":
        local_mask = np.asarray(patch["local_mask"])
        mask_y, mask_x = np.where(local_mask == 1)
        ax_patch.scatter(
            mask_x,
            mask_y,
            s=4.5,
            c=[color],
            alpha=0.58,
            linewidths=0,
            zorder=4,
        )

    add_seed_marker(
        ax_patch,
        float(patch["seed_local_x"]),
        float(patch["seed_local_y"]),
        color,
        size=25,
        linewidth=0.85,
        zorder=6,
    )

    ax_patch.text(
        0.045,
        0.955,
        str(patch_index),
        transform=ax_patch.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="white",
        bbox={
            "facecolor": color,
            "edgecolor": "white",
            "boxstyle": "circle,pad=0.18",
            "linewidth": 0.65,
        },
        zorder=1,
    )

    ax_patch.set_xticks([])
    ax_patch.set_yticks([])

    style_spines(ax_patch, color=color, linewidth=1.5, visible=True)

    ax_spectrum = fig.add_axes(
        [
            spectrum_left,
            spectrum_bottom,
            spectrum_width,
            spectrum_height,
        ],
        zorder=3,
    )

    spectrum, spectrum_title = get_patch_spectrum(patch, spectrum_mode)
    finite_mask = np.isfinite(spectral_wavenumbers) & np.isfinite(spectrum)

    if not np.any(finite_mask):
        raise ValueError(f"Patch {patch_index} has no finite spectral values.")

    plot_x = spectral_wavenumbers[finite_mask]
    plot_y = spectrum[finite_mask]

    ax_spectrum.plot(
        plot_x,
        plot_y,
        color=color,
        linewidth=1.8,
        zorder=3,
    )

    ax_spectrum.set_xlim(*common_x_limits)
    ax_spectrum.set_ylim(*common_y_limits)
    style_spectrum_axis(ax_spectrum)

    ax_spectrum.set_title(
        spectrum_title,
        fontsize=16,
        pad=2.0,
        color="#303030",
    )

    ax_spectrum.set_xlabel(
        "Wavenumber (cm$^{-1}$)",
        fontsize=16,
        labelpad=5.0,
    )

    ax_spectrum.set_ylabel(
        "Intensity",
        fontsize=16,
        labelpad=4.0,
    )
    ax_spectrum.yaxis.set_label_position("left")
    ax_spectrum.yaxis.set_label_coords(-0.25, 0.50)


# ============================================================
# OUTPUT FIGURES
# ============================================================
def save_exact_patch_overview(
    image_rgb: np.ndarray,
    selected_patches: Sequence[PatchRecord],
    colors: Sequence[str],
    out_base_no_ext: str,
) -> Tuple[str, str]:
    image_height, image_width = image_rgb.shape[:2]
    image_aspect = image_width / image_height

    overview_height = 8.0
    overview_width = overview_height * image_aspect

    fig, ax = plt.subplots(
        figsize=(overview_width, overview_height),
        facecolor="white",
    )
    ax.imshow(image_rgb)
    ax.axis("off")
    ax.set_title(
        "H&E overview and selected patch locations",
        fontsize=16,
        pad=7,
    )

    for patch_index, (patch, color) in enumerate(
        zip(selected_patches, colors),
        start=1,
    ):
        bbox = get_patch_bbox(patch)
        seed_x, seed_y = get_patch_seed(patch)

        add_patch_bbox(
            ax,
            bbox,
            color,
            linewidth=1.1,
            zorder=4,
        )
        add_seed_marker(
            ax,
            seed_x,
            seed_y,
            color,
            size=110,
            linewidth=1.2,
            zorder=5,
        )

        ax.text(
            seed_x,
            seed_y,
            str(patch_index),
            ha="center",
            va="center",
            color="white",
            fontsize=16,
            fontweight="bold",
            zorder=6,
        )

    png_path, pdf_path = save_figure_both(fig, out_base_no_ext, dpi=600)
    plt.close(fig)
    return png_path, pdf_path


def save_compact_coregistration(
    image_rgb: np.ndarray,
    selected_patches: Sequence[PatchRecord],
    spectral_wavenumbers: np.ndarray,
    colors: Sequence[str],
    out_base_no_ext: str,
    spectrum_mode: str = "mean_valid_pixels",
    patch_display_mode: str = "masked",
) -> Tuple[str, str]:
    he_box, card_boxes = build_four_patch_layout_boxes()

    figure_width, figure_height = calculate_figure_size(image_rgb, he_box)

    fig = plt.figure(figsize=(figure_width, figure_height), facecolor="white")

    common_x_limits, common_y_limits = compute_common_spectrum_limits(
        selected_patches, spectral_wavenumbers, spectrum_mode
    )
    shared_geometry = compute_common_card_content_geometry(fig, card_boxes)

    image_left, image_bottom, image_width_frac, image_height_frac = he_box
    ax_image = fig.add_axes(
        [image_left, image_bottom, image_width_frac, image_height_frac],
        zorder=2,
    )

    ax_image.imshow(
        image_rgb,
        interpolation="nearest",
        aspect="equal",
        resample=False,
    )
    ax_image.set_xticks([])
    ax_image.set_yticks([])
    ax_image.set_anchor("C")

    style_spines(ax_image, color="#A5A5A5", linewidth=0.8, visible=True)

    ax_image.text(
        0.012,
        0.985,
        "H&E",
        transform=ax_image.transAxes,
        ha="left",
        va="top",
        fontsize=21,
        color="#202020",
        bbox={
            "facecolor": "white",
            "alpha": 0.86,
            "boxstyle": "round,pad=0.2",
        },
    )

    for patch_index, (patch, color) in enumerate(
        zip(selected_patches, colors),
        start=1,
    ):
        bbox = get_patch_bbox(patch)
        seed_x, seed_y = get_patch_seed(patch)

        add_patch_bbox(
            ax_image,
            bbox,
            color,
            linewidth=1.8,
            zorder=4,
        )
        add_seed_marker(
            ax_image,
            seed_x,
            seed_y,
            color,
            size=95,
            linewidth=1.1,
            zorder=5,
        )

        text_x, text_y = bbox_center(bbox)

        ax_image.text(
            text_x,
            text_y,
            str(patch_index),
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color="white",
            bbox={
                "facecolor": color,
                "edgecolor": "white",
                "boxstyle": "circle,pad=0.2",
                "linewidth": 1.2,
                "alpha": 0.95,
            },
            zorder=7,
        )

    for patch_index, (patch, color, card_box) in enumerate(
        zip(selected_patches, colors, card_boxes),
        start=1,
    ):
        add_patch_spectrum_card(
            fig,
            patch,
            spectral_wavenumbers,
            color,
            card_box,
            patch_index,
            spectrum_mode,
            patch_display_mode,
            common_x_limits,
            common_y_limits,
            shared_geometry,
        )

    png_path, pdf_path = save_figure_both(fig, out_base_no_ext, dpi=600)
    plt.close(fig)
    return png_path, pdf_path


# ============================================================
# MAIN
# ============================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    input_image = resolve_input_path(
        explicit_path=args.image,
        candidates=DEFAULT_IMAGE_CANDIDATES,
        description="H&E image",
    )
    input_csv = resolve_input_path(
        explicit_path=args.csv,
        candidates=DEFAULT_CSV_CANDIDATES,
        description="spectral CSV",
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Image path: {input_image}")
    print(f"[INFO] CSV path: {input_csv}")
    print(f"[INFO] Output directory: {output_dir}")

    print("[INFO] Loading image...")
    image_rgb = load_image(str(input_image))
    image_height, image_width = image_rgb.shape[:2]
    print(f"[INFO] Image shape: H={image_height}, W={image_width}")

    print("[INFO] Loading CSV and extracting spectra...")
    (
        df,
        pixel_x,
        pixel_y,
        pixel_labels,
        spectral_wavenumbers,
        spectral_matrix,
    ) = load_csv_and_prepare_features(
        csv_path=str(input_csv),
        image_shape=image_rgb.shape,
        label_map=LABEL_MAP,
        final_ranges=FINAL_RANGES,
    )

    print(f"[INFO] CSV valid rows used: {len(df)}")
    print(f"[INFO] Spectral feature count: {len(spectral_wavenumbers)}")

    print("[INFO] Building dense grids...")
    spectra_grid, label_grid, present_grid = build_dense_grids(
        image_shape=image_rgb.shape,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        pixel_labels=pixel_labels,
        spectral_matrix=spectral_matrix,
    )

    print("[INFO] Extracting exact patches...")
    patches = extract_exact_patches(
        image_rgb=image_rgb,
        spectra_grid=spectra_grid,
        label_grid=label_grid,
        present_grid=present_grid,
        reverse_label_map=REVERSE_LABEL_MAP,
        patch_size=args.patch_size,
        min_pixels=args.min_pixels,
    )

    print(f"[INFO] Total extracted patches: {len(patches)}")
    if not patches:
        raise RuntimeError(
            "No patches were extracted. Check coordinates, labels, and patch size."
        )

    print("[INFO] Selecting spatially spread patches...")
    selected_patches = select_evenly_spread_patches(patches, args.n_show)
    colors = PALETTE[: len(selected_patches)]

    print("[INFO] Selected patches:")
    for patch_index, patch in enumerate(selected_patches, start=1):
        x1, y1, x2, y2 = patch["bbox"]
        print(
            f"  Patch {patch_index}: "
            f"({patch['seed_x']}, {patch['seed_y']}), "
            f"bbox=({x1}, {y1})-({x2}, {y2}), "
            f"pixels={patch['valid_pixel_count']}, "
            f"label={patch['label_name']}"
        )

    overview_base = str(output_dir / "exact_patch_overview")
    compact_base = str(
        output_dir / "exact_patch_compact_coregistration"
    )

    print("[INFO] Saving patch-location overview...")
    overview_png, overview_pdf = save_exact_patch_overview(
        image_rgb=image_rgb,
        selected_patches=selected_patches,
        colors=colors,
        out_base_no_ext=overview_base,
    )

    print("[INFO] Saving tightly packed co-registration figure...")
    compact_png, compact_pdf = save_compact_coregistration(
        image_rgb=image_rgb,
        selected_patches=selected_patches,
        spectral_wavenumbers=spectral_wavenumbers,
        colors=colors,
        out_base_no_ext=compact_base,
        spectrum_mode=args.spectrum_mode,
        patch_display_mode=args.patch_display_mode,
    )

    print(f"[INFO] Saved: {overview_png}")
    print(f"[INFO] Saved: {overview_pdf}")
    print(f"[INFO] Saved: {compact_png}")
    print(f"[INFO] Saved: {compact_pdf}")
    print("[INFO] Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
