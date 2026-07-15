from __future__ import annotations

from pathlib import Path
import hashlib
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from spectral_profiles.config import DEFAULT_CLASS_COLORS, DEFAULT_LABEL_ORDER, FALLBACK_COLORS


def plot_mean_profiles(mean_profiles: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if mean_profiles.empty:
        return written

    for source_file, group in mean_profiles.groupby("source_file", sort=False):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        for label in _ordered_labels(group["label"].unique()):
            label_group = group[group["label"] == label].sort_values("wavenumber")
            ax.plot(
                label_group["wavenumber"],
                label_group["mean_intensity"],
                color=_label_color(label),
                linewidth=1.8,
                label=_format_label(label),
            )

        ax.set_title(f"Mean Spectral Profiles - {source_file}")
        ax.set_xlabel("Wavenumber (cm^-1)")
        ax.set_ylabel("Mean intensity")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()

        output_path = output_dir / f"{_safe_stem(source_file)}_mean_profiles.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(output_path)

    return written


def plot_sampled_profiles(sampled_profiles: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if sampled_profiles.empty:
        return written

    for source_file, group in sampled_profiles.groupby("source_file", sort=False):
        fig, ax = plt.subplots(figsize=(8.5, 5.0))

        for sample_id, sample_group in group.groupby("sample_id", sort=False):
            sample_group = sample_group.sort_values("wavenumber")
            label = str(sample_group["label"].iloc[0])
            ax.plot(
                sample_group["wavenumber"],
                sample_group["intensity"],
                color=_label_color(label),
                alpha=0.65,
                linewidth=0.9,
            )

        labels = _ordered_labels(group["label"].unique())
        for label in labels:
            ax.plot([], [], color=_label_color(label), label=_format_label(label))

        ax.set_title(f"Sampled Pixel Profiles - {source_file}")
        ax.set_xlabel("Wavenumber (cm^-1)")
        ax.set_ylabel("Intensity")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()

        output_path = output_dir / f"{_safe_stem(source_file)}_sampled_pixel_profiles.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        written.append(output_path)

    return written


def _ordered_labels(labels: list[str] | pd.Series) -> list[str]:
    label_set = {str(label) for label in labels}
    ordered = [label for label in DEFAULT_LABEL_ORDER if label in label_set]
    ordered.extend(sorted(label_set.difference(DEFAULT_LABEL_ORDER)))
    return ordered


def _label_color(label: str) -> str:
    if label in DEFAULT_CLASS_COLORS:
        return DEFAULT_CLASS_COLORS[label]

    label_digest = hashlib.sha1(label.encode("utf-8")).hexdigest()
    fallback_index = int(label_digest[:8], 16) % len(FALLBACK_COLORS)
    return FALLBACK_COLORS[fallback_index]


def _format_label(label: str) -> str:
    return label.replace("_", " ").title()


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_") or "spectra"
