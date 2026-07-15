from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from matplotlib import rcParams
import matplotlib.pyplot as plt

DEFAULT_MODALITIES = {
    "Image-only": "img_reports",
    "Spectra-only": "spectra_reports",
    "Multimodal": "multimodal_reports",
}

DEFAULT_PALETTE = {
    "Image-only": "#2E86AB",
    "Spectra-only": "#A23B72",
    "Multimodal": "#F18F01",
}


@dataclass(frozen=True)
class AnalysisConfig:
    input_root: Path = Path("raw_reports")
    output_dir: Path = Path("outputs")
    summary_csv: str = "classification_metrics_summary.csv"
    t_test_csv: str = "t_test_results.csv"
    descriptive_csv: str = "descriptive_statistics.csv"
    report_name: str = "summary_report.txt"
    modalities: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_MODALITIES))
    palette: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PALETTE))


def configure_matplotlib() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.titleweight": "normal",
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "figure.titleweight": "normal",
            "figure.figsize": (6, 4),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.axisbelow": True,
        }
    )
