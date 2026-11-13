

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


LABEL_MAP = {
    "malignant_epithelium": 0,
    "malignant_stroma": 1,
    "non_malignant_epithelium": 2,
    "non_malignant_stroma": 3,
}
REVERSE_LABEL_MAP = {value: key for key, value in LABEL_MAP.items()}
MODE_MAPPINGS = {
    "mal_ep_vs_mal_st": ((0, 1), {0: 0, 1: 1}),
    "nonmal_ep_vs_nonmal_st": ((2, 3), {2: 0, 3: 1}),
    "mal_ep_vs_nonmal_ep": ((0, 2), {0: 0, 2: 1}),
    "mal_st_vs_nonmal_st": ((1, 3), {1: 0, 3: 1}),
    "mal_ep_vs_mal_st_vs_nonmal_ep_vs_nonmal_st": (
        (0, 1, 2, 3),
        {0: 0, 1: 0, 2: 1, 3: 1},
    ),
    "all": ((0, 1, 2, 3), {0: 0, 1: 1, 2: 2, 3: 3}),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ============================================================================
# USER DATA PATHS — EDIT ONLY THESE FOUR LINES
# ============================================================================
# Local Windows paths:
WINDOWS_SPECTRA_DIR = Path(r"breast_spectral_data")
WINDOWS_IMAGE_DIR = Path(r"breast_images")

# Linux/cloud paths:
CLOUD_SPECTRA_DIR = Path("/workspace/data/spectral")
CLOUD_IMAGE_DIR = Path("/workspace/data/images")
# ============================================================================


def platform_input_dirs() -> Tuple[Path, Path]:
    """Return the adjacent hard-coded spectra/image pair for this platform."""
    if os.name == "nt":
        return WINDOWS_SPECTRA_DIR, WINDOWS_IMAGE_DIR
    return CLOUD_SPECTRA_DIR, CLOUD_IMAGE_DIR


def application_root() -> Path:
    """Directory that owns outputs in both package and flat deployments."""
    module_root = Path(__file__).resolve().parent
    if module_root.name == "consolidated" and (module_root / "__init__.py").is_file():
        return module_root.parent
    return module_root


def image_directory_candidates(data_dir: Path) -> List[Path]:
    """Portable image locations inferred from the one required data directory."""
    data_dir = Path(data_dir).expanduser().resolve()
    parent = data_dir.parent
    name = data_dir.name
    derived_names = []
    for old, new in (
        ("_spectral_data", "_images"),
        ("spectral_data", "images"),
        ("_spectral", "_images"),
        ("spectral", "images"),
    ):
        if old in name.lower():
            start = name.lower().index(old)
            derived_names.append(name[:start] + new + name[start + len(old) :])
    candidates = [
        data_dir / "images",
        parent / "images",
        *(parent / value for value in derived_names),
        data_dir,
    ]
    unique = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def infer_image_dir(data_dir: Path) -> Path:
    """Find co-registered images """
    candidates = image_directory_candidates(data_dir)
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        try:
            if any(
                item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
                for item in candidate.iterdir()
            ):
                return candidate
        except OSError:
            continue
    return candidates[0]


@dataclass
class ExperimentConfig:
    # EDITABLE DEFAULT RUN SETTINGS.
    data_dir: Path
    image_dir: Optional[Path] = None
    output_dir: Path = Path("runs_consolidated")
    experiment_role: str = "single"
    parent_requested_modality: Optional[str] = None
    ablation_variant: Optional[str] = None
    split_strategy: str = "patient"
    patient_id_regex: Optional[str] = None
    reserved_test_files: List[str] = field(default_factory=list)
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    test_fraction: float = 0.10
    split_seeds: List[int] = field(default_factory=lambda: [128, 512, 924, 1002, 1016, 1024, 1523, 1666, 3303, 4025])

    mode: str = "all"
    modality: str = "spectra"
    patch_size: int = 19
    min_patches_per_class: int = 1
    min_patients_per_class: int = 1

    spectral_fit_max_samples: int = 50000

    latent_width: int = 48
    encoder_widths: Tuple[int, ...] = (64, 96, 128)
    attention_heads: int = 4
    dropout: float = 0.20
    head_dropout: float = 0.35
    modality_dropout: float = 0.20
    optical_density_jitter: float = 0.08
    rgb_aux_weight: float = 0.30
    spectra_aux_weight: float = 0.15
    label_smoothing: float = 0.05
    l2: float = 1e-4

    epochs: int = 35
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 12
    log_every_batches: int = 10
    balance_classes: bool = True
    strict_duplicate_audit: bool = True
    final_test: bool = True
    bootstrap_iterations: int = 2000
    run_explainability: bool = True
    run_ablation_suite: bool = False
    explain_samples: int = 40
    integrated_gradient_steps: int = 32
    occlusion_window: int = 5
    occlusion_stride: int = 3
    perturbation_repeats: int = 16
    perturbation_scale: float = 0.10
    run_shap: bool = True
    shap_top_features: int = 10
    shap_background_size: int = 32
    shap_nsamples: int = 100
    deterministic_ops: bool = True

    initial_wavenumber_ranges: Tuple[Tuple[float, float], ...] = (
        (1000, 1319),
        (1481, 1769),
        (2986, 3569),
    )
    final_wavenumber_ranges: Tuple[Tuple[float, float], ...] = (
        (1019, 1300),
        (1500, 1750),
        (3005, 3550),
    )
    savgol_window: int = 19
    savgol_polyorder: int = 4
    savgol_derivative: int = 1

    def validate(self) -> "ExperimentConfig":
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        self.image_dir = (
            infer_image_dir(self.data_dir)
            if self.image_dir is None or not str(self.image_dir).strip()
            else Path(self.image_dir).expanduser().resolve()
        )
        self.output_dir = Path(self.output_dir)
        if self.split_strategy not in {"patient", "patch"}:
            raise ValueError("split_strategy must be 'patient' or 'patch'")
        if self.experiment_role not in {"single", "ablation_variant"}:
            raise ValueError("experiment_role must be 'single' or 'ablation_variant'")
        if self.experiment_role == "ablation_variant" and not self.ablation_variant:
            raise ValueError("ablation_variant is required for an ablation-suite variant")
        if self.run_ablation_suite and self.final_test:
            raise ValueError(
                "run_ablation_suite=true cannot be combined with final_test=true: "
                "perform ablation/model selection with the test sealed, freeze the design, "
                "then run the selected configuration once with --run-ablation-suite false "
                "--final-test true"
            )
        if self.modality not in {"image", "spectra", "both"}:
            raise ValueError("modality must be image, spectra, or both")
        if self.mode not in MODE_MAPPINGS:
            raise ValueError(f"unknown classification mode: {self.mode}")
        fractions = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(fractions - 1.0) > 1e-8 or min(
            self.train_fraction, self.validation_fraction, self.test_fraction
        ) <= 0:
            raise ValueError("split fractions must be positive and sum to one")
        positive = {
            "patch_size": self.patch_size,
            "latent_width": self.latent_width,
            "attention_heads": self.attention_heads,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "bootstrap_iterations": self.bootstrap_iterations,
            "log_every_batches": self.log_every_batches,
            "explain_samples": self.explain_samples,
            "integrated_gradient_steps": self.integrated_gradient_steps,
            "occlusion_window": self.occlusion_window,
            "occlusion_stride": self.occlusion_stride,
            "perturbation_repeats": self.perturbation_repeats,
            "shap_top_features": self.shap_top_features,
            "shap_background_size": self.shap_background_size,
            "shap_nsamples": self.shap_nsamples,
        }
        for name, value in positive.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.explain_samples > 40:
            raise ValueError("explain_samples must be <= 40 for the predeclared explanation protocol")
        if self.shap_top_features > 10:
            raise ValueError(
                "shap_top_features must be <= 10 for the predeclared top-10 SHAP protocol"
            )
        if self.spectral_fit_max_samples < 0:
            raise ValueError("spectral_fit_max_samples must be >= 0")
        for name in ("dropout", "head_dropout", "modality_dropout", "label_smoothing"):
            value = float(getattr(self, name))
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not 0 <= self.optical_density_jitter < 1:
            raise ValueError("optical_density_jitter must be in [0, 1)")
        if not 0 < self.perturbation_scale <= 1:
            raise ValueError("perturbation_scale must be in (0, 1]")
        if self.latent_width % self.attention_heads:
            raise ValueError("latent_width must be divisible by attention_heads")
        if not self.encoder_widths or any(int(width) <= 0 for width in self.encoder_widths):
            raise ValueError("encoder_widths must contain positive values")
        if any(int(width) % self.attention_heads for width in self.encoder_widths):
            raise ValueError("every encoder width must be divisible by attention_heads")
        for name in (
            "learning_rate",
            "weight_decay",
            "l2",
            "rgb_aux_weight",
            "spectra_aux_weight",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if not self.split_seeds or len(set(self.split_seeds)) != len(self.split_seeds):
            raise ValueError("split_seeds must be non-empty and unique")
        if any(seed < 0 for seed in self.split_seeds):
            raise ValueError("split seeds must be non-negative")
        if self.savgol_window <= self.savgol_polyorder or self.savgol_window % 2 == 0:
            raise ValueError("savgol_window must be odd and larger than savgol_polyorder")
        return self

    @property
    def relevant_labels(self) -> Tuple[int, ...]:
        return tuple(MODE_MAPPINGS[self.mode][0])

    @property
    def class_mapping(self):
        return dict(MODE_MAPPINGS[self.mode][1])

    @property
    def num_classes(self) -> int:
        return max(self.class_mapping.values()) + 1

    @property
    def target_names(self) -> List[str]:
        names = {
            "mal_ep_vs_mal_st": ["malignant_epithelium", "malignant_stroma"],
            "nonmal_ep_vs_nonmal_st": [
                "non_malignant_epithelium",
                "non_malignant_stroma",
            ],
            "mal_ep_vs_nonmal_ep": [
                "malignant_epithelium",
                "non_malignant_epithelium",
            ],
            "mal_st_vs_nonmal_st": ["malignant_stroma", "non_malignant_stroma"],
            "mal_ep_vs_mal_st_vs_nonmal_ep_vs_nonmal_st": [
                "malignant",
                "non_malignant",
            ],
            "all": [REVERSE_LABEL_MAP[index] for index in range(len(REVERSE_LABEL_MAP))],
        }
        return list(names[self.mode])

    def to_dict(self):
        values = asdict(self)
        for key in ("data_dir", "image_dir", "output_dir"):
            values[key] = None if values[key] is None else str(values[key])
        return values

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _bool(value: str) -> bool:
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"cannot interpret {value!r} as boolean")


def build_parser() -> argparse.ArgumentParser:
    # ExperimentConfig above is the only editable source of defaults. Argument
    # declarations reference it directly so displayed and effective values
    # cannot disagree.
    hardcoded_data_dir, hardcoded_image_dir = platform_input_dirs()
    canonical_defaults = ExperimentConfig(
        data_dir=Path(
            os.environ.get("CONSOLIDATED_DATA_DIR", hardcoded_data_dir)
        ),
        image_dir=Path(
            os.environ.get("CONSOLIDATED_IMAGE_DIR", hardcoded_image_dir)
        ),
        output_dir=Path(
            os.environ.get(
                "CONSOLIDATED_OUTPUT_DIR", application_root() / "runs_consolidated"
            )
        ),
    )
    parser = argparse.ArgumentParser(
        description="Leakage-aware modality-balanced early-fusion classifier",
        allow_abbrev=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        default=canonical_defaults.data_dir,
        type=Path,
        help="Spectral CSV directory (or set CONSOLIDATED_DATA_DIR).",
    )
    parser.add_argument(
        "--image-dir",
        default=canonical_defaults.image_dir,
        type=Path,
        help=(
            "Optional co-registered image directory. If omitted, discover images "
            "inside/beside --data-dir; CONSOLIDATED_IMAGE_DIR also overrides discovery."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=canonical_defaults.output_dir,
        help="Experiment output directory (or set CONSOLIDATED_OUTPUT_DIR).",
    )
    parser.add_argument(
        "--split-strategy",
        choices=["patient", "patch"],
        default=canonical_defaults.split_strategy,
        help="Partition by disjoint patients or by individual patches.",
    )
    parser.add_argument("--patient-id-regex", default=canonical_defaults.patient_id_regex)
    parser.add_argument(
        "--reserved-test-files", nargs="*", default=canonical_defaults.reserved_test_files
    )
    parser.add_argument("--train-fraction", type=float, default=canonical_defaults.train_fraction)
    parser.add_argument(
        "--validation-fraction", type=float, default=canonical_defaults.validation_fraction
    )
    parser.add_argument("--test-fraction", type=float, default=canonical_defaults.test_fraction)
    parser.add_argument(
        "--split-seeds", nargs="+", type=int, default=canonical_defaults.split_seeds
    )
    parser.add_argument("--mode", choices=sorted(MODE_MAPPINGS), default=canonical_defaults.mode)
    parser.add_argument(
        "--modality",
        choices=["image", "spectra", "both"],
        default=canonical_defaults.modality,
        help="Active input modality for a single run; an ablation suite also runs controls.",
    )
    parser.add_argument("--patch-size", type=int, default=canonical_defaults.patch_size)
    parser.add_argument(
        "--min-patches-per-class", type=int, default=canonical_defaults.min_patches_per_class
    )
    parser.add_argument(
        "--min-patients-per-class", type=int, default=canonical_defaults.min_patients_per_class
    )
    parser.add_argument("--spectral-fit-max-samples", type=int, default=canonical_defaults.spectral_fit_max_samples)
    parser.add_argument("--latent-width", type=int, default=canonical_defaults.latent_width)
    parser.add_argument(
        "--encoder-widths", nargs="+", type=int, default=canonical_defaults.encoder_widths
    )
    parser.add_argument("--attention-heads", type=int, default=canonical_defaults.attention_heads)
    parser.add_argument("--dropout", type=float, default=canonical_defaults.dropout)
    parser.add_argument("--head-dropout", type=float, default=canonical_defaults.head_dropout)
    parser.add_argument(
        "--modality-dropout", type=float, default=canonical_defaults.modality_dropout
    )
    parser.add_argument(
        "--optical-density-jitter", type=float, default=canonical_defaults.optical_density_jitter
    )
    parser.add_argument("--rgb-aux-weight", type=float, default=canonical_defaults.rgb_aux_weight)
    parser.add_argument(
        "--spectra-aux-weight", type=float, default=canonical_defaults.spectra_aux_weight
    )
    parser.add_argument("--label-smoothing", type=float, default=canonical_defaults.label_smoothing)
    parser.add_argument("--l2", type=float, default=canonical_defaults.l2)
    parser.add_argument("--epochs", type=int, default=canonical_defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=canonical_defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=canonical_defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=canonical_defaults.weight_decay)
    parser.add_argument(
        "--early-stopping-patience", type=int, default=canonical_defaults.early_stopping_patience
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=canonical_defaults.log_every_batches,
        help="Emit an explicit optimizer-progress log every N training batches.",
    )
    parser.add_argument("--balance-classes", type=_bool, default=canonical_defaults.balance_classes)
    parser.add_argument(
        "--strict-duplicate-audit", type=_bool, default=canonical_defaults.strict_duplicate_audit
    )
    parser.add_argument(
        "--final-test",
        type=_bool,
        default=canonical_defaults.final_test,
        help="Evaluate the sealed test partition after the design is frozen.",
    )
    parser.add_argument(
        "--bootstrap-iterations", type=int, default=canonical_defaults.bootstrap_iterations
    )
    parser.add_argument(
        "--run-explainability",
        type=_bool,
        default=canonical_defaults.run_explainability,
        help="Generate held-out Grad-CAM, occlusion, perturbation, and applicable SHAP artifacts.",
    )
    parser.add_argument(
        "--run-ablation-suite",
        type=_bool,
        default=canonical_defaults.run_ablation_suite,
        help="Run the requested primary model plus explicitly labelled modality controls.",
    )
    parser.add_argument(
        "--explain-samples",
        type=int,
        default=canonical_defaults.explain_samples,
        help="Number of held-out patches explained per seed (maximum 40).",
    )
    parser.add_argument(
        "--integrated-gradient-steps",
        type=int,
        default=canonical_defaults.integrated_gradient_steps,
    )
    parser.add_argument("--occlusion-window", type=int, default=canonical_defaults.occlusion_window)
    parser.add_argument("--occlusion-stride", type=int, default=canonical_defaults.occlusion_stride)
    parser.add_argument(
        "--perturbation-repeats", type=int, default=canonical_defaults.perturbation_repeats
    )
    parser.add_argument(
        "--perturbation-scale",
        type=float,
        default=canonical_defaults.perturbation_scale,
        help="Relative per-channel standard-deviation used for sensitivity perturbations.",
    )
    parser.add_argument("--run-shap", type=_bool, default=canonical_defaults.run_shap)
    parser.add_argument(
        "--shap-top-features", type=int, default=canonical_defaults.shap_top_features
    )
    parser.add_argument(
        "--shap-background-size", type=int, default=canonical_defaults.shap_background_size
    )
    parser.add_argument("--shap-nsamples", type=int, default=canonical_defaults.shap_nsamples)
    parser.add_argument(
        "--deterministic-ops", type=_bool, default=canonical_defaults.deterministic_ops
    )
    return parser


def parse_config(argv: Optional[Sequence[str]] = None) -> ExperimentConfig:
    return ExperimentConfig(**vars(build_parser().parse_args(argv))).validate()
