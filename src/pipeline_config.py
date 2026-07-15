
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


VALID_MODES = [
    'mal_ep_vs_mal_st',
    'nonmal_ep_vs_nonmal_st',
    'mal_ep_vs_nonmal_ep',
    'mal_st_vs_nonmal_st',
    'mal_ep_vs_mal_st_vs_nonmal_ep_vs_nonmal_st',
    'all',
]
VALID_TRAINING_MODALITIES = ['image', 'spectra', 'both']
VALID_BALANCING_STRATEGIES = ['downsample', 'upsample', 'augment']
VALID_FUSION_STRATEGIES = ['early', 'image_only', 'spectra_only']
VALID_TRAINING_STRATEGIES = ['scratch', 'transfer_learning']

DEFAULT_INITIAL_WAVENUMBER_RANGES = [(1000, 1319), (1481, 1769), (2986, 3569)]
DEFAULT_FINAL_WAVENUMBER_RANGES = [(1019, 1300), (1500, 1750), (3005, 3550)]
DEFAULT_SEED_LIST = [44, 47, 51, 60, 67, 72, 80, 85, 91, 99]

LABEL_MAP = {
    'cancerous_epithelium': 0,
    'cancer_associated_stroma': 1,
    'normal_epithelium': 2,
    'normal_stroma': 3,
}
DISPLAY_NAMES = {
    0: 'cancerous epithelium',
    1: 'cancerous stroma',
    2: 'non-cancerous epithelium',
    3: 'non-cancerous stroma',
}


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {'true', '1', 'yes', 'y'}:
        return True
    if normalized in {'false', '0', 'no', 'n'}:
        return False
    raise argparse.ArgumentTypeError(f'Cannot interpret boolean value: {value}')


def model_output_units_for_mode(mode: str) -> int:
    if mode == 'all':
        return 4
    return 1


@dataclass(frozen=True)
class PipelineConfig:
    mode: str = 'all'
    training_modality: str = 'both'
    fusion_strategy: str = 'early'
    training_strategy: str = 'scratch'

    data_dir: str = os.path.join('..', 'spectra_csv_files')
    image_dir: str = os.path.join('..', 'aligned_he')
    annotated_image_dir: str = r'C:\Users\U0037942\Desktop\IP\Dataset\annotated pink cores'
    reserved_files: Optional[List[str]] = None
    output_root_dir: str = 'runs'
    patch_save_dir: Optional[str] = 'extracted_patches'

    patch_size: int = 5
    n_components: int = 32
    use_umap: bool = True
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = 'euclidean'

    balance_classes: bool = True
    balancing_strategy: str = 'downsample'
    samples_per_class: int = 20500

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4

    feature_extraction_epochs: int = 15
    fine_tune_epochs: int = 10
    transfer_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    freeze_backbone: bool = True
    staged_fine_tune: bool = True
    unfreeze_top_layers: int = 12
    transfer_head_units: int = 128
    source_weights_path: Optional[str] = None
    source_mode: Optional[str] = None
    source_num_classes_override: Optional[int] = None

    num_runs: int = 10
    seed_list: List[int] = field(default_factory=lambda: list(DEFAULT_SEED_LIST))

    initial_wavenumber_ranges: List[Tuple[int, int]] = field(
        default_factory=lambda: list(DEFAULT_INITIAL_WAVENUMBER_RANGES)
    )
    final_wavenumber_ranges: List[Tuple[int, int]] = field(
        default_factory=lambda: list(DEFAULT_FINAL_WAVENUMBER_RANGES)
    )
    savgol_window: int = 19
    savgol_polyorder: int = 4
    savgol_deriv: int = 1

    num_image_channels: int = 3
    label_map: Dict[str, int] = field(default_factory=lambda: dict(LABEL_MAP))
    display_names: Dict[int, str] = field(default_factory=lambda: dict(DISPLAY_NAMES))

    run_explainability: bool = True
    explain_num_samples: int = 24
    gradcam_enabled: bool = True
    shap_enabled: bool = True
    multimodal_quantification_enabled: bool = True

    save_full_model: bool = True
    save_preprocessing_bundle: bool = False

    @property
    def reverse_label_map(self) -> Dict[int, str]:
        return {value: key for key, value in self.label_map.items()}

    @property
    def display_reverse_map(self) -> Dict[int, str]:
        return {value: self.display_names[value] for value in self.display_names}

    @property
    def half_patch(self) -> int:
        return (self.patch_size - 1) // 2

    @property
    def source_mode_resolved(self) -> str:
        return self.source_mode if self.source_mode is not None else self.mode

    @property
    def source_output_units(self) -> int:
        if self.source_num_classes_override is not None:
            return self.source_num_classes_override
        return model_output_units_for_mode(self.source_mode_resolved)

    @property
    def target_output_units(self) -> int:
        return model_output_units_for_mode(self.mode)

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f'Invalid mode: {self.mode}')
        if self.training_modality not in VALID_TRAINING_MODALITIES:
            raise ValueError(f'Invalid training_modality: {self.training_modality}')
        if self.fusion_strategy not in VALID_FUSION_STRATEGIES:
            raise ValueError(f'Invalid fusion_strategy: {self.fusion_strategy}')
        if self.training_strategy not in VALID_TRAINING_STRATEGIES:
            raise ValueError(f'Invalid training_strategy: {self.training_strategy}')
        if self.balancing_strategy not in VALID_BALANCING_STRATEGIES:
            raise ValueError(f'Invalid balancing_strategy: {self.balancing_strategy}')
        if self.patch_size <= 0 or self.patch_size % 2 == 0:
            raise ValueError('patch_size must be a positive odd integer.')
        if self.n_components < 1:
            raise ValueError('n_components must be at least 1.')
        if self.batch_size < 1:
            raise ValueError('batch_size must be at least 1.')
        if self.num_runs < 1:
            raise ValueError('num_runs must be at least 1.')
        if len(self.seed_list) < self.num_runs:
            raise ValueError('seed_list must contain at least num_runs values.')
        if self.fusion_strategy in {'image_only', 'spectra_only'} and self.training_modality != 'both':
            raise ValueError(
                "fusion_strategy='image_only' and 'spectra_only' require training_modality='both'."
            )
        if self.training_strategy == 'transfer_learning' and not self.source_weights_path:
            raise ValueError('source_weights_path must be provided for transfer learning.')
        if self.training_strategy == 'transfer_learning' and not os.path.exists(str(self.source_weights_path)):
            raise FileNotFoundError(f'Source weights file not found: {self.source_weights_path}')
        if self.feature_extraction_epochs < 0 or self.fine_tune_epochs < 0:
            raise ValueError('feature_extraction_epochs and fine_tune_epochs must be non-negative.')
        if self.learning_rate <= 0 or self.transfer_learning_rate <= 0 or self.fine_tune_learning_rate <= 0:
            raise ValueError('All learning rates must be positive.')
        if self.unfreeze_top_layers < 0:
            raise ValueError('unfreeze_top_layers must be >= 0.')
        if self.use_umap and self.training_modality in {'spectra', 'both'} and self.n_components < len(self.final_wavenumber_ranges):
            raise ValueError(
                'For exact per-range UMAP allocation, n_components must be at least the number of non-empty ranges.'
            )

    def summary_lines(self) -> List[str]:
        return [
            '-' * 72,
            '[Config] Pipeline configuration loaded:',
            f'  Mode: {self.mode}',
            f'  Training Modality: {self.training_modality}',
            f'  Fusion Strategy: {self.fusion_strategy}',
            f'  Training Strategy: {self.training_strategy}',
            f'  Use UMAP: {self.use_umap}',
            f'  Requested UMAP Components: {self.n_components}',
            f'  Patch Size: {self.patch_size}x{self.patch_size}',
            f'  Balance Classes: {self.balance_classes}',
            f'  Balancing Strategy: {self.balancing_strategy}',
            f'  Data Directory: {self.data_dir}',
            f'  Image Directory: {self.image_dir}',
            f'  Output Root Directory: {self.output_root_dir}',
            f'  Patch Save Directory: {self.patch_save_dir}',
            f'  Source Weights Path: {self.source_weights_path}',
            f'  Source Output Units: {self.source_output_units}',
            f'  Target Output Units: {self.target_output_units}',
            f'  Feature Extraction Epochs: {self.feature_extraction_epochs}',
            f'  Fine-tune Epochs: {self.fine_tune_epochs}',
            f'  Freeze Backbone: {self.freeze_backbone}',
            f'  Staged Fine-tune: {self.staged_fine_tune}',
            f'  Unfreeze Top Layers: {self.unfreeze_top_layers}',
            f'  Batch Size: {self.batch_size}',
            f'  Main Learning Rate: {self.learning_rate}',
            f'  Transfer Learning Rate: {self.transfer_learning_rate}',
            f'  Fine-tune Learning Rate: {self.fine_tune_learning_rate}',
            f'  Number of Runs: {self.num_runs}',
            f'  Seeds: {self.seed_list[:self.num_runs]}',
            '-' * 72,
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Training pipeline for Spatially Adaptive Multimodal Learning from Co-Registered '
            'Histopathology and Infrared Spectroscopy for Breast Cancer Subtyping.'
        )
    )
    parser.add_argument('--mode', type=str, default='all', choices=VALID_MODES)
    parser.add_argument('--training_modality', type=str, default='both', choices=VALID_TRAINING_MODALITIES)
    parser.add_argument('--fusion_strategy', type=str, default='early', choices=VALID_FUSION_STRATEGIES)
    parser.add_argument('--training_strategy', type=str, default='scratch', choices=VALID_TRAINING_STRATEGIES)

    parser.add_argument('--data_dir', type=str, default=os.path.join('..', 'data')) # spectra files dir
    parser.add_argument('--image_dir', type=str, default=os.path.join('..', 'aligned_he')) # H&E images dir
    parser.add_argument('--annotated_image_dir', type=str, default=r'C:\Users\U0037942\Desktop\IP\Dataset\annotated pink cores')
    parser.add_argument('--reserved_files', nargs='*', default=None)
    parser.add_argument('--output_root_dir', type=str, default='runs')
    parser.add_argument('--patch_save_dir', type=str, default='extracted_patches')

    parser.add_argument('--patch_size', type=int, default=19)
    parser.add_argument('--n_components', type=int, default=64)
    parser.add_argument('--use_umap', type=str2bool, default=True)
    parser.add_argument('--umap_n_neighbors', type=int, default=15)
    parser.add_argument('--umap_min_dist', type=float, default=0.1)
    parser.add_argument('--umap_metric', type=str, default='euclidean')

    parser.add_argument('--balance_classes', type=str2bool, default=True)
    parser.add_argument('--balancing_strategy', type=str, default='downsample', choices=VALID_BALANCING_STRATEGIES)
    parser.add_argument('--samples_per_class', type=int, default=20500)

    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)

    parser.add_argument('--feature_extraction_epochs', type=int, default=15)
    parser.add_argument('--fine_tune_epochs', type=int, default=10)
    parser.add_argument('--transfer_learning_rate', type=float, default=1e-3)
    parser.add_argument('--fine_tune_learning_rate', type=float, default=1e-5)
    parser.add_argument('--freeze_backbone', type=str2bool, default=True)
    parser.add_argument('--staged_fine_tune', type=str2bool, default=True)
    parser.add_argument('--unfreeze_top_layers', type=int, default=12)
    parser.add_argument('--transfer_head_units', type=int, default=128)
    parser.add_argument('--source_weights_path', type=str, default='runs/run_seed_44/model_outputs/run_mode_all_ncomp32_patch5_modality_both_fusion_early_strategy_scratch_final_weights.weights.h5')
    parser.add_argument('--source_mode', type=str, default=None, choices=VALID_MODES)
    parser.add_argument('--source_num_classes_override', type=int, default=None)

    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--seed_list', nargs='*', type=int, default=list(DEFAULT_SEED_LIST))

    parser.add_argument('--run_explainability', type=str2bool, default=True)
    parser.add_argument('--explain_num_samples', type=int, default=24)
    parser.add_argument('--gradcam_enabled', type=str2bool, default=True)
    parser.add_argument('--shap_enabled', type=str2bool, default=True)
    parser.add_argument('--multimodal_quantification_enabled', type=str2bool, default=True)
    parser.add_argument('--save_full_model', type=str2bool, default=True)
    parser.add_argument('--save_preprocessing_bundle', type=str2bool, default=False)
    return parser


def config_from_namespace(namespace: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig(
        mode=namespace.mode,
        training_modality=namespace.training_modality,
        fusion_strategy=namespace.fusion_strategy,
        training_strategy=namespace.training_strategy,
        data_dir=namespace.data_dir,
        image_dir=namespace.image_dir,
        annotated_image_dir=namespace.annotated_image_dir,
        reserved_files=namespace.reserved_files,
        output_root_dir=namespace.output_root_dir,
        patch_save_dir=namespace.patch_save_dir,
        patch_size=namespace.patch_size,
        n_components=namespace.n_components,
        use_umap=namespace.use_umap,
        umap_n_neighbors=namespace.umap_n_neighbors,
        umap_min_dist=namespace.umap_min_dist,
        umap_metric=namespace.umap_metric,
        balance_classes=namespace.balance_classes,
        balancing_strategy=namespace.balancing_strategy,
        samples_per_class=namespace.samples_per_class,
        epochs=namespace.epochs,
        batch_size=namespace.batch_size,
        learning_rate=namespace.learning_rate,
        feature_extraction_epochs=namespace.feature_extraction_epochs,
        fine_tune_epochs=namespace.fine_tune_epochs,
        transfer_learning_rate=namespace.transfer_learning_rate,
        fine_tune_learning_rate=namespace.fine_tune_learning_rate,
        freeze_backbone=namespace.freeze_backbone,
        staged_fine_tune=namespace.staged_fine_tune,
        unfreeze_top_layers=namespace.unfreeze_top_layers,
        transfer_head_units=namespace.transfer_head_units,
        source_weights_path=namespace.source_weights_path,
        source_mode=namespace.source_mode,
        source_num_classes_override=namespace.source_num_classes_override,
        num_runs=namespace.num_runs,
        seed_list=list(namespace.seed_list),
        run_explainability=namespace.run_explainability,
        explain_num_samples=namespace.explain_num_samples,
        gradcam_enabled=namespace.gradcam_enabled,
        shap_enabled=namespace.shap_enabled,
        multimodal_quantification_enabled=namespace.multimodal_quantification_enabled,
        save_full_model=namespace.save_full_model,
        save_preprocessing_bundle=namespace.save_preprocessing_bundle,
    )
    config.validate()
    return config


def get_config(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    return config_from_namespace(namespace)


if __name__ == '__main__':
    runtime_config = get_config()
    for line in runtime_config.summary_lines():
        print(line)
