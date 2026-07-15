<h1 align="center">Spatially Adaptive Multimodal Learning from Co-Registered<br>Histopathology and Infrared Spectroscopy for Breast Cancer Subtyping</h1>

<p align="center"><strong>Training, preprocessing, explainability, and figure workflows for paired H&amp;E histopathology and infrared spectroscopy breast cancer subtyping.</strong></p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.4730312"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.4730312.svg" alt="Dataset DOI"></a>
</p>

## Overview

This repository contains research code for spatially adaptive multimodal learning from co-registered H&E histopathology images and infrared spectral data. The main pipeline trains image-only, spectra-only, and multimodal convolutional models; supporting folders reproduce the data-preparation utilities, preprocessing figures, patch/co-registration figures, infrared profile plots, and statistical performance figures.

The code is intended for research and reproducibility. Before using the results in a manuscript, report, or clinical-adjacent workflow, verify local data paths, preprocessing settings, random seeds, dependency versions, and generated artifacts.

## Data Source

The breast infrared pathology data used with this project should be obtained from Zenodo:

- Dataset:(https://zenodo.org/records/4730312)

The Zenodo record contains large archives and files including annotations, H&E images, chemical images, and data cubes. Download the required files from Zenodo.

## Repository Layout

```text
.
|-- README.md
|-- src/                              Main model training and explainability pipeline
|   |-- run_experiment.py             Command-line entry point
|   |-- experiment_pipeline.py         End-to-end experiment orchestration
|   |-- pipeline_config.py             CLI options, defaults, validation, labels
|   |-- spectral_file_loader.py        CSV/image loading and spectral preprocessing
|   |-- patch_extraction.py            Spatial patch extraction
|   |-- dataset_builder.py             Dataset assembly and train/validation/test split
|   |-- model_architecture.py          Spatially adaptive Keras model definitions
|   |-- model_training.py              Compile, fit, fine-tune, and save logic
|   |-- evaluation.py                  Test metrics, confusion matrix, ROC curves
|   |-- explainability_runner.py       Grad-CAM, SHAP, and modality quantification workflow
|   `-- training_plots.py             Training-history plots
|-- data preparation/                 Agilent FTIR extraction and annotation-labeling package
|-- preprocessing stages plot/        Spectral preprocessing figure generator
|-- patch plots/                      H&E patch and co-registration figure generator
|-- IR profile/                       Mean and sampled-pixel spectral profile tools
`-- stats plots/                      Classification-report statistics plots
```

The steps below are the recommended top-level path for a fresh reproduction.

## What The Pipeline Does

- Loads spectral data stored in CSV files and aligned H&E/RGB images with matching filename stems.
- Supports labels for `malignant_epithelium`, `malignant_stroma`, `non_malignant_epithelium`, and `non_malignant_stroma`.
- Filters infrared wavenumber ranges and applies Savitzky-Golay first-derivative preprocessing.
- Extracts coverage-preserving, non-overlapping spatial patches.
- Fits spectral reducers/scalers on training pixels only to reduce leakage risk.
- Trains image-only, spectra-only, and multimodal single-input models.
- Supports scratch training and transfer learning from compatible saved weights.
- Writes training histories, metrics, confusion matrices, ROC curves, optional model files, and explainability outputs.
- Generates standalone figures for preprocessing, patch co-registration, IR profiles, and model-performance statistics.

## Installation

Use Python 3.10 or newer. A GPU-enabled TensorFlow environment is recommended for full training, but the code can run on CPU for small smoke tests.

```powershell
git clone <repository-url>
Set-Location "Publication GitHub"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the main training dependencies:

```powershell
python -m pip install tensorflow numpy pandas matplotlib scikit-learn scipy pillow opencv-python umap-learn shap
```

`shap` is only required when spectral SHAP explainability is enabled. For GPU acceleration, install TensorFlow, CUDA, cuDNN, and drivers that match your hardware and TensorFlow version.

## Step 1: Download And Arrange Data

1. Download the needed files from [Zenodo record 4730312](https://zenodo.org/records/4730312).
2. Extract the archives outside the repository
3. Prepare two folders for model training:

```text
project-data/
|-- spectra_csv_files/
|   |-- sample_001.csv
|   `-- sample_002.csv
`-- aligned_he/
    |-- sample_001.png
    `-- sample_002.png
```

Each CSV file that contains spectral data must have a matching image with the same filename stem. Supported image extensions include `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, and `.bmp`.

Each training CSV should contain:

```text
Pixel_Y
Pixel_X
Label
Wavenumber_<value>
Wavenumber_<value>
...
```

Example spectral columns are `Wavenumber_1000`, `Wavenumber_1001.5`, and `Wavenumber_1750`.

## Step 2: Prepare Spectra And Labels

The `data preparation/` package extracts spectra from Agilent `.dmt` and `.seq` files and assigns pathology labels from annotated RGB images.


Extract one file or a directory of Agilent files:

```powershell
spectral-pathology extract "agilent_files" --output-dir data\spectra
```

Apply labels from matching annotation images:

```powershell
spectral-pathology label data\spectra --annotations-dir data\annotations --output-dir data\labeled --color-tolerance 3
```

Or run extraction and labeling together:

```powershell
spectral-pathology run "agilent_files" --annotations-dir data\annotations --spectra-dir data\spectra --output-dir data\labeled
```

Then copy or point the resulting labeled CSV files into the model-training `spectra_csv_files` folder, and place the corresponding aligned H&E images in `aligned_he`.

## Step 3: Run Model Experiments

Return to the repository root before running the main pipeline:

```powershell
Set-Location "src"
```

Run a short smoke test first:

```powershell
python .\src\run_experiment.py `
  --data_dir "spectra_csv_files" `
  --image_dir "aligned_he" `
  --num_runs 1 `
  --epochs 1 `
  --run_explainability false
```

Run the default multimodal experiment:

```powershell
python .\src\run_experiment.py `
  --data_dir "spectra_csv_files" `
  --image_dir "aligned_he" `
  --training_modality both `
  --fusion_strategy early `
  --mode all
```

Run image-only and spectra-only ablations:

```powershell
python .\src\run_experiment.py --data_dir "spectra_csv_files" --image_dir "aligned_he" --training_modality image --fusion_strategy early
python .\src\run_experiment.py --data_dir "spectra_csv_files" --image_dir "aligned_he" --training_modality spectra --fusion_strategy early
```

Valid classification modes are:

```text
all
mal_ep_vs_mal_st
nonmal_ep_vs_nonmal_st
mal_ep_vs_nonmal_ep
mal_st_vs_nonmal_st
mal_ep_vs_mal_st_vs_nonmal_ep_vs_nonmal_st
```

Important default preprocessing settings:

```text
Initial wavenumber ranges: 1000-1319, 1481-1769, 2986-3569
Final wavenumber ranges:   1019-1300, 1500-1750, 3005-3550
Savitzky-Golay window:     19
Savitzky-Golay polyorder:  4
Savitzky-Golay derivative: 1
Default seeds:             44, 47, 51, 60, 67, 72, 80, 85, 91, 99
```

Common options:

```text
--patch_size                   Patch side length; must be a positive odd integer
--n_components                 Requested spectral UMAP components
--use_umap                     true or false
--balance_classes              true or false
--balancing_strategy           downsample, upsample, or augment
--epochs                       Scratch-training epochs
--batch_size                   Batch size
--learning_rate                Scratch-training learning rate
--num_runs                     Number of repeated seeded runs
--seed_list                    Space-separated random seeds
--run_explainability           true or false
--save_full_model              Save full .keras model as well as weights
--save_preprocessing_bundle    Save fitted spectral transform bundle
```

## Step 4: Review Training Outputs

By default, model outputs are written under `runs/`:

```text
runs/
`-- run_seed_44/
    |-- model_outputs/
    |   |-- *_final_weights.weights.h5
    |   |-- *_final_model.keras
    |   |-- *_training_history.csv
    |   `-- *_training_plots.png
    |-- test_results/
    |   |-- classification_report.csv
    |   |-- confusion_matrix.csv
    |   |-- confusion_matrix.png
    |   `-- roc_curve.png
    |-- explainability/
    |   |-- gradcam/
    |   |-- spectra_shap/
    |   `-- multimodal_quantification/
    `-- logs/
```

Extracted patch CSVs are written to `extracted_patches/` unless `--patch_save_dir` is changed. If `--save_preprocessing_bundle true` is enabled, the fitted spectral transform bundle is saved under `runs/preprocessing/`.

## Step 5: Generate  Figures

### Spectral Preprocessing Figure

```powershell
Set-Location "preprocessing stages plot"
python -m pip install -r requirements.txt
python .\preprocessing_plots_generation.py --data-dir data --output-dir preprocessing_plot
Set-Location ..
```

Outputs:

```text
preprocessing stages plot/preprocessing_plot/full_preprocessing_figure.png
preprocessing stages plot/preprocessing_plot/full_preprocessing_figure.pdf
```

### Patch And Co-Registration Figures

```powershell
Set-Location "patch plots"
python -m pip install -r requirements.txt
python .\unified_patch_figures.py --image A7.jpg --csv A7_filtered.csv --output-dir unified_patch_figures
Set-Location ..
```

Outputs:

```text
patch plots/unified_patch_figures/exact_patch_overview.png
patch plots/unified_patch_figures/exact_patch_compact_coregistration.png
```

### Infrared Spectral Profile Figures

```powershell
Set-Location "IR profile"
python -m pip install -r requirements.txt
python .\run_all.py --data-dir data --output-dir results
Set-Location ..
```

Outputs include mean-profile tables, sampled-pixel tables, and PNG plots under:

```text
IR profile/results/plots/mean_profiles/
IR profile/results/plots/sampled_pixels/
```

### Statistical Performance Figures

Place classification report text files under `stats plots/raw_reports/` with the expected modality folders:

```text
raw_reports/
|-- img_reports/
|-- spectra_reports/
`-- multimodal_reports/
```

Then run:

```powershell
Set-Location "stats plots"
python -m pip install -r requirements.txt
python .\scripts\run_analysis.py --input-root raw_reports --output-dir outputs
Set-Location ..
```

Outputs include summary CSVs, t-test results, descriptive statistics, heatmaps, line plots, radar plots, modality boxplots, and significance plots under `stats plots/outputs/`.

## Example Figures


### Spectral preprocessing stages

<img src="preprocessing%20stages%20plot/preprocessing_plot/full_preprocessing_figure.png" alt="Full spectral preprocessing figure" width="900">

### H&E patch overview

<img src="patch%20plots/unified_patch_figures/exact_patch_overview.png" alt="H&E patch overview and selected patch locations" width="700">

### Patch and spectral co-registration

<img src="patch%20plots/unified_patch_figures/exact_patch_compact_coregistration.png" alt="Compact co-registration figure showing H&E patches and spectra" width="900">

### Mean infrared spectral profiles

<img src="IR%20profile/results/plots/mean_profiles/A7_filtered_mean_profiles.png" alt="Mean infrared spectral profiles for A7" width="900">

### Model-performance summary figures

<img src="stats%20plots/outputs/f1_score_heatmap_with_std.png" alt="F1 score heatmap with standard deviation" width="800">

## Transfer Learning

Transfer learning requires a compatible source-weights file produced by the same architecture family:

```powershell
python .\src\run_experiment.py `
  --data_dir "spectra_csv_files" `
  --image_dir "aligned_he" `
  --training_strategy transfer_learning `
  --source_weights_path "runs\run_seed_44\model_outputs\run_mode_all_ncomp32_patch5_modality_both_fusion_early_strategy_scratch_final_weights.weights.h5" `
  --source_mode all `
  --feature_extraction_epochs 15 `
  --fine_tune_epochs 10
```

Useful transfer-learning options:

```text
--freeze_backbone true
--staged_fine_tune true
--unfreeze_top_layers 12
--transfer_head_units 128
--transfer_learning_rate 1e-3
--fine_tune_learning_rate 1e-5
```

## Explainability

Explainability is controlled by these flags:

```text
--run_explainability true
--gradcam_enabled true
--shap_enabled true
--multimodal_quantification_enabled true
--explain_num_samples 24
```

Grad-CAM is used for image-supporting inputs. Spectral SHAP is used for spectra-only inputs. Multimodal perturbation quantification is used when the model input contains both image and spectral channels.

## Reproducibility Notes

- The pipeline sets Python, NumPy, and TensorFlow seeds for each run.
- Spectral preprocessing and dimensionality reduction are fit using training data only.
- Some TensorFlow and GPU operations can remain nondeterministic depending on hardware, drivers, and TensorFlow version.
`


## Validation And Troubleshooting

Syntax-check the Python files without launching training:

```powershell
python -m compileall src "data preparation\src" "IR profile\src" "stats plots\src"
```

Run a one-epoch smoke test before full experiments:

```powershell
python .\src\run_experiment.py --data_dir "spectra_csv_files" --image_dir "aligned_he" --num_runs 1 --epochs 1 --run_explainability false
```

Common issues:

- `No CSV files were found`: check that `--data_dir` points to the folder containing `.csv` files.
- `Corresponding image file was not found`: each CSV needs a matching image in `--image_dir` with the same filename stem.
- `Savitzky-Golay window is larger than available spectral length`: adjust the wavenumber ranges or lower the Savitzky-Golay window.
- `source_weights_path must be provided for transfer learning`: provide a compatible saved weights file when `--training_strategy transfer_learning` is used.

## Citation

If you use this repository, cite the associated manuscript
