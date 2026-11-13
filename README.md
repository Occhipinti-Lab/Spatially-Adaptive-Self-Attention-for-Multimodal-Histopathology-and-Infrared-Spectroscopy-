# CNN-SASA: Spatially Adaptive Self-Attention for Multimodal Histopathology and Infrared Spectroscopy Fusion in Breast Cancer

This project uses convolutional neural networks to classify breast tissue from co-registered haematoxylin and eosin (H&E) images and Fourier-transform infrared (FTIR) spectroscopy. It supports image-only, spectra-only, and multimodal early-fusion experiments using full preprocessed spectra.

## Models

- **CNN-SASA:** CNN incorporating spatial self-attention within the encoder.
- **CNN Baseline:** Corresponding CNN without spatial self-attention, intended as the attention ablation.


## Setup

Requires Python with the packages listed in `requirements.txt`. Keep the `consolidated/` directory intact and run commands from its parent directory.

```powershell
python -m pip install -r consolidated/requirements.txt
```

Organise the registered files as follows (spectra files must contain `Pixel_X`, `Pixel_Y`, `Label`, and `wn_*` spectral columns):

```text
project/
├── consolidated/
│   ├── __main__.py
│   ├── config.py
│   ├── model.py
│   └── ...
├── breast_spectral_data/
│   ├── spectra_file
│   └── ...
└── breast_images/
    ├── H&E_core_image
    └── ...
```

Image filenames must correspond to their spectral file filenames. Set an appropriate `--patient-id-regex` if patient IDs cannot be identified correctly from the filename prefix before the first underscore.

## Run

The following examples use PowerShell. Replace the two data paths with your own.

**Multimodal CNN-SASA (H&E + FTIR):**

```powershell
python -m consolidated --data-dir "./breast_spectral_data" --image-dir "./breast_images" --modality both --split-strategy patient --final-test false
```

**H&E-only or FTIR-only CNN-SASA:**

```powershell
python -m consolidated --data-dir "./breast_spectral_data" --image-dir "./breast_images" --modality image --split-strategy patient --final-test false
python -m consolidated --data-dir "./breast_spectral_data" --modality spectra --split-strategy patient --final-test false
```

**Run all three modality configurations with matching patient-level split seeds:**

```powershell
python -m consolidated --data-dir "./breast_spectral_data" --image-dir "./breast_images" --modality both --split-strategy patient --run-ablation-suite true --final-test false
```

**Final test evaluation (after choosing settings using validation results):**

```powershell
python -m consolidated --data-dir "./breast_spectral_data" --image-dir "./breast_images" --modality both --split-strategy patient --run-ablation-suite false --final-test true
```

Optional arguments include `--epochs 35`, `--batch-size 32`, `--patch-size 19`, `--split-seeds 128 512 924 1002 1016 1024 1523 1666 3303 4025`, `--run-explainability false`, and `--run-shap false`.

## Outputs

Outputs are saved by task, split strategy, modality, and seed. They include patient split manifests, training histories, classification reports, confusion matrices, performance metrics, and explainability results (Grad-CAM, integrated gradients, and spectral SHAP where applicable).

## Experimental protocol

Patient-level splitting is recommended to prevent overlapping patients across training, validation, and test sets. Spectral scaling is fitted on training data only. Compare models using identical patient partitions and report results across seeds. Keep final-test evaluation separate from model selection.
