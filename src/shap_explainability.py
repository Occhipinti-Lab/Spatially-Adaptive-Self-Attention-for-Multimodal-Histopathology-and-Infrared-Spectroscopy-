from __future__ import annotations

import logging
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from file_utils import ensure_dir
from model_inputs import ModelInput

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover
    shap = None


LOGGER = logging.getLogger(__name__)


def spectra_shap_summary(
    model: tf.keras.Model,
    X: ModelInput,
    y_true: np.ndarray,
    feature_names: Sequence[str],
    output_dir: str,
    max_background: int = 16,
    max_explain: int = 8,
) -> None:
    del y_true
    if shap is None:
        LOGGER.warning('SHAP is not installed. Skipping SHAP spectra analysis.')
        return
    if X.shape[0] == 0:
        return

    ensure_dir(output_dir)
    background = X[: min(max_background, X.shape[0])]
    explain_samples = X[: min(max_explain, X.shape[0])]

    try:
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(explain_samples)
    except Exception as exc:  # pragma: no cover
        LOGGER.warning('SHAP GradientExplainer failed: %s', exc)
        return

    if isinstance(shap_values, list):
        shap_array = np.asarray(shap_values)
        shap_array = np.mean(np.abs(shap_array), axis=0)
    else:
        shap_array = np.abs(np.asarray(shap_values))

    if shap_array.ndim == 5:
        importance = shap_array.mean(axis=(0, 1, 2))
    elif shap_array.ndim == 4:
        importance = shap_array.mean(axis=(0, 1, 2))
    else:
        LOGGER.warning('Unexpected SHAP tensor shape %s. Skipping SHAP summary.', shap_array.shape)
        return

    limit = min(len(feature_names), importance.shape[0])
    summary_df = pd.DataFrame({'feature': list(feature_names)[:limit], 'mean_abs_shap': importance[:limit]})
    summary_df.sort_values('mean_abs_shap', ascending=False, inplace=True)
    summary_df.to_csv(os.path.join(output_dir, 'spectra_shap_summary.csv'), index=False)

    top_df = summary_df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top_df['feature'], top_df['mean_abs_shap'])
    plt.xlabel('Mean |SHAP|')
    plt.ylabel('Feature')
    plt.title('Top Spectral SHAP Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spectra_shap_top20.png'), dpi=200, bbox_inches='tight')
    plt.close()
