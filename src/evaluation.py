from __future__ import annotations

import os
import logging
import textwrap
from itertools import cycle
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

from file_utils import ensure_dir
from model_inputs import ModelInput


LOGGER = logging.getLogger(__name__)


def _normalize_prediction_array(y_pred_prob: np.ndarray) -> np.ndarray:
    if y_pred_prob.ndim == 1:
        return np.column_stack([1.0 - y_pred_prob, y_pred_prob])
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 1:
        return np.column_stack([1.0 - y_pred_prob[:, 0], y_pred_prob[:, 0]])
    return y_pred_prob


def _has_test_samples(X_test: ModelInput) -> bool:
    return X_test.shape[0] > 0


def _save_confusion_matrix_plot(
    cm: np.ndarray,
    observed_labels: list[int],
    target_names: list[str],
    output_prefix: str,
) -> None:
    plt.figure(figsize=(10, 8))
    custom_cmap = LinearSegmentedColormap.from_list(
        'lightblue',
        ['#f0f8ff', '#add8e6', '#87ceeb', '#4682b4', '#1e3a8a'],
    )
    plt.imshow(cm, interpolation='nearest', cmap=custom_cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(observed_labels))
    wrapped_labels = ['\n'.join(textwrap.wrap(name, 12)) for name in target_names]
    plt.xticks(tick_marks, wrapped_labels, rotation=0)
    plt.yticks(tick_marks, wrapped_labels)
    threshold = cm.max() / 2.0 if cm.size else 0.0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                format(cm[row, col], 'd'),
                ha='center',
                va='center',
                color='white' if cm[row, col] > threshold else 'black',
            )
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, 'confusion_matrix.png'), dpi=200, bbox_inches='tight')
    plt.close()


def _save_roc_curve_plot(
    y_test_int: np.ndarray,
    y_pred_prob: np.ndarray,
    observed_labels: list[int],
    class_names_map: Dict[int, str],
    output_prefix: str,
) -> None:
    plt.figure(figsize=(8, 6))
    line_width = 2
    n_classes = y_pred_prob.shape[1]
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_test_int, y_pred_prob[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=line_width, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:
        color_cycle = cycle(plt.cm.get_cmap('tab10').colors)
        for class_index, color in zip(observed_labels, color_cycle):
            if class_index >= n_classes:
                continue
            fpr, tpr, _ = roc_curve((y_test_int == class_index).astype(int), y_pred_prob[:, class_index])
            roc_auc = auc(fpr, tpr)
            class_name = class_names_map.get(class_index, f'Class_{class_index}')
            plt.plot(fpr, tpr, color=color, lw=line_width, label=f'ROC {class_name} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=line_width)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_prefix, 'roc_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()


def evaluate_model(
    model: tf.keras.Model,
    X_test: ModelInput,
    y_test: np.ndarray,
    class_names_map: Dict[int, str],
    run_config_str: str,
    output_prefix: str,
) -> Tuple[Optional[np.ndarray], float]:
    del run_config_str
    ensure_dir(output_prefix)
    if not _has_test_samples(X_test):
        LOGGER.warning('Empty or malformed test data. Skipping evaluation.')
        return None, 0.0

    y_test_int = y_test.astype(np.int32)
    raw_pred_prob = model.predict(X_test, verbose=0)
    y_pred_prob = _normalize_prediction_array(raw_pred_prob)
    y_pred_int = np.argmax(y_pred_prob, axis=1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    LOGGER.info('Test loss: %.4f | Test accuracy: %.4f', test_loss, test_accuracy)

    observed_labels = sorted(np.unique(np.concatenate([y_test_int, y_pred_int])).tolist())
    target_names = [class_names_map.get(label, f'Class_{label}') for label in observed_labels]

    report = classification_report(
        y_test_int,
        y_pred_int,
        labels=observed_labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(os.path.join(output_prefix, 'classification_report.csv'))

    cm = confusion_matrix(y_test_int, y_pred_int, labels=observed_labels)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(os.path.join(output_prefix, 'confusion_matrix.csv'))
    _save_confusion_matrix_plot(cm, observed_labels, target_names, output_prefix)

    try:
        _save_roc_curve_plot(y_test_int, y_pred_prob, observed_labels, class_names_map, output_prefix)
    except Exception as exc:  # pragma: no cover
        LOGGER.exception('Failed to generate ROC curve: %s', exc)
        plt.close()

    return y_pred_prob, float(test_accuracy)
