from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd

from file_utils import ensure_dir


def plot_training_history(history_df: pd.DataFrame, plot_config_str: str, output_filename: str) -> None:
    ensure_dir(os.path.dirname(output_filename))
    epochs = range(1, len(history_df) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    if 'accuracy' in history_df and 'val_accuracy' in history_df:
        plt.plot(epochs, history_df['accuracy'], label='Training Accuracy', marker='.')
        plt.plot(epochs, history_df['val_accuracy'], label='Validation Accuracy', marker='.')
        plt.legend()
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.4)

    plt.subplot(1, 2, 2)
    if 'loss' in history_df and 'val_loss' in history_df:
        plt.plot(epochs, history_df['loss'], label='Training Loss', marker='.')
        plt.plot(epochs, history_df['val_loss'], label='Validation Loss', marker='.')
        plt.legend()
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.4)

    plt.suptitle(plot_config_str)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=200, bbox_inches='tight')
    plt.close()
