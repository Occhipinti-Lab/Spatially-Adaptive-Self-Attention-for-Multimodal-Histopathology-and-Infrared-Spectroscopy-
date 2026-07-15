from __future__ import annotations

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from file_utils import ensure_dir
from model_architecture import get_gradcam_target_layer_name
from model_inputs import ModelInput, get_sample_count, slice_model_samples
from pipeline_config import PipelineConfig


LOGGER = logging.getLogger(__name__)


def generate_gradcam_heatmaps(
    model: tf.keras.Model,
    X: ModelInput,
    y_true: np.ndarray,
    config: PipelineConfig,
    output_dir: str,
    max_samples: int,
) -> None:
    layer_name = get_gradcam_target_layer_name(config)
    if layer_name is None:
        LOGGER.info('Grad-CAM skipped because no image-supporting layer exists for the current modality.')
        return
    try:
        target_layer = model.get_layer(layer_name)
    except ValueError:
        LOGGER.warning('Grad-CAM target layer %s was not found in the model. Skipping Grad-CAM.', layer_name)
        return

    ensure_dir(output_dir)
    sample_count = min(max_samples, get_sample_count(X))
    if sample_count == 0:
        return

    indices = np.arange(sample_count)
    X_samples = slice_model_samples(X, indices)
    y_samples = y_true[indices].astype(np.int32)

    grad_model = tf.keras.models.Model(inputs=model.inputs, outputs=[target_layer.output, model.output])

    for local_index in range(sample_count):
        current_input = slice_model_samples(X_samples, np.asarray([local_index]))
        class_index = int(y_samples[local_index])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(current_input, training=False)
            if predictions.shape[-1] == 1:
                target = predictions[:, 0]
            else:
                target = predictions[:, class_index]
        gradients = tape.gradient(target, conv_outputs)
        if gradients is None:
            continue
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        conv_map = conv_outputs[0]
        heatmap = tf.reduce_sum(conv_map * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        if float(max_value) > 0:
            heatmap = heatmap / max_value
        heatmap_np = heatmap.numpy()

        base_image = current_input[0][..., :config.num_image_channels]
        heatmap_resized = tf.image.resize(heatmap_np[..., np.newaxis], base_image.shape[:2]).numpy().squeeze()

        plt.figure(figsize=(4, 4))
        plt.imshow(np.clip(base_image, 0, 1))
        plt.imshow(heatmap_resized, cmap='jet', alpha=0.45)
        plt.axis('off')
        plt.title(f'Grad-CAM sample {local_index} | class {class_index}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gradcam_sample_{local_index}.png'), dpi=200, bbox_inches='tight')
        plt.close()
