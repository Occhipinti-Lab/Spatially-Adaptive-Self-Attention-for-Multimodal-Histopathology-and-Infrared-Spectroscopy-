
from __future__ import annotations

import logging
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

from pipeline_config import PipelineConfig


LOGGER = logging.getLogger(__name__)
TensorShapeLike = Tuple[int, int, int]


class SpatiallyAdaptiveAttentionLayer(layers.Layer):
    def __init__(self, epsilon: float = 1e-5, residual_scale: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.residual_scale = residual_scale
        self.layer_norm = layers.LayerNormalization(epsilon=epsilon)

    def build(self, input_shape):
        self.height = int(input_shape[1])
        self.width = int(input_shape[2])
        self.channels = int(input_shape[3])
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        sequence_length = self.height * self.width
        reshaped = tf.reshape(inputs, (batch_size, sequence_length, self.channels))
        attention_scores = tf.matmul(reshaped, reshaped, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores / tf.sqrt(tf.cast(self.channels, tf.float32)), axis=-1)
        attended = tf.matmul(attention_scores, reshaped)
        attended = tf.reshape(attended, (batch_size, self.height, self.width, self.channels))
        residual = inputs + attended
        normalized = self.layer_norm(residual)
        return residual + (normalized * self.residual_scale)

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon, 'residual_scale': self.residual_scale})
        return config


def spatially_adaptive_self_attention(input_tensor: tf.Tensor, layer_name: str = '') -> tf.Tensor:
    prefix = f'sasa_{layer_name}_' if layer_name else 'sasa_'
    return SpatiallyAdaptiveAttentionLayer(name=f'{prefix}attention')(input_tensor)


def create_conv_block(
    input_tensor: tf.Tensor,
    filters: int,
    kernel_size: Tuple[int, int],
    block_name: str,
    use_attention: bool = True,
    dropout_rate: float = 0.2,
    l2_reg: float = 1e-3,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        padding='same',
        activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name=f'conv_{block_name}',
    )(input_tensor)
    x = layers.Dropout(dropout_rate, name=f'dropout_{block_name}')(x)
    if use_attention:
        x = spatially_adaptive_self_attention(x, layer_name=f'block_{block_name}')
    return x


def build_spatially_adaptive_patch_model(
    input_shape: Tuple[int, int, int],
    num_classes: int,
    l2_reg: float = 1e-3,
    use_attention: bool = True,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name='input_patch')
    x = create_conv_block(inputs, filters=32, kernel_size=(3, 3), block_name='1', use_attention=use_attention, l2_reg=l2_reg)
    x = create_conv_block(x, filters=16, kernel_size=(3, 3), block_name='2', use_attention=use_attention, l2_reg=l2_reg)
    x = layers.MaxPooling2D((2, 2), padding='same', name='pool_2')(x)
    x = create_conv_block(x, filters=16, kernel_size=(3, 3), block_name='3', use_attention=use_attention, l2_reg=l2_reg)
    x = layers.MaxPooling2D((2, 2), padding='same', name='pool_3')(x)
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = layers.Dropout(0.4, name='classifier_dropout')(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer=regularizers.l2(l2_reg), name='feature_embedding')(x)
    x = layers.Dropout(0.25, name='feature_embedding_dropout')(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    else:
        outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    return models.Model(inputs=inputs, outputs=outputs, name='SpatialAdaptiveMultimodalCNN')


def build_model(input_shape: TensorShapeLike, num_classes: int, fusion_strategy: str, l2_reg: float = 1e-3) -> tf.keras.Model:
    if fusion_strategy not in {'early', 'image_only', 'spectra_only'}:
        raise ValueError(f'Unsupported fusion_strategy: {fusion_strategy}')
    return build_spatially_adaptive_patch_model(input_shape=input_shape, num_classes=num_classes, l2_reg=l2_reg)


def build_transfer_model(
    input_shape: TensorShapeLike,
    config: PipelineConfig,
    l2_reg: float = 1e-3,
) -> tf.keras.Model:
    LOGGER.info('Building transfer-learning source model with source output units: %d', config.source_output_units)
    source_model = build_model(
        input_shape=input_shape,
        num_classes=config.source_output_units,
        fusion_strategy=config.fusion_strategy,
        l2_reg=l2_reg,
    )
    source_model.load_weights(config.source_weights_path)
    LOGGER.info('Loaded source weights from: %s', config.source_weights_path)

    feature_embedding = source_model.get_layer('feature_embedding').output
    x = layers.Dropout(0.30, name='transfer_head_dropout_1')(feature_embedding)
    x = layers.Dense(
        config.transfer_head_units,
        activation='swish',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='transfer_head_dense_1',
    )(x)
    x = layers.Dropout(0.25, name='transfer_head_dropout_2')(x)

    if config.target_output_units == 1:
        output = layers.Dense(1, activation='sigmoid', name='output_transfer')(x)
    else:
        output = layers.Dense(config.target_output_units, activation='softmax', name='output_transfer')(x)

    transfer_model = models.Model(inputs=source_model.inputs, outputs=output, name=f'{source_model.name}_Transfer')

    if config.freeze_backbone:
        freeze_backbone_layers(transfer_model)
    return transfer_model


def freeze_backbone_layers(model: tf.keras.Model) -> None:
    for layer in model.layers:
        if layer.name.startswith('transfer_head_') or layer.name == 'output_transfer':
            layer.trainable = True
        else:
            layer.trainable = False


def unfreeze_top_backbone_layers(model: tf.keras.Model, top_n: int) -> int:
    backbone_layers = [
        layer for layer in model.layers
        if not (layer.name.startswith('transfer_head_') or layer.name == 'output_transfer')
    ]
    if top_n <= 0:
        for layer in backbone_layers:
            layer.trainable = True
        return len(backbone_layers)

    selected = backbone_layers[-top_n:]
    for layer in selected:
        layer.trainable = True
    return len(selected)


def count_trainable_and_frozen_params(model: tf.keras.Model) -> Tuple[int, int]:
    trainable = int(sum(tf.keras.backend.count_params(weight) for weight in model.trainable_weights))
    frozen = int(sum(tf.keras.backend.count_params(weight) for weight in model.non_trainable_weights))
    return trainable, frozen


def get_gradcam_target_layer_name(config: PipelineConfig) -> str | None:
    if config.training_modality == 'spectra' or config.fusion_strategy == 'spectra_only':
        return None
    return 'conv_3'
