"""
The input tensor contains RGB and/or spectral channels.
RGB and spectra receive only independent 1x1 channel projections; they are then
fused before any spatial convolution.  Equal projection widths remove the raw
3-vs-K channel-count advantage while retaining genuine early spatial fusion.
"""

from __future__ import annotations

from typing import Sequence

import tensorflow as tf
from tensorflow.keras import layers

MODEL_ARCHITECTURE = "CNN-SASA"


@tf.keras.utils.register_keras_serializable(package="MBEF")
class ChannelSlice(layers.Layer):
    def __init__(self, start: int, end: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self.start = int(start)
        self.end = None if end is None else int(end)

    def call(self, inputs):
        return inputs[..., self.start : self.end]

    def get_config(self):
        return {**super().get_config(), "start": self.start, "end": self.end}


@tf.keras.utils.register_keras_serializable(package="CNN-SASA")
class CoRegisteredAugmentation(layers.Layer):
    """Apply one geometric transform to all co-registered features."""

    def call(self, inputs, training=None):
        if training is False:
            return inputs

        def augment():
            x = tf.image.random_flip_left_right(inputs)
            x = tf.image.random_flip_up_down(x)
            return tf.image.rot90(x, tf.random.uniform((), 0, 4, dtype=tf.int32))

        if training is None:
            return inputs
        return tf.cond(tf.cast(training, tf.bool), augment, lambda: tf.identity(inputs))


@tf.keras.utils.register_keras_serializable(package="CNN-SASA")
class OpticalDensityJitter(layers.Layer):
    """Perturb H&E optical density per sample without moving registered pixels."""

    def __init__(self, strength: float = 0.08, **kwargs):
        super().__init__(**kwargs)
        self.strength = float(strength)
        if self.strength < 0:
            raise ValueError("optical-density jitter strength must be non-negative")

    def call(self, inputs, training=None):
        if training is False or self.strength == 0:
            return inputs

        def augment():
            rgb = tf.clip_by_value(inputs, 1.0 / 255.0, 1.0)
            density = -tf.math.log(rgb)
            shape = (tf.shape(inputs)[0], 1, 1, 3)
            scale = tf.random.uniform(
                shape, 1.0 - self.strength, 1.0 + self.strength, dtype=inputs.dtype
            )
            offset = tf.random.uniform(
                shape, -self.strength / 2.0, self.strength / 2.0, dtype=inputs.dtype
            )
            return tf.clip_by_value(tf.exp(-(density * scale + offset)), 0.0, 1.0)

        if training is None:
            return inputs
        return tf.cond(tf.cast(training, tf.bool), augment, lambda: tf.identity(inputs))

    def get_config(self):
        return {**super().get_config(), "strength": self.strength}


@tf.keras.utils.register_keras_serializable(package="CNN-SASA")
class BalancedModalityDropout(layers.Layer):

    def __init__(self, rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.rate = float(rate)
        if not 0 <= self.rate < 1:
            raise ValueError("modality dropout rate must be in [0, 1)")

    def call(self, inputs, training=None):
        rgb, spectra = inputs
        if training is False or self.rate == 0:
            return rgb, spectra

        def dropped():
            batch = tf.shape(rgb)[0]
            draw = tf.random.uniform((batch, 1, 1, 1), dtype=rgb.dtype)
            drop_rgb = tf.cast(draw < self.rate / 2.0, rgb.dtype)
            drop_spectra = tf.cast(
                (draw >= self.rate / 2.0) & (draw < self.rate), spectra.dtype
            )
            return rgb * (1.0 - drop_rgb), spectra * (1.0 - drop_spectra)

        if training is None:
            return rgb, spectra
        return tf.cond(tf.cast(training, tf.bool), dropped, lambda: (rgb, spectra))

    def get_config(self):
        return {**super().get_config(), "rate": self.rate}


@tf.keras.utils.register_keras_serializable(package="MBEF")
class GatedEarlyFusion(layers.Layer):

    def __init__(self, gate_floor: float = 0.20, **kwargs):
        super().__init__(**kwargs)
        self.gate_floor = float(gate_floor)
        if not 0 <= self.gate_floor < 1:
            raise ValueError("gate_floor must be in [0, 1)")
        self.gate_dense = None

    def build(self, input_shape):
        rgb_shape, spectra_shape = input_shape
        if rgb_shape[-1] != spectra_shape[-1]:
            raise ValueError("fusion inputs must have equal latent widths")
        width = int(rgb_shape[-1])
        self.gate_dense = layers.Dense(2 * width, activation="sigmoid", name="gates")
        super().build(input_shape)

    def call(self, inputs):
        rgb, spectra = inputs
        rgb_mean = tf.reduce_mean(rgb, axis=(1, 2))
        spectra_mean = tf.reduce_mean(spectra, axis=(1, 2))
        gates = self.gate_dense(tf.concat([rgb_mean, spectra_mean], axis=-1))
        floor = tf.cast(self.gate_floor, gates.dtype)
        gates = floor + (1.0 - floor) * gates
        rgb_gate, spectra_gate = tf.split(gates, 2, axis=-1)
        rgb_gate = rgb_gate[:, None, None, :]
        spectra_gate = spectra_gate[:, None, None, :]
        return tf.concat([rgb * rgb_gate, spectra * spectra_gate], axis=-1)

    def get_config(self):
        return {**super().get_config(), "gate_floor": self.gate_floor}


@tf.keras.utils.register_keras_serializable(package="MBEF")
class SpatialAttention(layers.Layer):
    """Multi-head spatial self-attention."""

    def __init__(self, num_heads: int = 4, dropout: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.attention = None
        self.norm = layers.LayerNormalization(epsilon=1e-5, name="norm")

    def build(self, input_shape):
        channels = int(input_shape[-1])
        if channels % self.num_heads:
            raise ValueError("attention channels must be divisible by num_heads")
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=channels // self.num_heads,
            dropout=self.dropout,
            name="mha",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        shape = tf.shape(inputs)
        tokens = tf.reshape(inputs, (shape[0], shape[1] * shape[2], shape[3]))
        attended = self.attention(tokens, tokens, training=training)
        result = self.norm(tokens + attended)
        return tf.reshape(result, shape)

    def get_config(self):
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "dropout": self.dropout,
        }


@tf.keras.utils.register_keras_serializable(package="MBEF")
class SparseSmoothedCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, num_classes: int, label_smoothing: float = 0.0, name=None):
        super().__init__(name=name or "sparse_smoothed_categorical_crossentropy")
        self.num_classes = int(num_classes)
        self.label_smoothing = float(label_smoothing)

    def call(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        one_hot = tf.one_hot(y_true, self.num_classes)
        return tf.keras.losses.categorical_crossentropy(
            one_hot, y_pred, label_smoothing=self.label_smoothing
        )

    def get_config(self):
        return {
            **super().get_config(),
            "num_classes": self.num_classes,
            "label_smoothing": self.label_smoothing,
        }


def _project(x, width, name, l2):
    x = layers.Conv2D(
        width,
        1,
        use_bias=False,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name=f"{name}_projection",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_projection_norm")(x)
    x = layers.Activation("gelu", name=f"{name}_projection_gelu")(x)
    return x


def _conv_block(x, width, index, dropout, l2, heads):
    residual = x
    x = layers.SeparableConv2D(
        width,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=tf.keras.regularizers.l2(l2),
        pointwise_regularizer=tf.keras.regularizers.l2(l2),
        name=f"encoder_{index}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"encoder_{index}_norm")(x)
    x = layers.Activation("gelu", name=f"encoder_{index}_gelu")(x)
    x = layers.SpatialDropout2D(dropout, name=f"encoder_{index}_dropout")(x)
    if int(residual.shape[-1]) != int(width):
        residual = layers.Conv2D(width, 1, use_bias=False, name=f"encoder_{index}_skip")(residual)
    x = layers.Add(name=f"encoder_{index}_residual")([x, residual])
    x = SpatialAttention(
        num_heads=heads,
        dropout=min(dropout, 0.15),
        name=f"encoder_{index}_sasa",
    )(x)
    return x


def _auxiliary_head(x, num_classes, name, dropout, l2):
    x = layers.GlobalAveragePooling2D(name=f"{name}_pool")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name=name,
    )(x)


def build_model(
    input_shape: Sequence[int],
    num_classes: int,
    modality: str,
    spectral_channels: int,
    latent_width: int = 48,
    encoder_widths: Sequence[int] = (64, 96, 128),
    attention_heads: int = 4,
    dropout: float = 0.2,
    head_dropout: float = 0.35,
    modality_dropout: float = 0.2,
    optical_density_jitter: float = 0.08,
    l2: float = 1e-4,
    augment: bool = True,
):
    """Build a training model."""
    input_shape = tuple(map(int, input_shape))
    if len(input_shape) != 3 or min(input_shape) <= 0:
        raise ValueError("input_shape must be positive (H,W,C)")
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    expected = (
        (3 if modality in {"image", "both"} else 0)
        + (spectral_channels if modality in {"spectra", "both"} else 0)
    )
    if input_shape[-1] != expected:
        raise ValueError(f"expected {expected} channels for {modality}, got {input_shape[-1]}")

    inputs = layers.Input(input_shape, name="early_fusion_input")
    x_input = CoRegisteredAugmentation(name="comsa")(
        inputs
    ) if augment else inputs

    outputs = {}
    if modality == "both":
        rgb = ChannelSlice(0, 3, name="rgb_channels")(x_input)
        rgb = OpticalDensityJitter(
            optical_density_jitter if augment else 0.0,
            name="rgb_optical_density_jitter",
        )(rgb)
        spectra = ChannelSlice(3, 3 + spectral_channels, name="spectral_channels")(x_input)
        rgb = _project(rgb, latent_width, "rgb", l2)
        spectra = _project(spectra, latent_width, "spectra", l2)
        rgb_aux_map, spectra_aux_map = rgb, spectra
        rgb, spectra = BalancedModalityDropout(
            modality_dropout, name="balanced_modality_dropout"
        )([rgb, spectra])
        x = GatedEarlyFusion(gate_floor=0.20, name="gated_early_fusion")(
            [rgb, spectra]
        )
        # First and every subsequent spatial kernel sees both modalities.
        x = layers.Conv2D(
            encoder_widths[0],
            1,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(l2),
            name="early_fusion_mixer",
        )(x)
        outputs["rgb_aux"] = _auxiliary_head(
            rgb_aux_map, num_classes, "rgb_aux", head_dropout, l2
        )
        outputs["spectra_aux"] = _auxiliary_head(
            spectra_aux_map, num_classes, "spectra_aux", head_dropout, l2
        )
    else:
        feature_end = 3 if modality == "image" else spectral_channels
        x = ChannelSlice(0, feature_end, name=f"{modality}_channels")(x_input)
        if modality == "image":
            x = OpticalDensityJitter(
                optical_density_jitter if augment else 0.0,
                name="rgb_optical_density_jitter",
            )(x)
        x = _project(x, latent_width, modality, l2)

    for index, width in enumerate(encoder_widths, start=1):
        x = _conv_block(x, int(width), index, dropout, l2, attention_heads)
        if index < len(encoder_widths):
            x = layers.MaxPooling2D(2, padding="same", name=f"encoder_{index}_pool")(x)

    x = layers.Activation("linear", name="fusion_feature_map")(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(
        128,
        activation="gelu",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name="fused_embedding",
    )(x)
    x = layers.Dropout(head_dropout, name="fused_dropout")(x)
    outputs["fused"] = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        name="fused",
    )(x)
    return tf.keras.Model(
        inputs,
        outputs if modality == "both" else outputs["fused"],
        name="MBEF_SASA",
    )


def inference_model(training_model: tf.keras.Model) -> tf.keras.Model:

    return tf.keras.Model(
        training_model.input,
        training_model.get_layer("fused").output,
        name=f"{training_model.name}_inference",
    )


CUSTOM_OBJECTS = {
    cls.__name__: cls
    for cls in (
        ChannelSlice,
        CoRegisteredAugmentation,
        OpticalDensityJitter,
        BalancedModalityDropout,
        GatedEarlyFusion,
        SpatialAttention,
        SparseSmoothedCategoricalCrossentropy,
    )
}
