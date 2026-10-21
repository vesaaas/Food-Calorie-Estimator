"""Keras MobileNetV3 Small model definition for the 13-class mobile model."""

import tensorflow as tf

if __package__:
    from .dataset import PHASE5_CLASSES
else:
    from dataset import PHASE5_CLASSES


MODEL_NAME = "mobilenet_v3_small"
IMAGE_SIZE = 224
INPUT_DTYPE = "float32"


def create_model(*, pretrained: bool = True) -> tf.keras.Model:
    """Build a 13-way MobileNetV3 Small with the official ImageNet initialization.

    Keras' built-in preprocessing layer maps raw RGB [0,255] float input to [-1,1].
    All backbone layers are trainable for fine-tuning; the new output head is a
    freshly initialized softmax layer in the existing Phase 5 class order.
    """
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), include_top=False,
        weights="imagenet" if pretrained else None, include_preprocessing=True,
        pooling="avg",
    )
    backbone.trainable = True
    inputs = tf.keras.Input((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=tf.float32, name="image_rgb_0_255")
    # Leave training mode to the outer Model call so BatchNormalization uses
    # moving statistics during validation, export, and mobile inference.
    features = backbone(inputs)
    features = tf.keras.layers.Dropout(0.2, name="head_dropout")(features)
    outputs = tf.keras.layers.Dense(
        len(PHASE5_CLASSES), activation="softmax", name="class_probabilities",
    )(features)
    return tf.keras.Model(inputs, outputs, name=MODEL_NAME)
