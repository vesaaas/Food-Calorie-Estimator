"""TensorFlow adapter over the already validated Phase 5 split datasets."""

from collections import Counter
from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

if __package__:
    from .dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig, seed_random_generators
    from .phase5_data import MixedFoodDataset, create_phase5_data_pipeline
else:
    from dataset import DATA_DIR, DEVELOPMENT_CLASSES, PHASE5_CLASSES, DataConfig, seed_random_generators
    from phase5_data import MixedFoodDataset, create_phase5_data_pipeline


IMAGE_SIZE = 224
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 1, 3)


def _model_input_from_torch_image(image: torch.Tensor) -> np.ndarray:
    """Invert the verified Phase 2 normalization for Keras MobileNetV3 input."""
    if not isinstance(image, torch.Tensor) or tuple(image.shape) != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected a normalized [3,{IMAGE_SIZE},{IMAGE_SIZE}] tensor")
    pixels = image.detach().cpu().numpy().transpose(1, 2, 0)
    pixels = (pixels * IMAGENET_STD + IMAGENET_MEAN) * 255.0
    return np.ascontiguousarray(np.clip(pixels, 0.0, 255.0), dtype=np.float32)


def create_tf_phase5_datasets(
    *, local_root=None, food101_root: Path = DATA_DIR, batch_size: int = 32,
    seed: int = 42, image_size: int = IMAGE_SIZE,
):
    """Return deterministic tf.data views with the original Phase 5 split semantics.

    Phase 5 remains the authority for Food-101 stratification, official test
    membership, TurkishFoods-25 source splits, explicit label mapping, and leakage
    checks. This adapter reuses those transformed samples and reverses only the
    ImageNet normalization required by Keras MobileNetV3's built-in [-1,1] layer.
    """
    if image_size != IMAGE_SIZE:
        raise ValueError(f"MobileNetV3 contract requires {IMAGE_SIZE}x{IMAGE_SIZE} images")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if tuple(PHASE5_CLASSES[:10]) != DEVELOPMENT_CLASSES or PHASE5_CLASSES[10:] != (
        "borek", "kebap", "sarma",
    ):
        raise ValueError("Phase 5 class order has changed")
    config = DataConfig(
        root=food101_root, classes=DEVELOPMENT_CLASSES, image_size=image_size,
        batch_size=batch_size, num_workers=0, random_seed=seed,
    )
    seed_random_generators(seed)
    datasets, _ = create_phase5_data_pipeline(config, local_root=local_root) if local_root else create_phase5_data_pipeline(config)

    counts = {
        split: Counter(dataset.targets)
        for split, dataset in datasets.items()
    }
    class_counts = {
        split: {label: counts[split][index] for index, label in enumerate(PHASE5_CLASSES)}
        for split in datasets
    }
    train_counts = class_counts["train"]
    weights = np.asarray(
        [len(datasets["train"]) / (len(PHASE5_CLASSES) * train_counts[label]) for label in PHASE5_CLASSES],
        dtype=np.float32,
    )
    weights /= weights.mean()

    def build(dataset, split):
        def generator():
            for index in range(len(dataset)):
                image, label = dataset[index]
                yield _model_input_from_torch_image(image), np.int32(label)

        signature = (
            tf.TensorSpec((image_size, image_size, 3), tf.float32),
            tf.TensorSpec((), tf.int32),
        )
        result = tf.data.Dataset.from_generator(generator, output_signature=signature)
        if split == "train":
            class_weight_tensor = tf.constant(weights, dtype=tf.float32)
            result = result.shuffle(
                len(dataset), seed=seed, reshuffle_each_iteration=True,
            ).map(
                lambda image, label: (image, label, tf.gather(class_weight_tensor, label)),
                num_parallel_calls=1, deterministic=True,
            )
        return result.batch(batch_size, drop_remainder=False).prefetch(1)

    return (
        {split: build(dataset, split) for split, dataset in datasets.items()},
        datasets,
        {"class_counts": class_counts, "class_weights": weights.tolist()},
    )
