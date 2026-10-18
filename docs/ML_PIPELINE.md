# Phase 2 ML Pipeline

## Pipeline

~~~text
Food-101
    → selected classes
    → train / validation / official test
    → image transforms
    → PyTorch Dataset
    → DataLoader
    → future model training
~~~

The final step is a Phase 3 consumer. Phase 2 stops after producing verified image
batches and does not train a neural network.

## Loading and filtering

ml/src/download_dataset.py prepares the dataset under ml/data/raw/ using
torchvision.datasets.Food101. ml/src/dataset.py loads both official splits and
validates metadata, class agreement, image paths, file existence, and official split
isolation.

The selected classes are filtered by their Food-101 names. Food101Subset stores
indices into the original Food101 object, so no image files are duplicated. Each
subset exposes classes, class_to_idx, indices, targets, and image_paths.

The shared application label order is:

~~~text
0 pizza
1 hamburger
2 sushi
3 steak
4 ice_cream
5 donuts
6 omelette
7 french_fries
8 chicken_wings
9 spaghetti_bolognese
~~~

## Split behavior

The default DataConfig is:

~~~text
image_size=224
batch_size=32
num_workers=0
random_seed=42
validation_fraction=0.2
~~~

Each selected official training class is split into 600 training and 150 validation
samples. The selected official test samples remain all 250 per class. Paths are
sorted before deterministic per-class shuffling. SHA-256 image-content groups are
kept together so identical images cannot cross a split boundary.

## Transforms

ml/src/preprocessing.py returns separate transform pipelines.

Training transform:

1. RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1))
2. RandomHorizontalFlip(p=0.5)
3. ToTensor()
4. ImageNet normalization

Evaluation transform:

1. Resize shorter edge to 256 with bilinear interpolation and antialiasing
2. CenterCrop(224)
3. ToTensor()
4. ImageNet normalization

The normalization constants are mean (0.485, 0.456, 0.406) and standard deviation
(0.229, 0.224, 0.225). Images are converted to RGB before transformation.
Validation and test transforms contain no random operations.

## Dataset and DataLoader behavior

Each dataset returns an image and label:

- image: CPU torch.float32 tensor with shape [3, 224, 224] by default;
- label: CPU torch.int64 scalar using application labels 0–9.

Each DataLoader returns a batch:

- images: [batch_size, 3, 224, 224];
- labels: [batch_size].

The training loader shuffles. Validation and test loaders preserve deterministic
dataset order. All loaders use drop_last=False and pin_memory=False. The final
batch may be smaller than the configured batch size.

create_data_pipeline seeds Python, NumPy, and PyTorch generators and returns the
three datasets and three loaders. create_dataloaders supports configurable worker
counts and uses spawn when workers are enabled. Keep the if __name__ == "__main__"
guard in scripts that create loaders on macOS.

## Configuration

DataConfig accepts:

- root: Food-101 root directory;
- classes: ordered class names;
- image_size: positive output size;
- batch_size: positive batch size;
- num_workers: zero or more workers;
- random_seed: deterministic split and loader seed;
- validation_fraction: fraction of each selected training class.

The CLI scripts expose the same root, image-size, batch-size, worker, seed, and
validation-fraction options. The verification CLI intentionally requires the agreed
ten-class development configuration.

## Verification procedure

verify_pipeline.py verifies:

- selected classes, contiguous labels, and class mappings;
- complete train/validation partition and untouched official test selection;
- deterministic splits for repeated seed 42;
- unchanged test membership when the seed changes;
- path overlap and identical image bytes across splits;
- deterministic evaluation transforms;
- CPU tensor device, shapes, dtypes, finite values, and normalized bounds;
- first-batch reproducibility;
- complete iteration through all selected images with --all-batches.

The verified command was:

~~~bash
python ml/src/verify_pipeline.py --all-batches
~~~

It decoded and transformed 10,000 selected images and produced batches with shape
[32, 3, 224, 224] and labels [32]. A second real-data run with --num-workers 2
also passed. JSON reports are written under the ignored ml/data/processed/ directory.

## macOS and device boundary

Phase 2 does not select MPS, CUDA, or CPU for model execution. It creates CPU
tensors so Phase 3 can choose an appropriate device, typically mps, then cuda,
then cpu according to its own runtime policy. Zero workers are the default for
simple macOS behavior; positive workers are supported through spawn and were
verified with two workers.
