# ML Pipeline

## Pipeline

~~~text
Food-101
    → selected classes
    → train / validation / official test
    → image transforms
    → PyTorch Dataset
    → DataLoader
    → Phase 3 model training / validation
~~~

The final step is implemented as a Phase 3 consumer. Phase 2 stops after producing
verified image batches; its dataset and preprocessing implementation is unchanged.

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

## Phase 3 training

`model.py` creates torchvision EfficientNet-B0 with explicitly pinned
`EfficientNet_B0_Weights.IMAGENET1K_V1` initialization and replaces its final linear
layer with ten outputs. The class count/order comes from the Phase 2 development
configuration; no second class list is defined. All backbone and classifier
parameters are trainable. Dropout remains 0.2. Phase 2 transforms are reused rather
than replaced by a weights-specific transform pipeline.

`runtime.py` exposes `select_device("auto" | "mps" | "cuda" | "cpu")`. Automatic
selection prefers MPS, then CUDA, then CPU; an unavailable explicit request fails
clearly. The training loop simply moves images/labels to the selected device.
`seed_training` reuses Phase 2 Python/NumPy/PyTorch seeding, which also seeds
accelerators, and disables cuDNN benchmarking while requesting deterministic cuDNN
behavior. Seeds do not guarantee bitwise equality across backends or library versions.

`train.py` calls the existing `create_data_pipeline(DataConfig(...))`, then passes
only its train and validation loaders to the epoch functions in `training.py`.
The factory still creates test split metadata/index views and validates official
paths, as before. No test sample is passed to the model or used for any development
decision. Phase 3 does not download Food-101 automatically.

Default training settings:

| Setting | Value |
| --- | --- |
| Epochs | 5 |
| Batch size | 16 (CLI override; Phase 2 default remains 32) |
| Loss | CrossEntropyLoss |
| Optimizer | AdamW, all layers |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Workers | 0 |
| Seed | 42 |
| Input / validation fraction | Existing 224×224 / 0.2 |
| Device | auto (MPS → CUDA → CPU) |
| Pretrained | true |
| Best checkpoint | ml/artifacts/efficientnet_b0_best.pt |

The completed run produced the following train/validation losses and accuracies:

| Epoch | Train loss | Train accuracy | Validation loss | Validation accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0329 | 70.73% | 0.4178 | 87.47% |
| 2 | 0.4081 | 87.32% | 0.3315 | 89.87% |
| 3 | 0.2817 | 91.47% | 0.3185 | 90.87% |
| 4 | 0.1908 | 93.83% | 0.2781 | **92.00% (best)** |
| 5 | 0.1413 | 95.55% | 0.3019 | 91.53% |

Checkpoint metadata confirms epoch 4, 92.00% validation accuracy, MPS, seed 42,
ImageNet `IMAGENET1K_V1` provenance, and the existing class order. Training is
complete; the best epoch-4 checkpoint is used for final test evaluation and inference.

The training CLI can start a fresh run (do not run it to reproduce the completed
experiment unless a new training experiment is intended):

~~~bash
python ml/src/train.py --epochs 5 --batch-size 16 --lr 0.0001
python -m ml.src.train --help
~~~

The CLI also accepts `--root`, `--weight-decay`, `--num-workers`, `--seed`,
`--device`, `--checkpoint`, and `--no-pretrained`. Changing the seed deliberately
changes the development split through Phase 2's own logic; the default remains 42.
Use a different checkpoint path to retain multiple runs: each invocation starts
fresh and replaces that path whenever its own validation accuracy improves.

The model factory announces whether pretrained weights are cached or need a
download. Torchvision uses its normal torch hub cache, configurable via `TORCH_HOME`.
A network/cache failure stops the run with context; it never silently falls back to
random weights. `--no-pretrained` is an explicit random-initialization option.

Each training batch performs zero-grad, forward, cross-entropy loss, backward, and
AdamW step. Loss/accuracy are weighted by sample count, including partial final
batches. Validation runs after every epoch using `model.eval()` and
`torch.no_grad()`, without an optimizer or parameter/buffer updates. Nonfinite loss
stops the run. Epoch-level logs report train/validation loss and accuracy.

## Checkpoint saving and loading

Strictly improved validation accuracy saves a checkpoint; ties retain the earlier
epoch. `checkpoint.py` writes a temporary file beside the destination and replaces
the destination atomically. `ml/artifacts/` and checkpoint extensions are ignored.

Checkpoints contain a format version, model name/configuration, ImageNet weight
provenance, model and optimizer state, ordered classes and class-to-label mapping,
data/split/seed settings, descriptions from the actual Phase 2 transforms, training
configuration, best epoch and metrics, history through that epoch, and library/device
information. They represent the best model, not the final epoch of every run.

~~~python
from pathlib import Path
from ml.src.checkpoint import load_checkpoint
from ml.src.runtime import select_device

model, metadata = load_checkpoint(
    Path("ml/artifacts/efficientnet_b0_best.pt"), device=select_device(),
)
classes = metadata["classes"]  # logits index -> existing Phase 2 class name
~~~

Loading uses `torch.load(..., map_location="cpu", weights_only=True)`, verifies
format/model identity, class order/mapping, classifier size, preprocessing, and key
training metadata, then strictly loads model weights. Reconstruction uses
`pretrained=False` even when the saved provenance is ImageNet, so loading never
downloads weights. The returned model is on the requested device in eval mode.
Optimizer state is included, but exact interrupted-run resumption is not implemented:
RNG/loader position and subsequent training history are not restored.

## Phase 3 smoke verification

~~~bash
python ml/src/verify_training.py
~~~

This defaults to CPU, batch size 2, workers 0, seed 42, and random weights. It uses
one real Phase 2 training batch for a single backward/AdamW step and one real
validation batch for repeatability, mode, and state-preservation checks. It verifies
the ten-output classifier, finite `[2, 10]` logits/loss, backbone/head updates,
checkpoint state/logits/optimizer/class-mapping round-trip, offline loading, and
rejection of incompatible checkpoints. Temporary checkpoints are removed afterward.
The JSON report is ignored at `ml/data/processed/phase3_verification.json`.

Additional controlled checks cover device selection/overrides/unavailable-device
errors, seeded generators, unequal batch metric aggregation, pretrained enum/error
handling, and best-checkpoint selection including ties. These do not execute GPU
kernels or fetch pretrained weights. `--device mps` or `--device cuda` runs the real
model checks on that device when available; `--pretrained` explicitly allows a real
ImageNet download for verification. `--num-workers 2` exercises the existing spawn
loader path without changing it.

For Phase 2 regression checks, use its existing `verify_pipeline.py --all-batches`.
That verifies all selected images, including test images, solely for dataset and
transform integrity; it does not compute model performance. Phase 3 smoke checks
do not evaluate test samples.

## Final held-out test evaluation

`evaluate.py` loads and metadata-validates the saved checkpoint using the existing
loader. The evaluation root, seed, class order, image size, and validation fraction
come from that checkpoint's `DataConfig`; batch size and worker count can be changed
without changing split membership. It reuses `create_data_pipeline` and its existing
deterministic test transform. Before any image is decoded, it checks the official
source split, checkpoint class mapping/transform, exact selected official indices,
unique paths, 250 samples per class, and a sequential loader that drops no samples.
After the single test pass it again requires 2,500 total predictions and 250 per
class.

The report contains summed CrossEntropy loss divided by the total number of samples,
overall accuracy, sample count, per-class sample/correct counts and accuracy, and a
10×10 numeric confusion matrix. Matrix rows are true classes, columns are predicted
classes; row/column indices follow the saved class order. Results print to the
terminal and a JSON report is saved under ignored `ml/artifacts/` by default.
The report records the checkpoint SHA-256, epoch, and validation accuracy. The
evaluator checks that the checkpoint hash stays unchanged and does not create an
optimizer or save model state. Existing output files and checkpoint aliases are
refused to protect the model artifact.

Exact command for the reviewed final evaluation:

~~~bash
python ml/src/evaluate.py --checkpoint ml/artifacts/efficientnet_b0_best.pt --device mps --batch-size 16 --num-workers 0 --report ml/artifacts/phase3_final_test.json
~~~

The final evaluation completed on Apple MPS. Its report is
`ml/artifacts/phase3_final_test.json`, which records the checkpoint hash, epoch 4,
validation accuracy, metrics, and confusion matrix. Results were:

| Measure | Final held-out result |
| --- | ---: |
| CrossEntropy loss | 0.176723 |
| Accuracy | 94.48% |
| Samples | 2,500 |

| Food class | Correct / total | Accuracy |
| --- | ---: | ---: |
| pizza | 240 / 250 | 96.00% |
| hamburger | 231 / 250 | 92.40% |
| sushi | 231 / 250 | 92.40% |
| steak | 225 / 250 | 90.00% |
| ice_cream | 235 / 250 | 94.00% |
| donuts | 238 / 250 | 95.20% |
| omelette | 231 / 250 | 92.40% |
| french_fries | 246 / 250 | 98.40% |
| chicken_wings | 242 / 250 | 96.80% |
| spaghetti_bolognese | 243 / 250 | 97.20% |

These are the final held-out results. Do not use them for further model or
hyperparameter tuning. The command above describes the completed run; the report
path must be new if evaluation is run again.

Run synthetic-only verification (it loads the saved checkpoint for synthetic
tensors, but never loads Food-101 or consumes official images):

~~~bash
python ml/src/verify_evaluation.py
~~~

Six checks cover metric aggregation and confusion orientation, malformed inputs,
class/mapping/transform/split/sampler validation, a full 2,500-sample synthetic
loader pass, output-path protections, and real-checkpoint immutability with synthetic
images. It asserts eval/no-grad, unchanged parameters/buffers/gradients, and unchanged
checkpoint bytes. It runs no training steps and blocks Food-101 loading.

## Single-image prediction

`inference.py` provides `FoodImagePredictor.from_checkpoint(...)` and reusable
`.predict(image_path, top_k=3)`. The checkpoint loader validates the model, ordered
classes, and preprocessing metadata and restores weights without ImageNet downloads.
The predictor selects `auto`, `mps`, `cuda`, or `cpu` through `runtime.py`; it derives
both class-name/index directions from the checkpoint's `class_to_idx`.

Input images are checked for file existence and successful decoding, converted to
RGB, and transformed with the existing Phase 2 deterministic evaluation transform.
The checkpoint's input size must be 224×224. `build_transforms` supplies resize,
center crop, tensor conversion, and ImageNet normalization; no preprocessing
constants or steps are duplicated in the inference implementation.

For each call the model enters eval mode and a no-grad context. A single-item batch
produces ten logits, then softmax probabilities. The result includes the mapped
predicted name, class index, confidence in `[0, 1]`, and the requested descending
top-k list. The operation creates no optimizer, updates no model state, and never
writes to the checkpoint.

CLI example:

~~~bash
python ml/src/predict.py path/to/image.jpg --checkpoint ml/artifacts/efficientnet_b0_best.pt --device mps --top-k 3
~~~

The CLI prints the predicted class, confidence as a percentage, and top predictions.
PIL-readable formats such as JPEG, PNG, and WebP are accepted by the installed PIL
decoder. Missing, corrupt, and unreadable paths fail with a clear error. The reusable
API returns the same prediction data as a dictionary.

Run inference verification:

~~~bash
python ml/src/verify_inference.py
~~~

The verifier loads the saved best checkpoint and predicts on temporary synthetic
RGBA and grayscale PNGs. It checks RGB conversion, the `[1, 3, 224, 224]` forward
input, finite ten-class probabilities summing to one, checkpoint-derived names and
indices, descending top-k output, CLI formatting, eval/no-grad behavior, unchanged
weights/buffers/gradients, and unchanged checkpoint bytes. It uses no dataset images,
does no metric evaluation, and takes no optimizer steps.

The classifier recognizes only the configured ten classes and has no explicit
unknown or out-of-distribution class. Phase 3 is COMPLETE. Phase 4, the Nutrition
and Calorie Data Layer, is next and is not implemented yet.
