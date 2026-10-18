# Architecture

## Current and planned flow

~~~text
Food image
    ↓
Image preprocessing
    ↓
Food classification model       ← Phase 3
    ↓
Predicted Food-101 class
    ↓
Nutrition/calorie data layer    ← Phase 4
    ↓
Application/API/mobile layer    ← Phases 5–6
~~~

## Responsibilities

### Image input

A future application or API will provide a meal image. Phase 2 only handles images
already represented by Food-101 and does not implement upload or camera behavior.

### Image preprocessing

ml/src/preprocessing.py converts images to RGB tensors, applies training or
evaluation transforms, resizes them to model-ready dimensions, and applies ImageNet
normalization. ml/src/dataset.py attaches the appropriate transform to each split.

### Food classification model

The Phase 3 foundation consumes the DataLoaders produced by ml/src/dataset.py and
predicts one of the ten development classes in their existing order. Torchvision
EfficientNet-B0 uses explicit ImageNet `IMAGENET1K_V1` weights and a replacement
10-output linear classifier. All layers are trainable; Phase 2 transforms are reused.

- `model.py`: model configuration, pretrained initialization, classifier replacement.
- `runtime.py`: MPS/CUDA/CPU selection and shared seeded initialization.
- `training.py`: sample-weighted training/validation metrics; validation uses eval/no_grad.
- `checkpoint.py`: atomic saves and strict, metadata-validated restoration without downloads.
- `train.py`: CLI and AdamW/CrossEntropyLoss orchestration using train/validation only.
- `verify_training.py`: short real-data checks plus isolated device/selection/error checks.
- `evaluate.py`: explicit final official-test evaluation with overall/per-class metrics and confusion matrix.
- `verify_evaluation.py`: synthetic-only metric, loader, checkpoint-immutability, and CLI-path checks.
- `inference.py` / `predict.py`: reusable and command-line single-image prediction using checkpoint classes and Phase 2 transforms.
- `verify_inference.py`: synthetic PIL image checks for preprocessing, probabilities, mapping, and read-only inference.

Checkpoint selection uses validation accuracy, retaining the earlier epoch on ties.
Training uses only the train and validation loaders. The explicit evaluation entry
point passed the official test loader once and produced 94.48% accuracy, 0.176723
loss, and 2,500 samples. The completed test result is frozen and must not guide further
model selection. The shared Phase 2 factory still constructs all three split views
and checks official metadata/path integrity. Synthetic inference verification does
not load official Food-101 data.

Single-image inference loads that same best checkpoint, applies the saved
224×224 Phase 2 evaluation transform, and returns mapped softmax predictions. The
implementation does not train or write the checkpoint. Generated checkpoints and
evaluation JSON live in ignored `ml/artifacts/`. Phase 3 is complete. This is a
closed-set classifier for the configured ten classes; it has no explicit unknown or
out-of-distribution class.

### Nutrition/calorie data layer

Phase 4 will map a predicted food class to nutrition information using a separate
data source and explicit assumptions about serving size. Food-101 provides food
classification labels only; it does not directly provide calories, portion weights,
macronutrients, or nutrition values.

### Application, API, and mobile layers

Later phases may expose inference through an API and connect it to the mobile
application. Those components should consume a model prediction and nutrition
lookup result through stable interfaces. They are currently placeholders represented
by the existing mobile/ and nutrition/ directories.

## Phase 2 boundary

Phase 2 owns dataset download/reuse, metadata validation, class filtering, split
construction, transforms, DataLoaders, exploration, and verification. It does not
choose a model architecture, train weights, estimate calories, or implement a user
interface.
