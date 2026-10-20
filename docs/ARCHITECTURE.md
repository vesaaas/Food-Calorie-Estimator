# Architecture

## Current and planned flow

~~~text
Food image
    ↓
Image preprocessing
    ↓
13-class EfficientNet-B0       ← Phase 5 (separate from frozen Phase 3)
    ↓
Predicted canonical class + informational confidence
    ↓
Exact nutrition lookup + default serving
    ↓
Calories and macronutrients    ← Phases 4–5
    ↓
Editable portion and app flow  ← later phase
~~~

## Responsibilities

### Phase 5 classifier and nutrition integration

Phase 5 keeps the Phase 3 ten-class checkpoint intact. It creates a separate
13-class EfficientNet-B0 checkpoint, initializes only its compatible `features.*`
parameters from the Phase 3 checkpoint, and leaves the new 13-output head freshly
initialized. Training combines the existing Food-101 development split with the
three selected TurkishFoods-25 labels. The source `train`, `eval`, and `test` splits
remain separate; `yaprak_sarma` is explicitly mapped to canonical `sarma`.

The image-to-nutrition pipeline reuses the existing RGB transforms, predictor,
checkpoint validation, and Phase 4 exact catalog lookup. It returns classifier
confidence independently from the nutrition result. Confidence never scales
calories/macros and image pixels do not yield portion grams; the catalog default is
used until a user-selected amount is available. This classifier has no unknown class.

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

Phase 4 implements this handoff as exact class ID → USDA FDC catalog entry → default
or user-selected portion → calories and macros. `nutrition/src/catalog.py` validates
the local versioned JSON and source metadata; `calculator.py` performs per-100 g
scaling; `estimate.py` exposes a small CLI. This layer has no PyTorch dependency.
The class ID is the classifier's exact label string. It does not infer the class from
confidence, class array position, or a fuzzy name match.

Per-100 g energy, protein, carbohydrate, and total fat values form the canonical
basis. Each record supplies a default serving weight with its source or documented
reference basis. USDA records include source portion IDs where available; the
TürKomp börek entry uses a 100 g reference amount. The
calculator accepts the default, explicit grams, or a serving multiplier. Classifier
confidence never scales nutrition values. Image classification does not estimate
portion mass; it must be selected or entered by a user. The selected USDA entry is a
representative reference for a broad dish class, not an exact match to every recipe
or photographed meal. See `docs/NUTRITION.md` for record-level provenance and limits.

### Application, API, and mobile layers

Later phases may expose inference through an API and connect it to the mobile
application. Those components should consume a model prediction and nutrition lookup result
through stable interfaces. The mobile/ directory remains a placeholder;
nutrition/ contains the catalog and calculator, and Phase 5 provides a reusable
image-to-nutrition pipeline and CLI.

## Phase 2 boundary

Phase 2 owns dataset download/reuse, metadata validation, class filtering, split
construction, transforms, DataLoaders, exploration, and verification. It does not
choose a model architecture, train weights, estimate calories, or implement a user
interface.
