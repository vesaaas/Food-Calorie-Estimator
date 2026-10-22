# Architecture

## Current and planned flow

~~~text
Food image
    ↓
Image preprocessing
    ↓
13-class MobileNetV3 Small     ← current TFLite deployment model
    ↓
Predicted canonical class + informational confidence
    ↓
Exact nutrition lookup         ← Flutter catalog mirror, Python catalog canonical
    ↓
Default or user-edited portion ← Phase 7.6 mobile grams editor
    ↓
Calculated calories/macros     ← Phase 4 Python / Phase 7.6 Dart calculation
    ├── Mobile Scan display    ← current through Phase 7.6
    └── Persisted meal + goal  ← Phase 6 Python services (not mobile-integrated)
         ↓
       Daily and weekly summaries ← Phase 6 Python domain service
~~~

## Responsibilities

### Historical PyTorch EfficientNet-B0 baseline (Phases 3–5)

Phase 5 kept the Phase 3 ten-class checkpoint intact. It created a separate
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

### MobileNetV3 TFLite deployment model (Phase 7 groundwork)

The assignment-specified mobile model is implemented in parallel with the prior
baseline. `ml/src/mobilenet_v3.py` builds Keras MobileNetV3 Small with ImageNet
weights, a fresh 13-way softmax head, and the exact Phase 5 canonical class order.
The whole backbone is fine-tuned. Its smaller model size is the chosen mobile
tradeoff; the TFLite deployment result is reported separately from EfficientNet-B0.

`tf_phase5_data.py` adapts the existing verified Phase 5 dataset objects. It does
not change split membership or duplicate policy. It reverses Phase 2 ImageNet
normalization so Keras receives raw RGB float32 values in [0,255], then Keras' built-in
MobileNetV3 layer maps them to [-1,1]. Input is `[1,224,224,3]` NHWC after resizing
the shortest edge to 256 and center-cropping to 224. Output is `[1,13]` softmax
probabilities mapped by `ml/mobile/mobilenet_v3_labels.txt`.

Training and validation run through `train_mobilenet_v3.py`; only validation
accuracy selects the checkpoint. `evaluate_mobilenet_v3.py` evaluates the independent
Phase 5 mixed test set. `export_mobilenet_v3_tflite.py` applies TFLite dynamic-range
weight optimization and checks the serialized interpreter tensor contract.
`mobilenet_v3_pipeline.py` provides reusable image prediction and exact nutrition
lookup; `verify_mobilenet_v3.py` compares Keras and TFLite on real per-class test
images and confirms the Phase 3/5 artifacts remain unchanged.

The model contract is `ml/mobile/mobilenet_v3_tflite_contract_v2.json`. It specifies
preprocessing, tensor shape/dtype, ordered labels, and the meaning/limits of
confidence. Confidence is not calibrated, does not change nutrients, and does not
estimate portion size. The Flutter mobile flow now performs inference and nutrition
lookup; see the mobile responsibility below.

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

### Phase 6 application/domain layer

`app/meal_log.py` turns an exact food ID and user-selected portion into a validated
meal record by calling the existing nutrition calculator. It stores selected grams
and computed nutrients, plus timestamp and optional informational classifier
confidence. The confidence field is metadata only; the domain does not use it in
nutrition calculations or portion selection.

`app/repositories.py` defines the diary persistence protocol. `app/storage.py`
implements that contract with validated, atomic JSON writes; meal and goal services
depend on the protocol, not file operations. Missing storage means an empty diary.
The JSON file under `app/data/` is generated runtime data and Git-ignored, so a
future client can supply a SQLite or platform repository without rewriting domain
services.

`app/goals.py` validates and stores a positive finite daily calorie goal.
`app/dashboard.py` provides pure daily and weekly aggregation plus repository-backed
queries. Goal progress is a fraction and can exceed 1 when over goal; remaining
calories become negative. With no goal, remaining/progress are `None`. A week is the
inclusive seven calendar days ending on its requested date, including empty days as
zero totals. UTC is the default timezone for assigning timestamps to days, and the
services accept explicit timezones and clocks for deterministic use.

### Flutter mobile flow (Phase 7.4–7.6)

`mobile/lib/services/image_acquisition_service.dart` provides camera/gallery input;
`food_classification_service.dart` returns the canonical class ID and confidence.
`nutrition_service.dart` loads the bundled `mobile/assets/nutrition/foods.json`
mirror, looks up the exact class ID, and applies the catalog default or an edited
gram amount. The Python catalog remains canonical, and
`nutrition/src/verify_nutrition.py` checks the full mobile copy for parity. Confidence
is display metadata only; no portion weight is inferred from the image. The mobile
flow displays nutrition but does not persist meals or connect goals and dashboards.

Phase 6 is reusable Python/domain/application logic, not a terminal business flow.
`app/verify_phase6.py` performs offline service and persistence verification. Mobile
meal persistence, goals, dashboards, physical Android testing, and release work
remain outstanding.

## Phase 2 boundary

Phase 2 owns dataset download/reuse, metadata validation, class filtering, split
construction, transforms, DataLoaders, exploration, and verification. It does not
choose a model architecture, train weights, estimate calories, or implement a user
interface.
