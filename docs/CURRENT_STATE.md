# Current State

## Project status

Phase 1 — Project Setup: complete.

Phase 2 — Dataset & Preprocessing: complete and verified against the real Food-101
dataset.

Phase 3 — COMPLETE. EfficientNet-B0 training, validation-based checkpoint
selection, one-time final held-out evaluation, and single-image inference are
implemented and verified. The official test results are final and must not be used
for further model selection. Phase 4 — COMPLETE. Nutrition lookup and portion
calculation are implemented for the original ten classes and extended with three
regional records in Phase 5. Phase 5 is COMPLETE. Phase 6 is COMPLETE and provides
reusable Python domain logic for editable portions, meal logging, personal calorie
goals, and daily/weekly summaries. Phase 7 mobile UI and device/release work have
not started; the repository does not yet contain a mobile application or infer
portion weight from images.

## Environment and dependencies

- Python: 3.12.15 in .venv
- PyTorch: 2.14.1
- Torchvision: 0.29.1
- Pillow: 12.3.0
- NumPy: 2.5.4
- Pandas: 3.0.6
- Matplotlib: 3.11.2
- The development Mac supports MPS; availability depends on the execution environment.
  The Phase 3 sandbox smoke run used CPU; an approved run outside the sandbox
  automatically selected MPS and passed the same one-batch checks. CUDA is unavailable.
- Phase 5 pins Hugging Face `datasets==5.0.1` for the reproducible TurkishFoods-25
  Dataset API integration.

## Repository structure

~~~text
food-calorie-estimator/
├── app/                       # Phase 6 domain/application services
│   ├── models.py
│   ├── repositories.py         # Persistence protocol
│   ├── storage.py              # Atomic JSON repository adapter
│   ├── meal_log.py
│   ├── goals.py
│   ├── dashboard.py
│   └── verify_phase6.py
├── docs/
│   ├── ROADMAP.md
│   ├── CURRENT_STATE.md
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   ├── ML_PIPELINE.md
│   └── NUTRITION.md
├── ml/
│   ├── data/
│   │   ├── raw/                 # Git-ignored Food-101 archive and extraction
│   │   └── processed/           # Git-ignored reports and preview images
│   ├── notebooks/
│   ├── artifacts/              # Git-ignored generated model checkpoints
│   └── src/
│       ├── download_dataset.py
│       ├── download_phase5_dataset.py
│       ├── phase5_data.py
│       ├── train_phase5.py
│       ├── evaluate_phase5.py
│       ├── phase5_pipeline.py
│       ├── verify_phase5.py
│       ├── dataset.py
│       ├── preprocessing.py
│       ├── explore_dataset.py
│       ├── verify_pipeline.py
│       ├── model.py
│       ├── runtime.py
│       ├── training.py
│       ├── checkpoint.py
│       ├── train.py
│       ├── verify_training.py
│       ├── evaluate.py
│       ├── verify_evaluation.py
│       ├── inference.py
│       ├── predict.py
│       └── verify_inference.py
├── mobile/
├── nutrition/
│   ├── data/foods.json          # Versioned thirteen-class nutrition catalog
│   ├── data/usda_source_extracts.json  # Selected USDA rows and archive hashes
│   ├── data/phase5_source_extracts.json  # Regional nutrition source extracts
│   └── src/                     # Catalog validation, portion calculator, CLI, verification
├── README.md
├── requirements.txt
└── .gitignore
~~~

## Dataset checkpoint

The dataset is Food-101 loaded through torchvision.datasets.Food101. The official
dataset contains 101 classes, 75,750 training images, and 25,250 test images. Each
class has 750 official training images and 250 official test images.

The development subset uses these labels in the application label order 0–9:

~~~text
pizza, hamburger, sushi, steak, ice_cream, donuts, omelette,
french_fries, chicken_wings, spaghetti_bolognese
~~~

The verified subset contains:

| Split | Total | Per selected class |
| --- | ---: | ---: |
| Train | 6,000 | 600 |
| Validation | 1,500 | 150 |
| Official test | 2,500 | 250 |

The raw dataset is stored under ml/data/raw/. Processed reports and previews are
stored under ml/data/processed/. Both locations are excluded from Git, along with
model checkpoint extensions.

## Preprocessing checkpoint

- Image size: 224 × 224
- Training: RandomResizedCrop with scale 0.8–1.0 and ratio 0.9–1.1, horizontal flip p=0.5, tensor conversion, normalization.
- Evaluation: resize shorter edge to 256, center crop 224, tensor conversion, normalization.
- ImageNet mean: 0.485, 0.456, 0.406
- ImageNet standard deviation: 0.229, 0.224, 0.225
- All images are converted to RGB.

## DataLoader checkpoint

- Default batch size: 32
- Default workers: 0 for simple macOS compatibility
- Training loader shuffles; validation and test loaders do not
- drop_last=False, pin_memory=False
- Positive worker counts use the macOS spawn context
- Data and labels remain on CPU until the training loop moves a batch to its selected device
- Default random seed: 42

## Verification checkpoint

The following checks passed:

- Official archive MD5: 85eeb15f3717b99a5da872d97d918f87
- Metadata, class lists, image paths, file existence, and official split isolation
- Exact selected class mapping and per-class stratification
- No train/validation/test path overlap
- No identical image bytes across splits
- Deterministic split reproduction with seed 42
- Official test membership unchanged when the seed changes
- Deterministic evaluation transforms
- All 10,000 selected images decoded and transformed
- Real batches with shape [32, 3, 224, 224] and labels [32]
- DataLoader verification with num_workers=0 and num_workers=2
- Python compilation, imports, pip check, and Git whitespace checks

Reports are generated in ignored ml/data/processed/verification*.json files.

## Phase 3 foundation

- Architecture: torchvision EfficientNet-B0, explicit ImageNet `IMAGENET1K_V1`
  initialization, replacement 10-output classifier, dropout 0.2; fine-tune all layers.
- Training defaults: 5 epochs, batch size 16, AdamW learning rate 0.0001 and weight
  decay 0.0001, CrossEntropyLoss, workers 0, seed 42, automatic MPS → CUDA → CPU.
- The training CLI reuses DataConfig and create_data_pipeline. Class order, split
  logic, 224×224 inputs, and Phase 2 preprocessing are unchanged.
- Validation runs after every epoch in eval/no_grad mode. Best validation accuracy
  selects an atomic checkpoint; ties retain the earlier epoch.
- Checkpoints include model/optimizer state, ordered classes and mapping,
  model/pretrained configuration, Phase 2 preprocessing/data settings, epoch,
  training settings, metrics/history up to the best epoch, device, and library versions.
- Loading uses CPU mapping, weights-only deserialization, strict model state loading,
  class/preprocessing checks, and no ImageNet download. It returns an eval-mode model.
- Real CPU smoke verification passed with random weights: `[2, 10]` logits, finite
  loss, one backward/AdamW step, unchanged validation parameters/buffers/gradients,
  and checkpoint round-trip. Controlled checks cover device branches, download errors,
  validation-based selection, unequal batch weighting, and incompatible metadata.
- The same smoke checks passed on MPS and with two CPU DataLoader workers. The
  existing Phase 2 regression also passed on all 10,000 selected images (zero
  workers), and its two-worker check passed. The sandbox initially blocked
  `torch_shm_manager`; two-worker checks passed when rerun with approval outside it.
- The full five-epoch run completed on MPS with ImageNet pretrained weights,
  AdamW (learning rate/weight decay 0.0001), batch size 16, seed 42. Epochs 1–5
  train losses/accuracies were 1.0329/70.73%, 0.4081/87.32%, 0.2817/91.47%,
  0.1908/93.83%, and 0.1413/95.55%; validation losses/accuracies were
  0.4178/87.47%, 0.3315/89.87%, 0.3185/90.87%, 0.2781/92.00%, and
  0.3019/91.53%. Epoch 4 is best.
- The saved production checkpoint confirms epoch 4 and 92.00% validation accuracy.
  Training is complete; CUDA execution was not used. Earlier random-weight smoke
  metrics are implementation checks, not model-quality estimates.

## Phase 3 final evaluation status

- `ml/src/evaluate.py` is the final evaluation entry point. It uses the checkpoint's
  saved `DataConfig` with the existing Phase 2 pipeline and validates the official
  source split, class mapping, deterministic transforms, all selected test indices,
  and sequential no-drop sampling before reading images.
- Evaluation uses eval/no_grad and reports cross-entropy loss, accuracy, sample
  count, per-class accuracy, and a numeric confusion matrix. It checks the checkpoint
  SHA-256 before and after, and creates a new ignored JSON report without overwriting
  existing files.
- `ml/src/verify_evaluation.py` passed six synthetic-only checks. These used no
  Food-101 image and ran no training step. It loaded the production checkpoint only
  for synthetic 224×224 inputs; checkpoint bytes, weights, buffers, and gradients
  stayed unchanged.
- Final official test result: loss 0.176723, accuracy 94.48%, 2,500 samples on MPS.
  Epoch 4 was the evaluated checkpoint; its validation accuracy was 92.00%.
- Per-class test accuracy and the confusion matrix are recorded in
  `docs/ML_PIPELINE.md` and the ignored `ml/artifacts/phase3_final_test.json`.
- These held-out results are final and must not be used for further model or
  hyperparameter tuning.

## Phase 3 single-image inference

- `inference.py` loads through the metadata-validating checkpoint loader, selects a
  device through `runtime.py`, uses the saved class mapping, and builds preprocessing
  with `preprocessing.py` at the trained 224×224 size.
- `predict.py IMAGE_PATH --checkpoint ... --device mps --top-k 3` prints the mapped
  class, confidence, and top predictions. It supports PIL-readable image formats and
  converts inputs to RGB.
- `verify_inference.py` passed on temporary RGBA and grayscale PNGs. It confirmed
  `[1, 3, 224, 224]` model input, finite normalized softmax probabilities, mapping,
  descending top-k, eval/no_grad, unchanged state/gradients, and unchanged checkpoint
  bytes. No dataset images or test metrics were used.

The classifier recognizes only the ten configured Food-101 classes. It has no
explicit unknown or out-of-distribution class, so unrelated or ambiguous images
still receive one of those ten labels.

## Important decisions and limitations

- Food-101 is used for classification only. It does not contain portion sizes,
  calories, protein, carbohydrates, fat, or other nutrition measurements.
- Subsets are index views over the original images; no image copies are created.
- Validation is derived only from official training data. The official test split
  was used once for final evaluation after model selection; those results are frozen
  and must not guide further model or hyperparameter choices.
- The split code hashes selected training image bytes and keeps duplicate-content
  groups together. This fixed a real duplicate hamburger pair found during verification.
- The dataset adapter uses torchvision's private _image_files and _labels fields
  because Food101 does not expose public sample properties in the pinned version.
  Re-run verification after changing torchvision.
- Food-101 training labels are known to contain some noise, and visual near-duplicates
  are not automatically detected.
- A food class does not identify portion size or preparation details. Phase 4 uses a
  sourced representative record and serving; future UI work must let the user adjust
  portions and must not claim image-derived weight.

## Phase 4 nutrition layer — COMPLETE

- `nutrition/data/foods.json` originally contained the ten Phase 3 mappings and now
  contains those unchanged mappings plus three Phase 5 regional entries. Each row
  records source type/release,
  reference description, retrieval date, four per-100 g nutrients, and a source-backed
  default serving/portion ID.
- `nutrition/data/usda_source_extracts.json` preserves the selected USDA nutrient and
  portion rows, archive URLs, and SHA-256 hashes for local audit.
- `nutrition/src/catalog.py` loads and validates JSON, provenance, exact lookup, and
  optional exact class coverage. `calculator.py` scales from per-100 g values by a
  default serving, explicit grams, or serving multiplier. `estimate.py` provides the
  CLI. The core does not import PyTorch or depend on classifier confidence.
- `nutrition/src/verify_nutrition.py` runs offline checks for all thirteen
  classifier labels, USDA/TürKomp evidence, errors, calculations, non-mutation, CLI behavior,
  and no-PyTorch import.
- Food weight is not measured from an image. Broad class mappings are representative
  estimates with preparation and recipe limitations.

## Phase 5 — COMPLETE

- The source is TurkishFoods-25 pinned to revision
  `3e5026a17d14df81a25f0b6cdb16a58301fbb83c`, with named classes and official
  train/eval/test splits. Only `borek`, `kebap`, and `yaprak_sarma` are retained;
  the latter maps explicitly to canonical `sarma`.
- The final label order retains the Phase 3 ten labels and appends `borek`, `kebap`,
  `sarma`. The separate model initialization copies Phase 3 EfficientNet feature
  weights only and creates a fresh 13-output head. It trained five epochs on MPS;
  epoch 5 was selected at 92.24% validation accuracy.
- Exported train counts are 609 börek, 682 kebap, 393 sarma; validation counts are
  70, 80, 50; test counts are 78, 82, 59. Thirty-four repeated-content rows were
  excluded: 26 within-split duplicates and eight potential cross-split leaks. Test
  content took priority over validation, which took priority over training.
- The independent mixed test result is 0.17226 loss and 95.66% accuracy on 2,719
  samples. Per-class results are recorded in `docs/ML_PIPELINE.md` and the ignored
  `ml/artifacts/phase5_mixed_test.json` report.
- The real single-image pipeline predicted a source sarma test image and produced
  its catalog default serving and macros. Offline verifiers passed; Phase 3 model
  and evaluation artifact hashes remain unchanged.

## Phase 6 — COMPLETE

- Meal creation supports catalog defaults, serving multipliers, and explicit grams
  by calling `nutrition.src.calculator.estimate`; nutrition math is not duplicated.
- Immutable `MealEntry` records preserve exact food ID, selected grams, calculated
  nutrients, timezone-aware timestamp, portion mode, and optional classifier
  confidence metadata. Confidence does not affect grams or nutrients.
- `DiaryRepository` separates domain services from storage. The JSON adapter handles
  missing initial files, validates loaded records, and writes atomically. Runtime
  diary files under `app/data/` are Git-ignored.
- Daily summaries include calories/macros, meal count, goal, remaining calories,
  progress fraction, and over-goal state. Without a goal, goal-related values are
  `None`; when over goal, remaining calories are negative and progress may exceed 1.
- Weekly summaries use seven inclusive calendar dates ending on the requested date,
  include zero-total empty days, and default to UTC day boundaries. Services accept
  an explicit timezone and injectable clock for deterministic callers/tests.
- `python -m app.verify_phase6` is the offline domain and persistence verifier.

## Phase 7 — PLANNED

Phase 7 will add mobile UI/integration, camera/gallery input, deployment strategy,
physical Android testing, and APK/demo/release validation. It has not started.
Preserve the Phase 3 checkpoint and keep its frozen held-out result out of future
model selection.
