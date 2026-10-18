# Current State

## Project status

Phase 1 — Project Setup: complete.

Phase 2 — Dataset & Preprocessing: complete and verified against the real Food-101
dataset.

Phase 3 — COMPLETE. EfficientNet-B0 training, validation-based checkpoint
selection, one-time final held-out evaluation, and single-image inference are
implemented and verified. The official test results are final and must not be used
for further model selection. Phase 4, Nutrition and Calorie Data Layer, is next.
There is no calorie prediction, mobile implementation, or nutrition mapping yet.

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
- requirements.txt is pinned; the Phase 3 foundation adds no dependencies.

## Repository structure

~~~text
food-calorie-estimator/
├── docs/
│   ├── ROADMAP.md
│   ├── CURRENT_STATE.md
│   ├── ARCHITECTURE.md
│   ├── DATASET.md
│   └── ML_PIPELINE.md
├── ml/
│   ├── data/
│   │   ├── raw/                 # Git-ignored Food-101 archive and extraction
│   │   └── processed/           # Git-ignored reports and preview images
│   ├── notebooks/
│   ├── artifacts/              # Git-ignored generated model checkpoints
│   └── src/
│       ├── download_dataset.py
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
- A food class does not identify portion size or preparation details, so a future
  nutrition estimate will require a separate nutrition data and product-design decision.

## Next work

Begin Phase 4: Nutrition and Calorie Data Layer. Preserve the selected checkpoint,
keep the final held-out results out of model-selection decisions, keep generated
artifacts ignored, and preserve the existing class order.
