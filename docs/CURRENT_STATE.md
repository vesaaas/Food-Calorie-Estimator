# Current State

## Project status

Phase 1 — Project Setup: complete.

Phase 2 — Dataset & Preprocessing: complete and verified against the real Food-101
dataset.

The next phase is Phase 3 — food classification model training and evaluation.
There is no model training, calorie prediction, mobile implementation, or nutrition
mapping in the current codebase.

## Environment and dependencies

- Python: 3.12.15 in .venv
- PyTorch: 2.14.1
- Torchvision: 0.29.1
- Pillow: 12.3.0
- NumPy: 2.5.4
- Pandas: 3.0.6
- Matplotlib: 3.11.2
- PyTorch MPS is available on the development Mac, but Phase 2 remains device-independent.
- requirements.txt is pinned and requires no additional Phase 2 dependencies.

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
│   └── src/
│       ├── download_dataset.py
│       ├── dataset.py
│       ├── preprocessing.py
│       ├── explore_dataset.py
│       └── verify_pipeline.py
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
- Data and labels remain on CPU until a future training phase chooses a device
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

## Important decisions and limitations

- Food-101 is used for classification only. It does not contain portion sizes,
  calories, protein, carbohydrates, fat, or other nutrition measurements.
- Subsets are index views over the original images; no image copies are created.
- Validation is derived only from official training data. The official test split
  remains untouched for later final evaluation.
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

Implement Phase 3 model training and evaluation using the existing pipeline. Keep
model artifacts ignored, preserve the class order recorded in DATASET.md, and use
the validation split for development decisions while reserving the official test
split for final evaluation.
