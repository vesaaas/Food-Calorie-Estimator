# Food Calorie Estimator

An AI/ML project for recognizing food in meal images and, in later phases, mapping
predicted foods to estimated nutrition information.

## Current status

- Phase 1 — Project Setup: complete
- Phase 2 — Dataset & Preprocessing: complete and verified
- Phase 3 — Food classification model training: next

Food-101 currently supports food classification only. It does not contain calorie,
portion-size, or nutrition measurements. Nutrition mapping is planned for a later
phase.

## Setup

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

## Phase 2 commands

~~~bash
python ml/src/download_dataset.py
python ml/src/explore_dataset.py
python ml/src/verify_pipeline.py --all-batches
python ml/src/explore_dataset.py --view validation --output ml/data/processed/sample_grid.png
~~~

The pipeline uses ten Food-101 classes, 224×224 ImageNet-normalized tensors,
deterministic stratified train/validation splits, the official isolated test split,
and CPU PyTorch DataLoaders. Raw data and generated outputs are ignored by Git.

## Documentation

- [Roadmap](docs/ROADMAP.md)
- [Current state checkpoint](docs/CURRENT_STATE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Dataset layer](docs/DATASET.md)
- [Phase 2 ML pipeline](docs/ML_PIPELINE.md)

The detailed documents record the verified sample counts, class mapping, transforms,
DataLoader configuration, limitations, implementation decisions, and next phase.
