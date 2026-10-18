# Food Calorie Estimator

An AI/ML project for recognizing food in meal images and, in later phases, mapping
predicted foods to estimated nutrition information.

## Current status

- Phase 1 — Project Setup: complete
- Phase 2 — Dataset & Preprocessing: complete and verified
- Phase 3 — Complete: trained classifier, final held-out evaluation, and single-image inference
- Phase 4 — Next: Nutrition and Calorie Data Layer

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

## Phase 3 commands

Run the short verification first (one real train batch, validation, checkpoint
round-trip; random weights by default, no network download):

~~~bash
python ml/src/verify_training.py
~~~

Five epoch training completed on Apple MPS. The best checkpoint is epoch 4 with
92.00% validation accuracy. Final held-out evaluation reached 94.48% (loss
0.176723) on all 2,500 test images. Single-image prediction loads that checkpoint
and reuses Phase 2 preprocessing:

~~~bash
python ml/src/predict.py path/to/image.jpg --checkpoint ml/artifacts/efficientnet_b0_best.pt --device mps --top-k 3
~~~

See [the ML pipeline](docs/ML_PIPELINE.md) for per-class test results and inference
verification. The classifier recognizes only the configured ten classes and has no
explicit unknown or out-of-distribution class. Phase 4 nutrition/calorie data is
the next work; it is not implemented yet.

## Documentation

- [Roadmap](docs/ROADMAP.md)
- [Current state checkpoint](docs/CURRENT_STATE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Dataset layer](docs/DATASET.md)
- [ML pipeline and training](docs/ML_PIPELINE.md)

The detailed documents record the verified sample counts, class mapping, transforms,
DataLoader configuration, limitations, implementation decisions, and next phase.
