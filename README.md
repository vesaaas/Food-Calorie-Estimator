# Food Calorie Estimator

An AI/ML project that recognizes supported foods in meal images and maps classifier
labels to representative nutrition estimates with adjustable portion calculations.

## Current status

- Phase 1 — Project Setup: complete
- Phase 2 — Dataset & Preprocessing: complete and verified
- Phase 3 — Complete: trained classifier, final held-out evaluation, and single-image inference
- Phase 4 — COMPLETE: Nutrition and Calorie Data Layer
- Phase 5 — COMPLETE: regional food classes and image-to-nutrition integration
- Phase 6 — COMPLETE: editable portions, meal log, calorie goal, and summaries
- Phase 7 — IN PROGRESS: MobileNetV3/TFLite model preparation; mobile UI remains planned

Food-101 supports classification only. The nutrition catalog is a separate sourced
layer. Classification does not measure food weight; estimates use a configured
default or a portion selected by the user.

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
explicit unknown or out-of-distribution class. The final held-out result is a frozen
classification result and must not be interpreted as nutrition accuracy.

## Phase 4 nutrition examples

~~~bash
python nutrition/src/estimate.py pizza
python nutrition/src/estimate.py pizza --servings 1.5
python nutrition/src/estimate.py pizza --grams 180
python nutrition/src/verify_nutrition.py
~~~

The ten exact class mappings, USDA provenance, default servings, formula, and limits
are documented in [the nutrition layer guide](docs/NUTRITION.md).

## Phase 5 local food support and image-to-nutrition baseline

Phase 5 adds three Turkish regional classes to the existing ten-class label order:
`borek`, `kebap`, and `sarma`. Source label `yaprak_sarma` maps explicitly to
`sarma`. The pinned TurkishFoods-25 repository and preparation commands are
documented in [the dataset guide](docs/DATASET.md). Once requirements are installed:

~~~bash
python ml/src/download_phase5_dataset.py
python -m ml.src.verify_phase5
python nutrition/src/verify_nutrition.py
~~~

The separate training and evaluation commands are `python ml/src/train_phase5.py`
and `python ml/src/evaluate_phase5.py`. The mixed Phase 5 test result was 95.66%
classification accuracy on 2,719 samples. The Phase 3 checkpoint and its frozen
94.48% ten-class classification result remain historical and untouched.
To run an image through the Phase 5 checkpoint and estimate the configured default
serving, run:

~~~bash
python -m ml.src.phase5_pipeline path/to/meal.jpg
~~~

Confidence is informational and does not scale nutrition or estimate portion size.
The classifier has no unknown-food class; unrelated images may still be assigned to
one of its 13 labels.

The Phase 5 PyTorch EfficientNet-B0 is retained as the previous experiment and
historical 95.66% mixed-test baseline. The assignment's mobile deployment model is
now a separately trained Keras MobileNetV3 Small exported to TensorFlow Lite. On the
same 2,719-image held-out split, the selected Keras checkpoint scored 89.41%
accuracy (loss 0.33134); the optimized TFLite model scored 88.12%. These are new
MobileNetV3 results, not revisions to the EfficientNet-B0 results. See
[the mobile model contract](ml/mobile/mobilenet_v3_tflite_contract_v2.json) and
[the ML pipeline report](docs/ML_PIPELINE.md).

Run the offline TFLite/model regression check and classify a photo with the exported
model:

~~~bash
python -m ml.src.verify_mobilenet_v3
python -m ml.src.mobilenet_v3_pipeline path/to/meal.jpg
~~~

The TFLite input is RGB float32, NHWC `[1,224,224,3]`, pixel range 0–255. Resize
the shortest edge to 256 with bilinear interpolation, then center crop 224×224. The
model performs its own MobileNetV3 rescaling to [-1,1]. Outputs are 13 softmax
scores in the exact label-file order. Confidence remains informational. This is
model/deployment groundwork only: Phase 7 mobile UI, camera/gallery integration,
physical Android testing, and release have not been implemented.

## Phase 6 domain services

Phase 6 adds reusable Python application logic for editable portions, meal entries,
a personal calorie goal, and daily/weekly summaries. It reuses the nutrition
calculator; the JSON repository is only a replaceable persistence adapter. The
default diary file is `app/data/diary.json` and is ignored by Git. Dashboard dates
use UTC by default; a caller can supply another timezone. A week is the inclusive
seven calendar days ending on the requested date.

~~~bash
python -m app.verify_phase6
~~~

Python services can create entries with default servings, a serving multiplier, or
explicit grams. Classifier confidence may be stored as prediction metadata, but it
does not affect nutrition or portion weight. Phase 6 is domain logic only. Phase 7
still needs the mobile UI and camera/gallery integration, client packaging, physical
Android testing, and APK/demo/release work; this repository does not yet provide a
mobile application.

## Documentation

- [Roadmap](docs/ROADMAP.md)
- [Current state checkpoint](docs/CURRENT_STATE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Dataset layer](docs/DATASET.md)
- [ML pipeline and training](docs/ML_PIPELINE.md)
- [Nutrition data layer](docs/NUTRITION.md)
- Phase 6 services are described in [the architecture](docs/ARCHITECTURE.md) and
  [current state](docs/CURRENT_STATE.md).

The detailed documents record the verified sample counts, class mapping, transforms,
DataLoader configuration, limitations, implementation decisions, and next phase.
