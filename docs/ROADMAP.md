# Food Calorie Estimator Roadmap

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| Phase 1 | Project setup and Python environment | COMPLETE |
| Phase 2 | Food-101 dataset and preprocessing | COMPLETE |
| Phase 3 | EfficientNet-B0 classification, evaluation, and image inference | COMPLETE |
| Phase 4 | USDA nutrition catalog and portion calculations | COMPLETE |
| Phase 5 | Local foods and full image-to-nutrition integration | COMPLETE |
| Phase 6 | Editable portions, daily log, goals, and dashboards | COMPLETE |
| Phase 7 | Mobile experience, final validation, and release | IN PROGRESS — model/TFLite preparation |

Phases 1–6 are complete. Phase 4 built an independent nutrition data layer for the
original ten labels. Phase 5 trained and evaluated its separate model and passed
end-to-end verification. Phase 6 adds reusable domain services for portions, meal
logging, goals, and summaries. Phase 7 model/deployment groundwork is in progress;
mobile UI and release work remain.

## Phase 2 — Complete

The verified Food-101 pipeline loads the real dataset, creates deterministic
development splits, prevents path and identical-content leakage, applies separate
transforms, and loads selected images through CPU DataLoaders.

## Phase 3 — Complete

Training used EfficientNet-B0 with ImageNet weights, full fine-tuning, AdamW (learning
rate and weight decay 0.0001), batch size 16, seed 42, and Apple MPS. Validation
accuracy reached 92.00% at epoch 4; epoch 4 is the selected checkpoint. Final held-out
evaluation reported loss 0.176723 and 94.48% classification accuracy on 2,500
official test images. This result is frozen and must not guide further model
selection or be described as calorie-estimation accuracy. Single-image inference is
implemented and verified synthetically. Detailed historical results remain in
`docs/ML_PIPELINE.md`.

## Phase 4 — Nutrition and calorie data layer

- Maintain a local, versioned USDA FDC nutrition catalog for the existing ten exact
  classifier labels.
- Preserve source record IDs, descriptions, data types, retrieval metadata, and
  source-backed default servings.
- Provide exact class lookup and per-100 g based calories, protein, carbohydrates,
  and fat calculations for default servings or user-selected portions.
- Validate catalog fields, provenance, numeric values, and exact class coverage.
- Keep this layer independent of PyTorch, classifier confidence, image-weight
  estimation, and application interfaces.

## Phase 5 — Local foods and image-to-nutrition integration (COMPLETE)

- Use the pinned TurkishFoods-25 source and its predefined `train`, `eval`, and
  `test` splits; retain only `borek`, `kebap`, and `yaprak_sarma`.
- Explicitly map source `yaprak_sarma` to canonical classifier label `sarma`.
- Preserve the first ten Phase 3 class labels/order, append `borek`, `kebap`,
  `sarma`, and train a separate 13-class EfficientNet-B0 using both datasets.
- Transfer compatible Phase 3 backbone weights only; initialize the expanded head
  for 13 outputs. Keep the Phase 3 checkpoint and frozen 94.48% result intact.
- Extend the versioned nutrition catalog with sourced regional references and
  provenance. Integrate image → prediction → exact nutrition lookup → default
  serving. Confidence remains informational; no image-based portion estimate.
- Exported data, training, separate evaluation, nutrition extension, and offline and
  image-to-nutrition checks succeeded. The 2,719-sample mixed test result was 95.66%
  accuracy (loss 0.17226); the report and per-class results are in
  `docs/ML_PIPELINE.md`.

## Phase 6 — Editable portions, meal log, goals, and dashboards (COMPLETE)

- `app/meal_log.py` creates meal entries from exact nutrition catalog IDs using the
  existing calculator for defaults, serving multipliers, or explicit grams.
- `app/repositories.py` defines the persistence boundary; `app/storage.py` provides
  validated atomic JSON storage, with missing initial storage treated as an empty
  diary. Runtime diary data is Git-ignored.
- `app/goals.py` stores a finite positive daily calorie goal. `app/dashboard.py`
  provides daily and weekly nutrition totals, progress and goal-overage information.
- The weekly interval is seven inclusive calendar dates ending on the requested
  date. UTC is the default day boundary; callers may supply an explicit timezone.
- `python -m app.verify_phase6` verifies portion modes, persistence, goals,
  aggregation, malformed data handling, and confidence independence.
- This phase is reusable Python/domain/application logic. It has no mobile UI and
  does not infer portion weight from images.

## Phase 7 — Mobile, device validation, and release (IN PROGRESS)

- Add the assignment-required MobileNetV3 TensorFlow Lite model as a separate
  deployment path while retaining PyTorch EfficientNet-B0 results as historical
  baselines. The MobileNetV3 model is trained, independently evaluated, exported,
  and verified; see `docs/ML_PIPELINE.md`.

- Implement the mobile experience for camera/gallery input, prediction and
  confidence display, nutrition lookup, editable portion, meal logging, goal, and
  daily/weekly progress.
- Choose and implement the deployment/integration strategy without replacing the
  verified Phase 5 PyTorch model blindly.
- Perform physical Android testing, final end-to-end validation, and prepare the
  APK/demo/release deliverables.

The mobile interface, camera/gallery integration, packaging/integration with the
mobile client, and device/release work belong to Phase 7 and have not started.
