# Food Calorie Estimator Roadmap

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| Phase 1 | Project setup and Python environment | COMPLETE |
| Phase 2 | Food-101 dataset and preprocessing | COMPLETE |
| Phase 3 | EfficientNet-B0 classification, evaluation, and image inference | COMPLETE |
| Phase 4 | USDA nutrition catalog and portion calculations | COMPLETE |
| Phase 5 | Local foods and full image-to-nutrition integration | COMPLETE |
| Phase 6 | Editable portions, daily log, goals, and dashboards | PLANNED |
| Phase 7 | Mobile experience, final validation, and release | PLANNED |

Phases 1–4 are complete. Phase 4 built an independent nutrition data layer for the
original ten labels. Phase 5 has trained and evaluated its separate model and passed
end-to-end verification. Later phases remain planned.

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

## Later phases — Planned

### Meal portions, log, and progress

- Provide an editable portion interface using serving multipliers or user-entered
  weight; do not infer weight from an image without a separately validated method.
- Let users add and revisit meals in a daily log.
- Let users set a personal daily calorie goal.
- Build a daily calorie progress dashboard and weekly calorie progress dashboard.

### Mobile, validation, and release

- Implement the mobile meal-photo and nutrition experience.
- Validate end-to-end behavior, local foods, nutrition provenance, portion editing,
  logging, goals, and daily/weekly summaries.
- Harden, document, and deliver the final release.

The later phases still include editable portion UI, daily meal log, personal daily
calorie goal, daily and weekly dashboards, mobile experience, and final
validation/release. Do not begin them as part of Phase 5.
