# Food Calorie Estimator Roadmap

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| Phase 1 | Project setup and Python environment | COMPLETE |
| Phase 2 | Food-101 dataset and preprocessing | COMPLETE |
| Phase 3 | EfficientNet-B0 classification, evaluation, and image inference | COMPLETE |
| Phase 4 | USDA nutrition catalog and portion calculations | COMPLETE |
| Phase 5 | Local foods and full image-to-nutrition integration | PLANNED |
| Phase 6 | Editable portions, daily log, goals, and dashboards | PLANNED |
| Phase 7 | Mobile experience, final validation, and release | PLANNED |

Phases 1–4 are complete. Phase 4 built an independent nutrition data layer for the
current ten classifier labels. The later phases below make the remaining project
assignment requirements explicit; they are not Phase 4 work.

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

## Future work — Planned

### Local food support and image-to-nutrition integration

- Collect and document a local dish dataset and classes, with appropriate usage
  rights, labels, and validation data.
- Extend the classifier and its training/evaluation pipeline to recognize local
  foods; define versioned class mappings and validation before adding local-food
  nutrition entries.
- Integrate image prediction → exact food mapping → nutrition lookup into the
  application flow. Keep class confidence separate from nutrition values and provide
  a useful response for unsupported/uncertain foods.

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

Phase 5–7 sequencing may be refined during implementation, but all listed assignment
requirements remain future work until then.
