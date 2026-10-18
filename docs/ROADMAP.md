# Food Calorie Estimator Roadmap

## Status

| Phase | Scope | Status |
| --- | --- | --- |
| Phase 1 | Project setup and Python environment | COMPLETE |
| Phase 2 | Dataset and preprocessing | COMPLETE |
| Phase 3 | Food classification model training and evaluation | NEXT |
| Phase 4 | Nutrition and calorie data layer | PLANNED |
| Phase 5 | Inference API and application integration | PLANNED |
| Phase 6 | Mobile experience, validation, and release hardening | PLANNED |

Phase 2 is marked complete because the verified Food-101 pipeline loads the real
dataset, creates deterministic development splits, prevents path and identical
content leakage, applies separate transforms, and successfully loads every selected
image through CPU DataLoaders.

## Phase 3 — Next

The next phase should add a food classification model using the existing train,
validation, and test DataLoaders. It should define model selection, training,
checkpointing, evaluation, and inference behavior. It must keep the official test
split isolated until final evaluation.

Phase 3 is intentionally not implemented in this checkpoint.
