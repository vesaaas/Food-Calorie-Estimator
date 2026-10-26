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
| Phase 7 | Mobile App Implementation (7.1–7.8) | COMPLETE — automated and physical iPhone verification passed |
| Phase 8 | Full Testing & Validation | COMPLETE |
| Phase 9 | Final Documentation and repository review | COMPLETE |
| Phase 10 | Demo & Submission | PLANNED |

Phases 1–6 are complete. Phase 4 built an independent nutrition data layer for the
original ten labels. Phase 5 trained and evaluated its separate model and passed
end-to-end verification. Phase 6 adds reusable domain services for portions, meal
logging, goals, and summaries. Phase 7 implementation now supports
camera/gallery input, on-device classification, nutrition lookup, editable grams,
local meal logging, a calorie goal, and Today/Weekly dashboards. Phase 7.7 and 7.8
physical iPhone verification passed, as did the final end-to-end Phase 7 flow.
Flutter production persistence is versioned local SQLite with a safe, idempotent
legacy JSON migration; the JSON source is preserved. Automated migration tests and
physical iPhone verification of migration, meal/goal/photo persistence, dashboards,
and restart behavior are complete. Phase 8 automated and physical iPhone validation
is complete. Phase 9 final documentation and repository review is complete.
Android physical-device validation was not performed.

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
- `app/repositories.py` defines the standalone Python Phase 6 persistence boundary;
  `app/storage.py` provides validated atomic JSON storage for those Python
  services. Flutter production persistence is a separate SQLite adapter.
- `app/goals.py` stores a finite positive daily calorie goal. `app/dashboard.py`
  provides daily and weekly nutrition totals, progress and goal-overage information.
- The weekly interval is seven inclusive calendar dates ending on the requested
  date. UTC is the default day boundary; callers may supply an explicit timezone.
- `python -m app.verify_phase6` verifies portion modes, persistence, goals,
  aggregation, malformed data handling, and confidence independence.
- This phase is reusable Python/domain/application logic. It has no mobile UI and
  does not infer portion weight from images.

## Phase 7 — Mobile App Implementation (IMPLEMENTATION AND IPHONE VERIFICATION COMPLETE)

- Add the assignment-required MobileNetV3 TensorFlow Lite model as a separate
  deployment path while retaining PyTorch EfficientNet-B0 results as historical
  baselines. The MobileNetV3 model is trained, independently evaluated, exported,
  and verified; see `docs/ML_PIPELINE.md`.

- Phase 7.4 added camera/gallery acquisition. Phase 7.5 added the MobileNetV3 TFLite
  deployment model and was verified on a physical iPhone.
- Phase 7.6 added canonical-catalog nutrition lookup and editable portions; physical
  iPhone verification confirmed French Fries at 110 g and recalculation at 200 g.
- Phase 7.7 added local meal snapshots, today's diary, manual food entry, persisted
  meal photos, and a persisted daily calorie goal.
- Phase 7.8 connects the Dashboard to the shared diary state. Today shows snapshot
  calories/macros, meal count, and optional goal progress. Weekly shows a rolling
  seven-device-local-day calorie chart, total, seven-day average, and meal count.
  Physical iPhone verification passed. The completed Phase 7 flow also passed final
  end-to-end physical iPhone verification using a successful Profile-mode build,
  install, and launch. The known closed-set limitation is documented in
  `docs/ML_PIPELINE.md` and does not block Phase 7 completion.

- SQLite is production persistence for Flutter. `sqflite` stores meal snapshots in
  `meals`, the daily goal in `goals`, and the one-time JSON migration marker in
  `app_metadata`; `logged_at_utc` is indexed. Migration preserves meals, goals, and
  photo references, marks completion only after successful writes, and retains the
  source JSON. Automated migration tests and physical iPhone verification passed.
- The final physical check confirmed old meals/goals/photos survived migration,
  add/edit/remove, goal updates, Today/Weekly summaries, restart persistence,
  Camera/Gallery classification, editable grams, classified meal saves, and manual
  entry. Core app operation was checked offline.

## Phase 8 — Full Testing & Validation (COMPLETE)

Automated regression verification passed: Flutter analysis and all 74 Flutter
tests, including SQLite migration and persistence tests;
17 nutrition verification tests; 13 Phase 6 regression tests; the MobileNetV3/
TFLite verifier; and the confidence evaluator self-test. The held-out TFLite
accuracy remains 88.12% on 2,719 images, and representative TFLite/Keras top-1
predictions agreed on 38 of 39 images. `git diff --check` passed.

The stale local-day UI issue was reproduced and fixed with app-level lifecycle
resume handling and a shared `DiaryController` date-change notification. It
refreshes Diary Today and Dashboard Today/Weekly without changing or persisting
meals, goals, photos, or nutrition snapshots; same-day resume does not notify.

Physical iPhone validation passed for Profile build/install/launch, camera/gallery
and real TFLite inference, nutrition and editable portions, Diary/Goal/Dashboard,
force-quit and independent relaunch persistence, saved photos, and operation with
Airplane Mode enabled and Wi-Fi off, including offline classification. Android
build/runtime validation was not performed because this environment had no Android
SDK, emulator, toolchain, or physical Android device. It is an explicit limitation,
not a tested platform claim or blocker for this deliverable.

The 0.55 cutoff remains a low-confidence rejection threshold for a closed-set
13-class model, not food-versus-non-food detection. Unsupported foods and non-food
objects can still receive supported-class predictions at confidence >= 0.55. On
iOS, duplicate Objective-C class warnings between `LiteRt.framework` and
`LiteRtMetalAccelerator.framework` were observed while the tested build, launch,
and on-device inference completed successfully. No framework workaround was
applied, and the warning has not been confirmed harmless.

## Phase 9 — Final Documentation (COMPLETE)

The README, architecture, ML results and limitations, nutrition guide, roadmap, and
current-state notes now match the verified app. Rejected robustness experiments are
identified as experiments; their generated data, reports, and checkpoints remain
ignored by Git.

## Phase 10 — Demo & Submission (PLANNED)

Prepare the demo and submission deliverables from the validated application.
