# Current Production State

Food Calorie Estimator is a Flutter app for on-device food classification, nutrition
estimation, meal logging, and daily/weekly summaries. The core workflow is local and
works offline.

## Mobile production model

The deployed classifier is Keras MobileNetV3 Small exported to TensorFlow Lite. It
recognizes 13 supported classes and achieved **88.12% top-1 accuracy** on a held-out
test set of **2,719 images**. The app applies a **0.55 low-confidence retry
threshold**; this is not food/non-food or unsupported-food detection. The complete
class list, preprocessing contract, evaluation method, and limitations are in
[`ML_PIPELINE.md`](ML_PIPELINE.md).

## Application and persistence

The Flutter app supports camera/gallery recognition, manual food selection, local
nutrition lookup, editable gram portions, meal add/edit/remove, saved meal photos,
a daily calorie goal, and Today/Weekly dashboards. `sqflite` stores the production
diary in a versioned app-private SQLite database. The `meals` table stores
nutrition snapshots and photo references, `goals` stores the calorie goal, and
`app_metadata` records migration state. Photos are stored in app-private
Application Support storage. A one-time transactional migration supports existing
JSON diaries; SQLite is the production persistence mechanism. Dashboard data is
derived from persisted meals.

## Validation

- Flutter static analysis completed cleanly and all **74 Flutter tests** passed.
- Nutrition verification passed **17 checks**; Phase 6 domain regression passed
  **13 checks**.
- The production TFLite verifier covered the full 2,719-image test split.
- Physical iPhone validation covered inference, nutrition and portions, diary,
  dashboards, migration, persistence across relaunch, and offline workflows.
- Android physical-device validation was not performed.

## Limitations

The classifier is closed-set, so unsupported foods and non-food images may map to a
supported class. The 0.55 threshold only handles low-confidence predictions.
Portion weight is user-entered and is not inferred from the image. Nutrition values
are representative estimates. See [`NUTRITION.md`](NUTRITION.md) and
[`ML_PIPELINE.md`](ML_PIPELINE.md) for details.

The Python Phase 6 domain services retain a JSON repository adapter for standalone
compatibility and verification. The Flutter app uses SQLite; the JSON adapter is not
its production storage layer.
