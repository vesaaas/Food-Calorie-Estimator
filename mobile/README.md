# Food Calorie Estimator mobile app

The Flutter client supports camera/gallery food recognition, on-device
MobileNetV3 Small TFLite inference, nutrition lookup, editable portions, manual food
entry, meal logging, a daily calorie goal, and Today/Weekly dashboards. The app's
core workflow is local and offline-first.

## Run

From this directory:

```bash
flutter pub get
flutter run
```

The production model is `assets/models/mobilenet_v3_small_phase7_v2.tflite`. It
supports the 13 labels in the root README and
`../ml/mobile/mobilenet_v3_labels.txt`. Images are converted to RGB, resized with
the shortest edge at 256, then center-cropped to 224×224. The 0.55 cutoff handles
low-confidence predictions only; it does not detect non-food or unsupported foods.

## Nutrition

`assets/nutrition/foods.json` is the app's bundled mirror of the canonical catalog
at `../nutrition/data/foods.json`. Exact class IDs select food references. The user
can edit the default portion in grams; nutrients are scaled from per-100 g values.
Portion weight is not inferred from the image. Manual food entry selects from the
same catalog without running classification.

## Local persistence

Production persistence uses `sqflite` and a versioned app-private SQLite database.
The `meals` table stores immutable nutrition snapshots, timestamp, portion details,
confidence metadata, and a safe photo reference. The `goals` table stores the daily
calorie goal; `app_metadata` records migration state. Meal photos remain in
app-private Application Support storage. Today and rolling seven-day Weekly views
are derived from persisted meals and grouped by device-local dates.

A one-time transaction migrates an existing `food_calorie_diary.json` file into
SQLite after validating its records. The marker is saved only after the writes
succeed, and the source is retained. The app does not use JSON for current
production persistence.

## Verification

```bash
flutter analyze
flutter test
```

The current suite has 74 passing tests. Physical iPhone validation covered camera
and gallery classification, nutrition, editable grams, meal and photo persistence,
SQLite migration, goals, dashboards, relaunch persistence, and offline workflows.
Android physical-device validation has not been performed.
