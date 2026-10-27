# Architecture

## Production data flow

```text
Camera / gallery image
        ↓
RGB resize (short edge 256) and center crop (224 × 224)
        ↓
MobileNetV3 Small in TensorFlow Lite (on device)
        ↓
Canonical food class ───────────── Manual food selection
        ↓                                  ↓
Exact local nutrition catalog lookup ←────┘
        ↓
Default serving or user-entered grams → calories and macronutrients
        ↓
SQLite meal snapshot + app-private photo reference
        ↓
Diary, Today dashboard, Weekly dashboard, and calorie goal
```

## Mobile application

The Flutter client separates image acquisition, classification, nutrition lookup,
diary state, and persistence. Camera and gallery images are passed to the bundled
TFLite model through `flutter_litert`. Its output index maps through the ordered
labels in `ml/mobile/mobilenet_v3_labels.txt`. The 0.55 threshold is only a
low-confidence retry policy. Classifier confidence is informational and does not
change nutrition values or portion size.

The mobile nutrition catalog is a bundled mirror of `nutrition/data/foods.json`.
Exact food IDs select a catalog entry; selected grams scale per-100 g nutrient
values. Manual entry uses the same catalog and meal-saving flow without invoking
the classifier.

## SQLite and offline behavior

`DiaryController` coordinates scan results, meals, goals, and dashboards through a
`DiaryRepository` boundary. Production uses `SqliteDiaryRepository` with a
versioned database managed by `sqflite`:

- `meals` stores stable IDs, food IDs and display names, UTC timestamps, portion
  mode and grams, saved calories/macros, optional confidence, photo reference, and
  ordering.
- `goals` stores the daily calorie goal.
- `app_metadata` records successful migration state.
- An index on `logged_at_utc` supports date-based reads.

Nutrition is saved as a meal snapshot, so catalog edits do not rewrite earlier
entries. Photo files are stored in app-private Application Support storage; SQLite
stores safe app-owned references. Today and rolling seven-day Weekly summaries are
derived from locally persisted meal records and grouped by device-local calendar
dates. Core classification, nutrition, logging, and dashboard workflows require no
network connection.

When a legacy `food_calorie_diary.json` is present, its contents are validated and
copied into SQLite in one transaction. The migration marker is written only after
successful database writes, and the source file is retained. This is a one-time
backward-compatibility path; JSON is not current production persistence.

## Production ML contract

The deployed model is TensorFlow/Keras MobileNetV3 Small exported as a float32
TensorFlow Lite model with dynamic-range weight optimization. It accepts NHWC
`[1,224,224,3]` RGB float32 input in `[0,255]` and returns 13 softmax scores. Keras
performs its built-in input rescaling; callers must not apply ImageNet mean/std
normalization. The held-out TFLite result is 88.12% on 2,719 images. The model is
closed-set and has no explicit unknown class. Details and evaluation methodology are
in [`ML_PIPELINE.md`](ML_PIPELINE.md).

## Nutrition boundary

The nutrition layer uses exact class IDs and a versioned catalog. Per-100 g
calories, protein, carbohydrates, and fat are scaled by the selected or default
portion. Catalog provenance and calculation assumptions are described in
[`NUTRITION.md`](NUTRITION.md). Neither the classifier nor the nutrition layer infers
portion mass or exact recipe composition from pixels.

## Python research and domain modules

The repository also contains reproducible dataset/model tooling and standalone
Python nutrition and meal-domain services. `app/storage.py` is a JSON adapter for
the standalone Python services only; the Flutter production app persists to SQLite.
Historical PyTorch EfficientNet-B0 models are retained for development and evaluation
context. They are not bundled or used for production mobile inference.
