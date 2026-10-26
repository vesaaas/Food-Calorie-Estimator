# Food Calorie Estimator

A fully offline Flutter app that classifies meal photos into 13 supported food
classes, calculates a representative nutrition estimate, and keeps a local meal
diary with daily and weekly summaries. It runs MobileNetV3 Small through TensorFlow
Lite on-device; it does not send images or diary data to a service.

## Features

- Capture a meal with Camera or choose an image from Gallery.
- Classify one of 13 supported foods with the bundled MobileNetV3 Small TFLite model.
- Look up calories, protein, carbohydrates, and fat in a locally bundled catalog
  based on USDA FoodData Central and regional Turkish-food references.
- Start from a reference serving or edit the portion in grams. The app does not
  estimate portion weight from pixels.
- Add, edit, and remove meal records; preserve meal photos in app-private storage.
- Add a supported food manually without running image classification.
- Set a daily calorie goal and view Today and rolling seven-day Weekly dashboards.
- Persist meals and the goal locally with SQLite; no account or network connection
  is required for the core app.

## Supported foods

`pizza`, `hamburger`, `sushi`, `steak`, `ice_cream`, `donuts`, `omelette`,
`french_fries`, `chicken_wings`, `spaghetti_bolognese`, `borek`, `kebap`, and
`sarma`.

## Architecture

~~~text
Camera / Gallery ──> MobileNetV3 Small TFLite ──> supported food label
                                                    │
Manual food selection ──────────────────────────────┤
                                                    ↓
                                  local nutrition catalog + editable grams
                                                    ↓
                                 saved meal snapshot + app-private photo
                                                    ↓
                             SQLite repository → Diary / Today / Weekly / Goal
~~~

The Flutter app uses `sqflite` with a versioned local database. The `meals` table
stores each nutrition snapshot, timestamp, classifier confidence metadata, portion
mode, and photo reference. The `goals` table stores the daily calorie goal;
`app_metadata` records successful one-time migrations. A timestamp index supports
date-oriented queries. The shared `DiaryRepository` boundary keeps persistence
behind the `DiaryController`.

On first access, the SQLite repository validates and imports the prior
`food_calorie_diary.json` file in one database transaction. It marks migration
complete only after the meal and goal writes succeed. The original JSON file is
preserved, and the migration marker prevents importing it again. Nutrition values
are saved as a snapshot when a meal is logged, so later catalog changes do not
rewrite diary history. Photo files stay in app-private Application Support storage;
SQLite retains their safe app-owned references.

## Machine learning and evaluation

The production model is MobileNetV3 Small exported to TFLite. On the 2,719-image
held-out 13-class test split, TFLite top-1 accuracy was **88.12%**. The model accepts
RGB float32 images resized by shortest edge to 256 pixels and center-cropped to
224×224; the bundled model performs its own input rescaling. The app keeps a 0.55
low-confidence policy: predictions below that value prompt the user to retry. This
cutoff is not a food/non-food detector or a general unsupported-food detector.

EfficientNet-B0 is historical, not the deployed mobile model. The original
10-class Phase 3 test result was **94.48% on 2,500 images**. The separate 13-class
Phase 5 EfficientNet experiment scored 95.66% on its mixed test split; results from
different class sets and experiments are not direct comparisons. A later
MobileNetV3 robustness fine-tuning candidate reached 88.23% on the same held-out
split but regressed on 7 of 13 classes, so it was rejected. Three-way triage and
binary food/non-food experiments were also rejected and are not part of the app.

The 13-class model is closed-set: an unsupported food such as cake or a non-food
object can still be assigned a supported label, sometimes with high confidence.
Nutrition values represent reference foods and portions; they are not laboratory
measurements of the photographed meal.

## Technology

- Flutter / Dart mobile client
- TensorFlow Lite via `flutter_litert`
- MobileNetV3 Small image classifier
- SQLite via `sqflite`
- USDA FoodData Central and regional-food nutrition references
- Python, PyTorch, TensorFlow, and torchvision for the dataset/model pipeline and
  offline verification

## Setup and run

Python tooling (needed for ML and nutrition verification):

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
~~~

Run the mobile app with Flutter (and Xcode for a physical iPhone):

~~~bash
cd mobile
flutter pub get
flutter run
~~~

The production TFLite asset is bundled at
`mobile/assets/models/mobilenet_v3_small_phase7_v2.tflite`. The ordered label list
and input contract are in `ml/mobile/`.

## Verification

~~~bash
cd mobile
flutter analyze
flutter test
cd ..
python nutrition/src/verify_nutrition.py
python -m app.verify_phase6
python -m ml.src.verify_mobilenet_v3
~~~

The Flutter suite covers persistence, migration, diary, goals, dashboards,
classification boundaries, and photo handling. The nutrition verifier checks the
bundled catalog against its canonical data; the Phase 6 Python verifier protects
the standalone Python domain services.

## Validation and limitations

The current app passed automated Flutter, nutrition, and Phase 6 checks and was
manually verified on a physical iPhone, including SQLite migration, meal/photo and
goal persistence across force-close/relaunch, Camera/Gallery classification,
editable grams, manual entry, and Today/Weekly dashboards. Core flows were also
checked offline. Android physical-device validation has not been performed.

The classifier recognizes only the 13 listed classes. Unsupported foods and
non-food images may map to a supported class; 0.55 only rejects low-confidence
predictions. Image-based portion-size estimation is not implemented. iOS logs
duplicate Objective-C class warnings from the bundled LiteRT frameworks; no
workaround has been applied.

See [the roadmap](docs/ROADMAP.md), [architecture](docs/ARCHITECTURE.md),
[ML pipeline](docs/ML_PIPELINE.md), [nutrition data](docs/NUTRITION.md), and
[dataset notes](docs/DATASET.md) for implementation and evaluation details.
