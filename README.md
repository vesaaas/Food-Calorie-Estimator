# Food Calorie Estimator

Food Calorie Estimator is an offline-first Flutter app that recognizes meals from
photos, estimates nutrition from a curated food catalog, and keeps a local meal
diary. Classification runs on the device. Users can adjust the portion size,
review the resulting nutrition estimate, and track daily and weekly intake.

## Key features

- Capture a photo or choose one from the gallery for food recognition.
- Run MobileNetV3 Small through TensorFlow Lite on-device across 13 supported foods.
- Look up calories, protein, carbohydrates, and fat in a USDA-based local catalog,
  with sourced references for regional foods.
- Edit portion size in grams; weight is not inferred from the image.
- Add foods manually, then add, edit, or remove meals from the diary.
- Set a daily calorie goal and review Today and rolling seven-day Weekly dashboards.
- Persist meal snapshots, goals, and photo references locally with SQLite.
- Use core app workflows without an account or network connection.

## Architecture

```text
Camera / gallery image → TFLite classifier → food class → nutrition catalog
  → editable grams → SQLite meal snapshot → diary and dashboards
```

The Flutter app uses `sqflite` and a versioned app-private database. `meals` stores
nutrition snapshots and meal metadata, `goals` stores the daily calorie goal, and
`app_metadata` records migration state. Meal photos are kept in app-private
Application Support storage and referenced by their meal records. Today and Weekly
summaries are calculated from locally persisted meals. A one-time transactional
migration imports an existing JSON diary when present; JSON is not the production
storage mechanism.

## AI/ML model

The production mobile classifier is **MobileNetV3 Small**, developed with
TensorFlow/Keras and exported to TensorFlow Lite for on-device inference. On the
held-out 13-class test set of **2,719 images**, the deployed TFLite model achieved
**88.12% top-1 accuracy**. The input is RGB, resized with the shortest edge at 256
pixels, then center-cropped to 224×224.

The app uses **0.55** as a low-confidence handling threshold. It prompts users to
retry below that value; it does not detect non-food or unsupported foods. Historical
EfficientNet-B0 experiments are documented separately and are not the production
mobile model.

## Supported foods

`pizza`, `hamburger`, `sushi`, `steak`, `ice_cream`, `donuts`, `omelette`,
`french_fries`, `chicken_wings`, `spaghetti_bolognese`, `borek`, `kebap`, `sarma`.

## Technology stack

- Python, TensorFlow/Keras, TensorFlow Lite, and torchvision for ML data and evaluation
- Flutter and Dart for the mobile application
- SQLite through `sqflite` for local persistence
- Food-101 and selected TurkishFoods-25 classes for model development
- USDA FoodData Central and documented regional references for nutrition data
- Git and GitHub for version control and project hosting

## Testing and validation

- `flutter analyze` completed cleanly; **74 Flutter tests** passed.
- Nutrition catalog verification passed **17 checks**.
- Python meal, goal, and dashboard regression verification passed **13 checks**.
- The production TFLite verifier evaluated all 2,719 held-out images.
- Manual physical-device validation was performed on iPhone, including offline use,
  camera/gallery inference, SQLite migration, meal/photo/goal persistence, and both
  dashboards. Android physical-device validation was not performed.

## Limitations

The classifier is closed-set: unsupported foods and non-food images may be assigned
to one of the 13 supported classes, including with high confidence. The 0.55 policy
only handles low-confidence predictions and is not an open-set detector. Portion
weight is entered by the user rather than inferred from an image. Nutrition values
are estimates based on representative reference foods and do not capture every
recipe or preparation method.

## Running the project

Install the mobile dependencies and launch from the project root:

```bash
cd mobile
flutter pub get
flutter run
```

For Python ML and nutrition tooling, use Python 3.12 and install the pinned project
requirements:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The production model is bundled at
`mobile/assets/models/mobilenet_v3_small_phase7_v2.tflite`; its ordered labels and
input contract are under `ml/mobile/`.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Current production state](docs/CURRENT_STATE.md)
- [ML pipeline and evaluation](docs/ML_PIPELINE.md)
- [Nutrition methodology](docs/NUTRITION.md)
- [Dataset details](docs/DATASET.md)
- [Engineering roadmap](docs/ROADMAP.md)
