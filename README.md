# Food Calorie Estimator

Food Calorie Estimator is an offline-first Flutter app that classifies meal photos,
estimates nutrition from a curated food catalog, and keeps a local meal diary.
Classification runs on the device. Users confirm supported-food predictions and can
adjust portion size before logging nutrition estimates.

## Key features

- Capture a photo or choose one from the gallery for food recognition.
- Run MobileNetV3 Small through TensorFlow Lite with 13 supported food labels and a
  separate `non_food` rejection label.
- Confirm a supported prediction before nutrition lookup; retry or choose one of the
  supported foods manually when the prediction is wrong.
- Look up calories, protein, carbohydrates, and fat in a USDA-based local catalog,
  with sourced references for regional foods.
- Edit portion size in grams; weight is not inferred from the image.
- Add, edit, and remove meals from the diary, then review daily and weekly summaries.
- View seven-day calories, protein, carbohydrates, fat, and estimated macro-energy
  distribution in the Weekly dashboard.
- Persist meal nutrition snapshots, goals, and photo references locally with SQLite.
- Use core workflows without an account or network connection.

## Architecture

```text
Camera / gallery → EXIF-corrected RGB preprocessing → 14-output TFLite model
  → non_food rejection OR supported-food confirmation
  → catalog lookup → editable portion → SQLite meal snapshot → diary/dashboard
```

The Flutter app uses `sqflite` and a versioned app-private database. `meals` stores
nutrition snapshots and meal metadata, `goals` stores the daily calorie goal, and
`app_metadata` records migration state. Meal photos are kept in app-private
Application Support storage. Today and Weekly summaries are calculated from
persisted meals. A one-time transactional migration imports an existing JSON diary;
JSON is not the production storage mechanism.

## AI/ML model

The current mobile candidate is MobileNetV3 Small exported to float32 TensorFlow
Lite. Its 14 output labels, in order, are:

```text
pizza, hamburger, sushi, steak, ice_cream, donuts, omelette,
french_fries, chicken_wings, spaghetti_bolognese, borek, kebap, sarma, non_food
```

Input is EXIF-corrected RGB, resized with the shortest edge at 256 pixels using
bilinear interpolation, then center-cropped to 224×224. The model receives NHWC
float32 `[1,224,224,3]` values in `[0,255]`; MobileNetV3's internal rescaling maps
the pixels to `[-1,1]`, so callers do not normalize them again. The frozen policy is
argmax over all 14 outputs: indices 0–12 are supported foods and index 13 is
`non_food`. There is no 0.55 acceptance threshold. Confidence is informational.

On the held-out evaluation, the 14-class TFLite candidate had **90.33% supported-food
classification accuracy**, **99.63% supported-food acceptance**, **94.33% non-food
rejection**, and **2.03% unsupported-food rejection**. Supported-food acceptance
means an image was assigned a supported label; it is distinct from correct
classification. The model rejects many non-food objects but is **not a reliable
unsupported-food detector**. Unsupported foods may still be assigned to supported
classes, including with high confidence. The previous 13-class model remains in
`mobile/assets/models/mobilenet_v3_small_phase7_v2.tflite` for rollback.

The Phase 9.5 float32 TFLite export matched its selected 14-class Keras checkpoint
on the frozen held-out metrics and passed representative Keras/TFLite parity checks.
These 14-class results are separate from the previous deployed 13-class TFLite
result of **88.12%** on 2,719 supported-food images. The 13-class Keras result of
**89.41%** is also a different runtime/model comparison; neither baseline should be
conflated with the 14-class candidate.

## User confirmation and nutrition

An argmax result for a supported class shows its predicted name and informational
confidence and asks whether it is correct. Nutrition lookup and diary saving remain
locked until the user confirms. If it is wrong, the user can retry with Camera or
Gallery, manually select one of the 13 supported categories and confirm that choice,
or say that the food is not listed. A manually chosen result is labeled as
user-selected. `non_food` is rejected before nutrition lookup or diary saving.

Manual selection is a user choice, not model detection. If the actual food is outside
the 13 supported categories, the app explains that it is unsupported and does not
require the user to choose an incorrect category.

## Weekly dashboard

The Weekly dashboard retains its seven-day calorie chart and adds daily protein,
carbohydrate, and fat charts from persisted meal nutrition snapshots. Its estimated
macro-energy distribution converts protein and carbohydrates at 4 kcal/g and fat at
9 kcal/g, then normalizes those three contributions to 100%. This chart describes
estimated energy from the logged macros; its percentages are **not percentages of
total logged calories**. No micronutrient visualization is provided.

## Supported foods

`pizza`, `hamburger`, `sushi`, `steak`, `ice_cream`, `donuts`, `omelette`,
`french_fries`, `chicken_wings`, `spaghetti_bolognese`, `borek`, `kebap`, `sarma`.

## Technology stack

- Flutter and Dart for the mobile application
- TensorFlow/Keras and TensorFlow Lite for model development and on-device inference
- SQLite through `sqflite` for local persistence
- Food-101 and selected TurkishFoods-25 data for food-class model development
- USDA FoodData Central and documented regional references for nutrition data
- Python tooling for ML experiments, dataset preparation, and nutrition verification

## Testing and validation

- `flutter analyze` completed cleanly and the Flutter suite passed **90 tests**.
- Nutrition catalog verification passed 17 checks; Python meal, goal, and dashboard
  regression verification passed 13 checks.
- The 14-class candidate's held-out metrics were evaluated separately in Phase 9.4
  and its float32 TFLite export and parity were checked in Phase 9.5.
- Physical Android checks for the Phase 9.6–9.7 app changes were **reported by the
  user as successful**. This is not an automated test or an independently reproduced
  result. Device/version details and broader device coverage should be recorded with
  final release verification.
- Earlier v1.0.0 physical-device validation is historical and does not replace
  v1.1.0 release checks.

## Android APK

The latest **public** release remains **v1.0.0**. Its APK is available through
[GitHub Releases](https://github.com/vesaaas/Food-Calorie-Estimator/releases/download/v1.0.0/app-release.apk).
The v1.1.0 candidate (`1.1.0+2`) has been built locally and its APK signature,
package metadata, and checksum have been verified. It has not been published.
Final Android APK physical-device testing remains pending. See the
[Android release signing guide](docs/ANDROID_RELEASE.md) and
[v1.1.0 release notes](docs/RELEASE_NOTES_v1.1.0.md).

The original v1.0.0 APK was debug-signed. A dedicated release key is used for
v1.1.0, so Android will not allow a direct update over the v1.0.0 installation.
Removing the old app before installing v1.1.0 may erase its local diary and saved
meal photos. Back up local data first.

The locally verified v1.1.0 APK is `mobile/build/app/outputs/flutter-apk/app-release.apk`
(103,563,167 bytes). Its SHA-256 is:

```text
b71e0eb55b2fa5e4bec0ce1ae7c3c237809f9540a97d603f84e5d07ca535c696
```

It is package `com.example.food_calorie_estimator`, version name/code `1.1.0` / `2`,
min SDK 24, target SDK 36, signed with APK Signature Scheme v2. The signer
certificate SHA-256 is recorded in the release guide. This build artifact is local
and ignored by Git; do not interpret it as a published GitHub release.

The validated v1.0.0 release APK SHA-256 is:

```text
0aadd6c4e112d788e83201f45d0473f9c3488ab3b3d1f1c845737fbb50fa6c39
```

## Limitations

The `non_food` class is not a general OOD detector. Unsupported foods can still
receive high-confidence supported-food predictions. User confirmation is a safety
step, not a substitute for unsupported-food detection. Portion weight is entered by
the user rather than inferred from an image. Nutrition values are estimates based on
representative reference foods and do not capture every recipe or preparation.

## Running the project

From the project root, install the Flutter dependencies and launch the app:

```bash
cd mobile
flutter pub get
flutter run
```

For Python ML and nutrition tooling, use Python 3.12 and install the pinned project
requirements in a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Current production state](docs/CURRENT_STATE.md)
- [ML pipeline and evaluation](docs/ML_PIPELINE.md)
- [Android release signing and upgrade impact](docs/ANDROID_RELEASE.md)
- [v1.1.0 release notes and verification status](docs/RELEASE_NOTES_v1.1.0.md)
- [Nutrition methodology](docs/NUTRITION.md)
- [Dataset details](docs/DATASET.md)
- [Engineering roadmap](docs/ROADMAP.md)
