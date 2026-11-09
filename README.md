# Food Calorie Estimator

Food Calorie Estimator is an offline-first Flutter application for recognizing a
limited set of foods, estimating nutrition from a curated catalog, and keeping a
private meal diary on the device. The current maintenance candidate is
**v1.1.1+3**; the latest published Android release is **v1.1.0**.

## Features

- Capture or select a meal photo and run MobileNetV3 Small inference locally with
  TensorFlow Lite.
- Recognize 13 supported categories: `pizza`, `hamburger`, `sushi`, `steak`,
  `ice_cream`, `donuts`, `omelette`, `french_fries`, `chicken_wings`,
  `spaghetti_bolognese`, `borek`, `kebap`, and `sarma`; a fourteenth `non_food`
  output rejects many non-food objects.
- Confirm supported predictions before nutrition is shown. When a prediction is
  wrong, retry with Camera or Gallery, manually select and confirm a supported
  category, or choose “My food isn't listed.”
- Estimate calories, protein, carbohydrates, and fat from a sourced local catalog;
  adjust portion grams before saving.
- Add, edit, and remove meals in a SQLite diary. Review daily goals and seven-day
  calories, macros, and estimated macro-energy distribution.
- Use the core recognition, nutrition, and diary workflows without an account or
  network connection.

## Model and limitations

The mobile model is a float32 14-output MobileNetV3 Small TFLite model. Its ordered
labels are the 13 supported foods above followed by `non_food`. Images are oriented
using EXIF metadata, decoded to RGB, resized with their shortest edge at 256 pixels,
center-cropped to 224×224, and passed as float32 NHWC values in `[0,255]`. The model
performs its own pixel rescaling. The frozen policy is argmax, without a confidence
acceptance threshold; displayed confidence is informational.

On the held-out evaluation, supported-food classification accuracy was 90.33%,
supported-food acceptance 99.63%, non-food rejection 94.33%, and unsupported-food
rejection 2.03%. Acceptance and correct classification are different measures. The
model is **not a reliable unsupported-food detector**: an unsupported food may be
classified as a supported category, even with high confidence. User confirmation is
a safeguard, not a substitute for detection. Portion weight is entered by the user
and is not inferred from a photo.

Nutrition values are representative estimates from the bundled catalog, not a
measurement of a particular recipe or meal. The weekly macro-energy chart converts
protein and carbohydrates at 4 kcal/g and fat at 9 kcal/g, then normalizes those
three contributions to 100%. Its percentages are not shares of total logged
calories. The app does not provide micronutrient charts.

## Technology

- Flutter/Dart for the mobile application and TensorFlow Lite for on-device
  MobileNetV3 inference
- SQLite (`sqflite`) for the local meal diary and persisted nutrition snapshots
- TensorFlow/Keras and Python tools for model training and evaluation
- Food-101 and selected TurkishFoods-25 data for supported foods; a selected Open
  Images V7 subset for the `non_food` training label
- USDA FoodData Central and documented regional references for the nutrition
  catalog

## Privacy and local data

Inference runs on-device. The diary and saved meal photos are stored in app-private
local storage; this project does not transmit them to a server as part of the scan
and logging flow; the Android manifest does not request Internet permission. The
SQLite database and photos are not application-level encrypted and rely on the
operating system's app sandbox. Saved photo files are copied as supplied and may
retain EXIF metadata. The v1.1.1 candidate explicitly excludes app data from
Android cloud backup and device-transfer rules. Local records can therefore be
lost when uninstalling the app or changing devices; keep an independent copy of
anything that must be retained. Android backup behavior can vary by manufacturer.

## Android installation

Download the APK from the [Food Calorie Estimator v1.1.0 GitHub
Release](https://github.com/vesaaas/Food-Calorie-Estimator/releases/tag/v1.1.0).
The published asset is named `app-release.apk`. Review the release notes and
checksum before installation.

The original v1.0.0 APK was debug-signed; v1.1.0 uses a dedicated release signer.
Android cannot install v1.1.0 as an in-place update over v1.0.0. Removing the old
installation may erase its local diary and meal photos. Back up important data
before uninstalling. The app has no in-app diary export or migration tool; if an
independent backup is unavailable, uninstalling may permanently lose local data.
The v1.1.1 candidate retains the v1.1.0 release signing identity, so it is intended
to update v1.1.0 installations; it is not yet published.

The archived [v1.0.0 APK](https://github.com/vesaaas/Food-Calorie-Estimator/releases/download/v1.0.0/app-release.apk)
has SHA-256 `0aadd6c4e112d788e83201f45d0473f9c3488ab3b3d1f1c845737fbb50fa6c39`.
The v1.1.0 APK checksum and package/signing details are in the
[published release notes](docs/RELEASE_NOTES_v1.1.0.md).

## Run locally

Install Flutter dependencies and launch from `mobile/`:

```bash
cd mobile
flutter pub get
flutter run
```

Run the mobile checks from the same directory:

```bash
flutter analyze
flutter test
```

Python ML and nutrition utilities use the pinned root `requirements.txt`; see the
ML pipeline and nutrition guides before preparing local datasets. Training data,
checkpoints, and experiment outputs are not required to run the app.

## Repository layout

```text
mobile/       Flutter app, tests, and versioned model/catalog assets
ml/           Model and dataset pipeline plus isolated experiments
nutrition/    Canonical nutrition catalog and calculation tools
app/          Python domain and storage utilities
docs/         Architecture, data, ML, nutrition, and release documentation
```

## Platform and project status

Android is the released platform. The v1.1.0 Android APK passed functional testing
on a physical Samsung device, including recognition, nutrition, meal logging, and
dashboard flows. The v1.1.1 maintenance candidate includes Android backup/privacy
hardening and documentation/version updates. Its isolated release build and static
artifact checks passed; physical-device verification and publication are pending.
The Flutter project contains an iOS implementation, but
iOS Release-mode device testing and App Store distribution signing are pending.

The v1.1.0 release and its published APK are unchanged. The 13-class TFLite model
remains bundled as a rollback asset.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Current state](docs/CURRENT_STATE.md)
- [Engineering roadmap](docs/ROADMAP.md)
- [Dataset and split methodology](docs/DATASET.md)
- [ML pipeline and evaluation](docs/ML_PIPELINE.md)
- [Nutrition methodology](docs/NUTRITION.md)
- [Android release guide](docs/ANDROID_RELEASE.md)
- [Published v1.1.0 release notes](docs/RELEASE_NOTES_v1.1.0.md)
- [v1.1.1 maintenance candidate notes](docs/RELEASE_NOTES_v1.1.1.md)
