# Food Calorie Estimator mobile app

The Flutter client provides offline camera/gallery classification, user-confirmed
nutrition estimates, editable portions, manual selection, meal logging, daily goals,
and Today/Weekly dashboards. Production persistence uses a versioned app-private
SQLite database.

## Run

From this directory:

```bash
flutter pub get
flutter run
```

## Model and scan flow

The current model is `assets/models/mobilenet_v3_small_phase9_5_14class.tflite` with
ordered labels in `assets/models/labels_phase9_5_14class.txt`:

```text
pizza, hamburger, sushi, steak, ice_cream, donuts, omelette,
french_fries, chicken_wings, spaghetti_bolognese, borek, kebap, sarma, non_food
```

The input contract is EXIF-corrected RGB, shortest edge resized to 256 with bilinear
interpolation, then center-cropped to 224×224. The model input is float32 NHWC
`[1,224,224,3]`, range `[0,255]`; normalization to `[-1,1]` is inside the model.
The decision is argmax across all 14 outputs, with no confidence threshold.
Confidence is displayed as informational metadata.

`non_food` results are rejected before nutrition lookup or diary saving. A supported
prediction requires the user to confirm the predicted food before nutrition is
loaded. If incorrect, the user can retry with Camera/Gallery or select a supported
food manually and confirm it. “My food isn't listed” lets the user decline safely;
unsupported foods are not forced into an incorrect category. The classifier is not a
reliable unsupported-food detector. The old 13-class model remains bundled at
`assets/models/mobilenet_v3_small_phase7_v2.tflite` for rollback.

## Nutrition

`assets/nutrition/foods.json` is the app's bundled mirror of the canonical catalog
at `../nutrition/data/foods.json`. Exact class IDs select a catalog entry. The user
can edit the default portion in grams; nutrients are scaled from per-100 g values.
Portion weight is not inferred from the image. Manual food entry selects from the
same catalog without running classification.

## Local persistence and dashboards

The `meals` table stores immutable nutrition snapshots, timestamps, portion details,
confidence metadata, and safe photo references. Today and rolling seven-day Weekly
views are derived from persisted meals and grouped by device-local calendar dates.
Weekly charts show calories, protein, carbohydrates, and fat per day. The estimated
macro-energy distribution applies 4 kcal/g to protein and carbohydrates and 9 kcal/g
to fat, then normalizes those macro contributions to 100%; it is not a percentage of
total logged calories. The app does not show micronutrient charts.

A one-time transaction migrates an existing `food_calorie_diary.json` file into
SQLite after validating records. The marker is saved only after writes succeed, and
the source is retained. JSON is not current production persistence.

## Verification and release state

```bash
flutter analyze
flutter test
```

The last recorded check completed cleanly with **90 passing Flutter tests**.
Physical Samsung Android testing of the published v1.1.0 APK completed
successfully, covering the main scan, nutrition, diary, and dashboard workflows.
The local APK's SHA-256, package/version metadata, and APK v2 signature were
verified before publication. See
[`../docs/RELEASE_NOTES_v1.1.0.md`](../docs/RELEASE_NOTES_v1.1.0.md).

The current maintenance candidate is `1.1.1+3`. It adds Android backup exclusions
for app-private data without changing the database or photo-storage format. An
isolated release build and package, signer, and backup-manifest checks passed;
physical-device verification remains pending.

An iOS Release app product was built locally with version 1.1.0 (build 2), but
strict local code-signature verification reported `CSSMERR_TP_NOT_TRUSTED`. This
does not establish App Store distribution-signing readiness. iOS Release-mode
physical-device testing also remains pending.

The v1.0.0 APK used the debug signing identity. A dedicated release key begins
with v1.1.0, so it cannot directly update v1.0.0. Removing the old installation
may erase the local diary and meal photos; back them up first. Android-managed
backups are excluded by the v1.1.1 candidate policy, so do not rely on device
transfer to preserve the diary. The app has no in-app export or migration feature.
See [`../docs/ANDROID_RELEASE.md`](../docs/ANDROID_RELEASE.md). The root README
preserves the v1.0.0 archival checksum and links to the published v1.1.0 release
evidence.
