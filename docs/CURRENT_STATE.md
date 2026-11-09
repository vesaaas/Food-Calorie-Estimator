# Current Production State

The current Flutter application provides on-device food classification, nutrition
estimation, local meal logging, and daily/weekly summaries. Core workflows operate
offline. The public Android release is v1.1.0 (`1.1.0+2`). The v1.1.1 (`1.1.1+3`)
maintenance candidate adds explicit Android backup exclusions and documentation
updates. An isolated release build and package, signer, and backup-manifest checks
passed; physical-device verification remains pending.

## Mobile model and decision policy

The current app uses the Phase 9.5 float32 MobileNetV3 Small TFLite candidate with
14 outputs. Ordered labels are `pizza`, `hamburger`, `sushi`, `steak`, `ice_cream`,
`donuts`, `omelette`, `french_fries`, `chicken_wings`, `spaghetti_bolognese`,
`borek`, `kebap`, `sarma`, `non_food`. The existing 13-class model asset remains
bundled for rollback.

Input is EXIF-corrected RGB, resized with shortest edge 256 using bilinear
interpolation, then center-cropped to 224×224. The model receives float32 NHWC
`[1,224,224,3]` values in `[0,255]`; the model graph internally rescales pixels to
`[-1,1]`. The frozen decision policy is argmax across all 14 outputs, with indices
0–12 for supported foods and index 13 for `non_food`. There is no confidence
acceptance threshold; confidence is informational only.

The app rejects `non_food` before nutrition lookup or diary saving. For supported
predictions, it requires explicit user confirmation before nutrition appears or
saving is enabled. Users can retry with Camera/Gallery or manually select and
confirm a supported category. They may decline with “My food isn't listed.” The
classifier detects many non-food objects but is not a reliable detector of
unsupported foods.

## Held-out evaluation

The 14-class TFLite candidate recorded **90.33% supported-food classification
accuracy**, **99.63% supported-food acceptance**, **94.33% non-food rejection**, and
**2.03% unsupported-food rejection**. Acceptance and classification accuracy are
different: an accepted prediction may still be assigned to the wrong supported
class. Phase 9.5's float32 export matched the selected 14-class Keras checkpoint on
the frozen held-out metrics and passed representative parity checks.

These results are separate from the previous 13-class TFLite model's 88.12% accuracy
on 2,719 supported-food images and the 13-class Keras result of 89.41%. Those are
different model/runtime baselines. The 14-class model is not a reliable detector of
unsupported foods, which may still receive high-confidence supported-class labels.
See [`ML_PIPELINE.md`](ML_PIPELINE.md) for evaluation and limitations.

## Application and persistence

The Flutter app supports camera/gallery inference, confirmation and manual food
selection, local nutrition lookup, editable gram portions, meal add/edit/remove,
saved meal photos, a daily calorie goal, and Today/Weekly dashboards. `sqflite`
stores the production diary in a versioned app-private SQLite database. Meals store
nutrition snapshots and photo references; dashboard data is derived from persisted
meals. Weekly charts show calories and daily protein, carbohydrates, and fat. The
macro-energy distribution estimates energy from those three logged macros using
4/4/9 kcal per gram and normalizes their sum to 100%; it is not a share of total
logged calories. Micronutrient charts are not present.

## Validation status

- Flutter analysis completed cleanly and the full Flutter suite passed **90 tests**.
- Phase 9.5 recorded Keras/TFLite parity checks for the frozen 14-class candidate.
- Physical Samsung Android testing of v1.1.0 completed successfully, covering
  recognition, nutrition, meal logging, and dashboard workflows. This is a device
  test record, separate from automated checks.
- `mobile/pubspec.yaml` declares the v1.1.1 maintenance candidate version `1.1.1+3`.
- The original v1.0.0 APK checksum, package, version, and signer were verified
  locally; v1.0.0 was signed with the Android debug identity. Starting with v1.1.0,
  release signing uses a dedicated keystore configured through ignored
  `mobile/android/key.properties`. That identity change prevents a direct update
  over v1.0.0. Removing the existing app may erase the local SQLite diary and saved
  photos; users need a backup before reinstalling.
- The published v1.1.0 APK at
  `mobile/build/app/outputs/flutter-apk/app-release.apk` is 103,563,167 bytes with
  SHA-256 `b71e0eb55b2fa5e4bec0ce1ae7c3c237809f9540a97d603f84e5d07ca535c696`.
  `aapt` verified package `com.example.food_calorie_estimator`, version name/code
  `1.1.0` / `2`, min SDK 24 and target SDK 36. `apksigner` verified the APK v2
  signature and certificate SHA-256
  `c9e973cf8899bd79bfc31d4c6021ba53a9d503841e0aa948eb332ed8fbe922c7`. The GitHub
  release is linked from the root README.
- Android release signing uses the dedicated configuration and does not fall back
  to debug signing. The v1.0.0 debug signer differs, so a direct update is not
  possible; uninstalling may delete the local diary and meal photos. Preserve a
  backup before migration. Do not put the release keystore or local signing
  properties under version control.
- A local iOS Release app product reports version 1.1.0/build 2, but strict
  `codesign` verification returned `CSSMERR_TP_NOT_TRUSTED`; this is not evidence
  of App Store distribution-signing readiness. iOS Release-device testing remains
  pending.
- The current Flutter Gradle plugin resolves `flutter.minSdkVersion` to API 24,
  which supports the original APK's v2 signing scheme.
- Final v1.1.0 Android device testing and publication are complete. The v1.1.1
  maintenance candidate build and static artifact checks are complete; final
  physical-device checks and publication remain pending.
  iOS Release-device testing and App Store distribution signing remain pending.

## Android backup and data retention

Starting with the v1.1.1 candidate, the Android manifest disables Auto Backup and
references explicit legacy and Android 12+ extraction rules that exclude app data
from cloud backup and device transfer. This includes the app-private SQLite diary
and saved meal photos. No stored records or storage formats were changed. Android
backup behavior can vary by device manufacturer; this policy is not a user-facing
export feature. Local diary data and photos may be lost when uninstalling or
changing devices, so retain a separate copy of anything that must be kept.
The v1.1.0 manifest did not define backup exclusions, so Android's default may have
included app data when backup was enabled. The new policy does not delete copies
already held by an Android backup provider.

See [`ANDROID_RELEASE.md`](ANDROID_RELEASE.md) for signing setup, upgrade
compatibility, and the local diary data implications.

## Limitations

The `non_food` output is not a general out-of-distribution detector. Unsupported
foods can be classified as supported foods, even with high confidence. User
confirmation reduces accidental acceptance but does not detect unsupported foods.
Portion weight is user-entered, and catalog nutrition values are representative
estimates. See [`NUTRITION.md`](NUTRITION.md).

The Python domain services retain a JSON repository adapter for standalone
compatibility and verification. The Flutter app uses SQLite; JSON is not its
production storage layer.
