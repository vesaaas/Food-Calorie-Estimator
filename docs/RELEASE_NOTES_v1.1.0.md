# Food Calorie Estimator v1.1.0

Version: **1.1.0+2**. These notes describe the locally verified Android candidate;
they do not announce a published release.

## What's new

- Expanded the MobileNetV3 Small classifier to 13 supported food categories plus
  `non_food`, with explicit rejection of `non_food` predictions.
- Added confirmation before nutrition is shown, manual selection of a supported
  food, retry from camera/gallery, and a safe option to report that a food is not
  listed.
- Added seven-day protein, carbohydrate, and fat charts and an estimated
  macro-energy distribution based on 4/4/9 kcal per gram.
- Continued offline meal logging with SQLite-backed nutrition snapshots.

## Model limitations

The frozen argmax policy has no confidence acceptance threshold. Held-out TFLite
results were 90.33% supported-food classification accuracy, 99.63% supported-food
acceptance, 94.33% non-food rejection, and 2.03% unsupported-food rejection.
Unsupported foods can still receive supported-food predictions; the model is **not a
reliable unsupported-food detector**. Confirmation is a user safeguard, not a
detector.

## Android installation and verification

The original v1.0.0 APK was debug-signed; v1.1.0 uses a dedicated release signer.
Android cannot directly update a v1.0.0 installation with this signer. Uninstalling
the old app may delete its local diary and saved meal photos, so back up local data
before reinstalling.

The local candidate APK is
`mobile/build/app/outputs/flutter-apk/app-release.apk` (103,563,167 bytes), package
`com.example.food_calorie_estimator`, version name/code `1.1.0` / `2`, min SDK 24,
target SDK 36. Its SHA-256 is
`b71e0eb55b2fa5e4bec0ce1ae7c3c237809f9540a97d603f84e5d07ca535c696`; its APK v2
signature was verified. The signer certificate fingerprint and verification detail
are in [`ANDROID_RELEASE.md`](ANDROID_RELEASE.md). The APK is a local ignored build
output and is not a published GitHub release. Final Android APK physical-device
testing remains pending.

## iOS status

A local iOS Release app product with version 1.1.0/build 2 was produced. Strict
local signature verification reported `CSSMERR_TP_NOT_TRUSTED`; App Store
distribution-signing readiness has not been established. Physical iOS Release-mode
testing remains pending.
