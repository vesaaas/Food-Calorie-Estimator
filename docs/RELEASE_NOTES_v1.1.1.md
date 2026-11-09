# Food Calorie Estimator v1.1.1 — Android maintenance candidate

Version: **1.1.1+3**. This document describes an unpublished, device-untested
candidate. Its isolated Android release build and package/version, signing, and
backup-manifest checks completed successfully. The published v1.1.0 release remains
unchanged.

## Changes

- Disable Android Auto Backup and add explicit exclusion rules for app-private data
  in legacy full backups and Android 12+ cloud backup and device transfer. The
  exclusions cover the SQLite diary, saved meal photos, and other app data.
- Add repository ignore protections for local environment files, signing material,
  database exports, and local logs.
- Correct documentation to distinguish published v1.1.0 facts from the v1.1.1
  candidate and to use the actual published v1.1.0 APK asset name, `app-release.apk`.

The changes do not modify the model, food labels, nutrition catalog, database
schema, or prediction behavior.

## Data retention

The app stores its diary and saved meal photos locally. The v1.1.1 Android policy
excludes app data from Android-managed cloud backup and device transfer. Behavior
can vary by device manufacturer. Uninstalling or changing devices may result in
loss of the local diary and photos; keep an independent copy of information that
must be retained. The v1.1.0 manifest did not define backup exclusions, so Android's
default behavior may have included app data when backups were enabled. The new
rules do not delete copies already held by an Android backup provider. No data
export workflow is added in this candidate.

## Model limitations

The model retains its 14-class MobileNetV3 Small TFLite output and frozen argmax
policy: 13 supported food classes followed by `non_food`. On the evaluated
held-out unsupported-food cohort, only **2.03%** were rejected as unsupported.
This is not reliable unsupported-food detection. The candidate does not change
model weights, prediction behavior, or the user confirmation flow.

## Upgrade and platform status

The published v1.1.0 Android release uses a dedicated release signer. The v1.1.1
candidate is configured to retain that signing identity and is intended to update
v1.1.0 installations. The debug-signed v1.0.0 installation cannot be updated
directly; uninstalling it may erase its locally stored diary and meal photos.

Android release build and static artifact verification are complete for the
isolated local candidate. Physical-device checks remain pending.
iOS Release-mode device testing and App Store distribution signing are also
pending; this candidate includes no iOS release claim.

## Isolated candidate build record

The local candidate was built from an isolated source copy; it is not a published
APK. It is 103,564,315 bytes, package `com.example.food_calorie_estimator`, version
name/code `1.1.1` / `3`, with minimum/target SDK 24 / 36. Its SHA-256 is
`09de5811df6db37647f6367d0f8edd02b971b628ecb53a43e5622e5527dffd77`. APK Signature
Scheme v2 verification passed, and its signer certificate matches the dedicated
v1.1.0 release signer. The compiled manifest and backup resources were inspected;
physical-device verification remains pending.
