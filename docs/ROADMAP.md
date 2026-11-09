# Engineering Roadmap

The project has completed its core data, mobile, persistence, and reviewer-requested
Phase 9 implementation work. Android v1.1.0 is published and passed functional
testing on a physical Samsung Android device. A v1.1.1 maintenance candidate is
being prepared; its isolated release build is verified and its physical-device
verification remains pending.

| Phase | Scope | Status |
| --- | --- | --- |
| Phase 1 | Project setup and dataset preparation | COMPLETE |
| Phase 2 | Food classification model development and evaluation | COMPLETE |
| Phase 3 | USDA nutrition and portion calculations | COMPLETE |
| Phase 4 | Local food integration | COMPLETE |
| Phase 5 | Meal logging, calorie goals, and dashboards | COMPLETE |
| Phase 6 | Flutter mobile application and on-device TFLite inference | COMPLETE |
| Phase 7 | SQLite persistence and offline functionality | COMPLETE |
| Phase 8 | Testing, validation, and project documentation | COMPLETE |
| Phase 9 | Weekly macro dashboard, 14-class non-food handling, parity, and user confirmation | COMPLETE |
| Phase 9.8 | v1.1.0 release and post-release documentation | COMPLETE |
| Maintenance | v1.1.1 Android backup hardening and release preparation | IN PROGRESS |

## Phase 9 results and limitations

The 14-class model distinguishes the 13 supported food labels from a `non_food`
label and rejects `non_food` before nutrition lookup or meal logging. Supported
predictions require explicit user confirmation; users may retry or manually choose
one of the supported foods, and can decline when the food is not listed. The model
is not a reliable unsupported-food detector. Held-out unsupported-food rejection
was 2.03%; unsupported foods can still be assigned to supported classes.

The Weekly dashboard includes daily calories, protein, carbohydrates, and fat, plus
an estimated macro-energy distribution using 4/4/9 kcal per gram. The distribution
is normalized over logged macro contributions, not total logged calories. No
micronutrient charts are implemented.

Physical Samsung Android testing of v1.1.0 completed successfully, covering
recognition, nutrition, meal logging, and dashboard workflows. This is a device
test record, separate from automated checks. The previously recorded Flutter
analysis was clean and the full suite had 90 passing tests.

## Release preparation in progress

- The public v1.1.0 release is `1.1.0+2`; the maintenance candidate version is
  `1.1.1+3`. The v1.0.0 APK and its original release metadata remain unchanged.
- v1.1.0 intentionally begins a dedicated release signing identity. Existing
  v1.0.0 installations cannot update directly; uninstalling can erase the local
  SQLite diary and saved photos. Communicate the need to back up local data before
  installing the new release.
- The published v1.1.0 Android APK at `mobile/build/app/outputs/flutter-apk/app-release.apk` is
  103,563,167 bytes; its SHA-256, package/version/SDK metadata and v2 signature have
  verified. The release is published; its signer certificate fingerprint is in
  [`ANDROID_RELEASE.md`](ANDROID_RELEASE.md).
- Final v1.1.0 APK functional testing passed on a physical Samsung Android phone.
- A local iOS Release product exists, but strict signature verification returned
  `CSSMERR_TP_NOT_TRUSTED`; distribution signing and iOS Release-device testing are
  pending.
- Keep a secure independent backup of the configured release keystore and its
  credentials. Do not store them in the repository.
- v1.1.1 adds explicit Android cloud-backup and device-transfer exclusions for
  local app data. Its isolated release build and static artifact checks passed;
  final physical-device checks and publication remain pending.
- The v1.1.1 candidate retains the v1.1.0 release signing identity; v1.0.0 remains
  incompatible for in-place upgrades because it was debug-signed.

Broader Android device coverage remains future work. Image-based portion estimation
also remains separate future work; current portion weights are user-entered.
