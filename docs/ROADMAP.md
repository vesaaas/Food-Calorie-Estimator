# Engineering Roadmap

The project has completed its core data, mobile, persistence, and reviewer-requested
Phase 9 implementation work. The local v1.1.0 Android APK is verified; final
physical-device checks and publication remain pending.

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
| Phase 9.8 | Final documentation and release preparation | IN PROGRESS |

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

Phase 9 physical Android checks were reported successful by the user. This is
user-reported verification rather than an automated or independently reproduced
check. Flutter analysis is clean and the full suite has 90 passing tests.

## Release preparation in progress

- The v1.0.0 APK package, version, checksum, and debug signing certificate have
  been verified. The planned app version is `1.1.0+2`.
- v1.1.0 intentionally begins a dedicated release signing identity. Existing
  v1.0.0 installations cannot update directly; uninstalling can erase the local
  SQLite diary and saved photos. Communicate the need to back up local data before
  installing the new release.
- The local Android APK at `mobile/build/app/outputs/flutter-apk/app-release.apk` is
  103,563,167 bytes; its SHA-256, package/version/SDK metadata and v2 signature have
  been verified. The signer certificate fingerprint is in
  [`ANDROID_RELEASE.md`](ANDROID_RELEASE.md). This APK is not yet published.
- Final v1.1.0 APK physical-device testing remains pending. Earlier Phase 9 checks
  were user-reported and do not substitute for testing this release artifact.
- A local iOS Release product exists, but strict signature verification returned
  `CSSMERR_TP_NOT_TRUSTED`; distribution signing and iOS Release-device testing are
  pending.
- Keep a secure independent backup of the configured release keystore and its
  credentials. Do not store them in the repository.
- Build and publish only after those checks are complete and approved.

Broader Android device coverage remains future work. Image-based portion estimation
also remains separate future work; current portion weights are user-entered.
