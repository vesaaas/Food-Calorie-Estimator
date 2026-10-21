# Food Calorie Estimator mobile client

This Flutter client targets iOS and Android from one codebase. It currently
contains the Phase 7.3 mobile UI foundation: Scan, Diary, Dashboard, and Goal
screens with working bottom navigation. Photo selection and goal saving are
placeholders; no user data is persisted yet.

Future phases will connect camera and gallery input, on-device TFLite food
classification, nutrition estimates and editable portions, diary persistence,
and daily/weekly dashboard data and charts. These integrations are not part of
the current UI foundation.

Run locally from this directory with `flutter run`. Verify the client with
`flutter analyze` and `flutter test`.
