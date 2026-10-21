# Food Calorie Estimator mobile client

This Flutter client targets iOS and Android from one codebase. It currently
contains the Phase 7.3 mobile UI foundation and Phase 7.4 camera/gallery image
acquisition: Scan, Diary, Dashboard, and Goal screens with working bottom
navigation. Captured or selected meal photos can be previewed on the Scan
screen. No user data is persisted yet.

On-device TFLite food classification and inference, nutrition estimates and
editable portions, diary persistence, and daily/weekly dashboard data and
charts remain future integrations.

Physical iPhone testing was completed successfully. The app was built,
installed, and launched on the device. Take Photo and Choose from Gallery were
manually verified, and both captured and selected images displayed correctly
in the preview. Scan, Diary, Dashboard, and Goal navigation was also manually
verified.

Run locally from this directory with `flutter run`. Verify the client with
`flutter analyze` and `flutter test`.
