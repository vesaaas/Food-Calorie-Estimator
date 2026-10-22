# Food Calorie Estimator mobile client

This Flutter client targets iOS and Android from one codebase. It currently
contains the Phase 7.3 mobile UI foundation, Phase 7.4 camera/gallery image
acquisition, and Phase 7.5 on-device TFLite food classification. Captured or
selected meal photos are previewed and classified on the Scan screen. The
MobileNetV3 Small deployment model is bundled at
`assets/models/mobilenet_v3_small_phase7_v2.tflite`; preprocessing follows the
ML deployment contract (RGB, resize shortest edge to 256, center crop to 224,
float32 values in [0,255]). Confidence is the model's softmax score, not a
nutrition estimate. No user data is persisted yet.

Nutrition estimates and editable portions, diary persistence, and daily/weekly
dashboard data and charts remain future integrations.

Physical iPhone verification passed: the app built, installed, and launched;
Gallery inference predicted Pizza at 59.7% confidence, and Camera inference
predicted French Fries at 99.9% confidence. The `flutter_litert` runtime also
reported duplicate Objective-C classes between `LiteRt.framework` and
`LiteRtMetalAccelerator.framework`. This remains a known release/runtime
consideration; it has not been locally worked around or confirmed harmless.

Phase 7.4 camera/gallery image acquisition and Scan, Diary, Dashboard, and Goal
navigation were also manually verified.

Run locally from this directory with `flutter run`. Verify the client with
`flutter analyze` and `flutter test`.
