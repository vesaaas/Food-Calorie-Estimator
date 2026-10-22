# Food Calorie Estimator mobile client

This Flutter client targets iOS and Android from one codebase. It currently
contains the Phase 7.3 mobile UI foundation, Phase 7.4 camera/gallery image
acquisition, Phase 7.5 on-device TFLite food classification, Phase 7.6 editable
nutrition portions, Phase 7.7 local diary and calorie goal, and the Phase 7.8
Today/Weekly dashboard. Captured or
selected meal photos are previewed and classified on the Scan screen. The
MobileNetV3 Small deployment model is bundled at
`assets/models/mobilenet_v3_small_phase7_v2.tflite`; preprocessing follows the
ML deployment contract (RGB, resize shortest edge to 256, center crop to 224,
float32 values in [0,255]). Confidence is the model's softmax score, not a
nutrition estimate. Nutrition uses the verified catalog mirrored from
`../nutrition/data/foods.json` at `assets/nutrition/foods.json`. The Scan screen
shows the catalog's default serving and lets the user edit its portion in grams;
calories and macros are recalculated from the per-100 g values. Portion weight is
not estimated from the image, and classifier confidence does not affect nutrition.
The Python nutrition verifier checks that the bundled catalog matches its
canonical source. Phase 7.7 stores meal nutrition snapshots and the optional
positive daily calorie goal together in a versioned JSON file in the app-private
Application Support directory. The JSON repository validates the saved document,
uses a same-directory temporary file and replacement rename, and is accessed through
the injectable shared `DiaryController`. Meals retain the food ID, display name,
time, portion mode/grams, calculated calories/macros, and optional confidence at
the time they are added. Historical meals are not recalculated from the catalog.
The mobile confidence gate accepts predictions at or above 0.55; lower scores show
an identification retry message and do not load nutrition or allow Add Meal. This
is a low-confidence rule for a closed-set classifier, not a food-versus-non-food
detector. The supported classes are pizza, hamburger, sushi, steak, ice cream,
donuts, omelette, french fries, chicken wings, spaghetti bolognese, borek, kebap,
and sarma. Unsupported foods and non-food objects can occasionally receive
high-confidence predictions; during physical testing, a laptop was classified as
Donuts at about 64%, and a cake was also classified as Donuts above 50%. The Scan
screen explains the supported set before scanning and provides a list of classes.
A successful Add Meal copies the selected photo into the app-private
`Application Support/meal_photos` directory and stores only its validated opaque
file reference in the diary JSON. Existing entries without a photo remain valid.
Users can also choose Add Manually to select one of the same 13 supported foods;
this uses the canonical nutrition catalog without running the classifier and
saves no confidence or photo. Diary rows show a small saved-photo thumbnail or a
neutral placeholder for meals without an available photo.
The Scan form resets after a successful save; failed saves preserve the current
result. Diary rows open a detail sheet, and removing a meal removes its photo when
the app-owned file can be safely deleted; orphan files are pruned after a later
successful diary load.
The Diary shows only meals on the device-local current calendar date; timestamps are
stored in UTC. Removing a meal preserves the other meals and goal. Scan, Diary, and
Goal share the same controller and repository state. Image weight is not inferred,
and confidence remains metadata only.

The Dashboard is connected to the shared diary controller. Today shows saved
calorie and macro totals, meal count, and the optional goal/progress state. Weekly
shows a rolling seven-local-day calorie bar chart, total, seven-day average, and
meal count. Empty dates are zero, weekly average includes all seven dates, and
UTC-stored meal timestamps are grouped by device-local calendar date. Dashboard
values use saved meal snapshots and update when meals or the goal change; they do
not recalculate from the catalog or use confidence/photo metadata.
`flutter analyze`, `flutter test`, Python nutrition parity verification, and the
Phase 6 regression verifier cover the current implementation.

Physical iPhone verification passed: the app built, installed, and launched;
Gallery inference predicted Pizza at 59.7% confidence, and Camera inference
predicted French Fries at 99.9% confidence. The `flutter_litert` runtime also
reported duplicate Objective-C classes between `LiteRt.framework` and
`LiteRtMetalAccelerator.framework` while the tested build, launch, and on-device
inference completed successfully. No framework workaround was applied, and the
warning has not been confirmed harmless; it remains a release/runtime
consideration.

Phase 7.6 nutrition and editable-portion verification also passed on the iPhone.
French Fries loaded its canonical 110 g default serving and displayed 317.9 kcal,
3.8 g protein, 40.9 g carbohydrates, and 15.4 g fat. Editing the portion to
200 g recalculated the estimate to 578.0 kcal, 7.0 g protein, 74.4 g
carbohydrates, and 28.0 g fat. The app did not estimate food weight from the
image.

Phase 7.4 camera/gallery image acquisition and Scan, Diary, Dashboard, and Goal
navigation were also manually verified.

Phase 8 validation passed on the physical iPhone: saved meal, goal, photo, and
Dashboard totals remained correct after force-quit and independent Home Screen
relaunch. Diary, Goal, Dashboard, nutrition data, saved photos, and Camera/Gallery
TFLite classification also worked with Airplane Mode enabled and Wi-Fi disabled.
Android build/runtime validation was not performed because no Android SDK,
emulator/toolchain, or physical Android device was available; Android runtime is
not claimed as tested. The model remains a closed-set classifier over the 13 foods
listed above: its 0.55 cutoff rejects low-confidence predictions but does not detect
non-food images. Unsupported foods and non-food objects can still receive a
supported-food prediction at confidence >= 0.55.

Automated Phase 8 verification passed: `flutter analyze`, all 68 Flutter tests,
17 nutrition verification tests, 13 Phase 6 regression tests, the MobileNetV3/
TFLite verifier, and the confidence evaluator self-test. The lifecycle-resume
date refresh is covered by four focused tests; classifier preprocessing and output
mapping are covered by six focused tests.

Run locally from this directory with `flutter run`. Verify the client with
`flutter analyze` and `flutter test`.
