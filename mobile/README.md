# Food Calorie Estimator mobile app

The Flutter client supports Camera/Gallery image acquisition, on-device
MobileNetV3 Small TFLite classification, local nutrition lookup, editable grams,
manual food entry, meal photos, a diary, a daily goal, and Today/Weekly dashboards.
The production model supports the 13 exact labels listed in the root README.

## Persistence

The app uses `sqflite` and a versioned SQLite database in app-private Application
Support storage. `SqliteDiaryRepository` implements the existing
`DiaryRepository` interface. The `meals` table stores the current `MealEntry`
fields, including nutrition snapshots, classifier confidence, UTC timestamp,
portion mode/grams, and the validated meal-photo reference. `goals` stores the
optional daily calorie goal; `app_metadata` stores the successful one-time
migration marker. A timestamp index supports date-based reads.

On first database access, the repository reads the existing
`food_calorie_diary.json` through `JsonDiaryRepository`, validates it with the
current Dart models, and copies meals and the goal in one SQLite transaction. The
marker is committed only with successful writes. The JSON file is preserved and
not re-imported on later launches. Photo files remain in the existing
`Application Support/meal_photos` directory, with safe photo references persisted
alongside meals.

Meals retain their nutrition values as a snapshot; the app does not recalculate
older records when catalog data changes. Stored timestamps are UTC and are grouped
by device-local calendar dates for Diary Today and the rolling seven-day Weekly
dashboard. The shared controller keeps Scan, Diary, Goal, and Dashboard in sync.

## Classification and nutrition

Images are preprocessed as RGB, resized with the shortest edge at 256, center
cropped to 224×224, and passed to the bundled
`assets/models/mobilenet_v3_small_phase7_v2.tflite`. The model returns one of 13
closed-set classes. The 0.55 cutoff is a low-confidence retry policy only; it does
not detect non-food or unsupported food. Confidence does not scale nutrition. The
catalog mirror at `assets/nutrition/foods.json` is checked against the canonical
catalog in `../nutrition/data/foods.json`. User-selected grams scale the reference
nutrition values; portion weight is not inferred from pixels.

Manual entry selects one of the same supported foods directly from the catalog and
does not run the classifier. Classified and manual entries use the same meal
snapshot and SQLite persistence flow. Selected photos are copied into app-private
storage; meals without photos remain supported.

## Run and verify

~~~bash
flutter pub get
flutter run
flutter analyze
flutter test
~~~

Physical iPhone verification is complete, including legacy JSON migration, meals,
goal, photos, add/edit/remove flows, both dashboards, Camera/Gallery classification,
editable grams, manual entry, force-close/relaunch persistence, and offline use.
Android physical-device validation has not been performed. The current iOS runtime
logs duplicate Objective-C class warnings from LiteRT frameworks during startup;
no workaround has been applied.
