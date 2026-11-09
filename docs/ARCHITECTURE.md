# Architecture

## Production data flow

```text
Camera / gallery image
        ↓
EXIF-corrected RGB → resize shortest edge 256 → center crop 224 × 224
        ↓
14-output MobileNetV3 Small TensorFlow Lite model (on device)
        ↓
argmax: non_food rejection OR supported-food prediction
        ↓                                  ↓
reject before nutrition       user confirmation / manual selection
                                           ↓
                                exact local nutrition lookup
                                           ↓
                    default serving or user-entered grams
                                           ↓
                 SQLite meal snapshot + app-private photo reference
                                           ↓
                      Diary, Today, Weekly dashboard, goals
```

## Mobile application

The Flutter client separates image acquisition, classification, nutrition lookup,
diary state, and persistence. Camera and gallery images are passed to the bundled
TFLite model through `flutter_litert`. The model's ordered labels are bundled at
`mobile/assets/models/labels_phase9_5_14class.txt`:

```text
pizza, hamburger, sushi, steak, ice_cream, donuts, omelette,
french_fries, chicken_wings, spaghetti_bolognese, borek, kebap, sarma, non_food
```

Its input is float32 NHWC `[1,224,224,3]` RGB in `[0,255]`. The Phase 9.5
preprocessing contract resizes the shortest edge to 256 with bilinear interpolation,
then center-crops to 224×224; EXIF orientation is applied before sizing. The model
graph internally rescales pixels to `[-1,1]`. The frozen decision policy is argmax
with no confidence threshold. Confidence shown in the UI is informational.

Index 13 (`non_food`) is rejected before nutrition lookup and diary saving. For
indices 0–12, the predicted food and confidence are shown, but nutrition lookup is
locked until the user confirms. On decline, users can retry with Camera/Gallery or
manually choose and confirm a supported class. “My food isn't listed” is a safe
decline path and does not force an incorrect food choice. This user confirmation
does not make the model a reliable unsupported-food detector. The previous
13-class TFLite asset remains bundled at
`mobile/assets/models/mobilenet_v3_small_phase7_v2.tflite` for rollback.

The nutrition catalog is a bundled mirror of `nutrition/data/foods.json`. Exact food
IDs select catalog entries; selected grams scale per-100 g nutrient values. Manual
entry uses the same catalog without invoking image classification.

## SQLite and offline behavior

`DiaryController` coordinates scan results, meals, goals, and dashboard summaries
through a `DiaryRepository` boundary. Production uses `SqliteDiaryRepository` with
a versioned database managed by `sqflite`:

- `meals` stores stable IDs, food IDs and display names, UTC timestamps, portion
  mode and grams, saved calories/macros, optional confidence, photo reference, and
  ordering.
- `goals` stores the daily calorie goal.
- `app_metadata` records successful migration state.
- An index on `logged_at_utc` supports date-based reads.

Nutrition is saved as a meal snapshot, so catalog edits do not rewrite earlier
entries. Photo files are stored in app-private Application Support storage. Today
and seven-day Weekly summaries are derived from persisted meals and grouped by
device-local calendar dates. Weekly charts show daily calories, protein,
carbohydrates, and fat. Estimated macro-energy distribution applies 4 kcal/g to
protein and carbohydrates and 9 kcal/g to fat, then normalizes those three macro
contributions to 100%. It does not represent a share of total logged calories. No
micronutrient chart is implemented.

When a legacy `food_calorie_diary.json` is present, its contents are validated and
copied into SQLite in one transaction. The migration marker is written only after
successful database writes, and the source file is retained. JSON is a one-time
backward-compatibility path, not current production persistence.

SQLite and photo files are stored in app-private Application Support storage and are
not application-level encrypted. Photo files are copied without metadata stripping
and may retain source EXIF metadata. The app scan/log flow contains no network
upload, and the Android manifest does not request Internet permission. The v1.1.1
candidate excludes app data from Android-managed backup and device-transfer rules;
Android device-transfer behavior can vary by manufacturer. Local data can be lost
on uninstall or device change.

## Production model evaluation

The Phase 9.5 float32 TFLite export was checked against the selected 14-class Keras
checkpoint and matched its frozen held-out metrics. The TFLite candidate recorded
90.33% supported-food classification accuracy, 99.63% supported-food acceptance,
94.33% non-food rejection, and 2.03% unsupported-food rejection. These results are
distinct from the previous 13-class TFLite result (88.12% on 2,719 supported-food
images) and the 13-class Keras result (89.41%). See [`ML_PIPELINE.md`](ML_PIPELINE.md)
for cohorts, interpretation, and limitations.

## Nutrition boundary

The nutrition layer uses exact class IDs and a versioned catalog. Per-100 g
calories, protein, carbohydrates, and fat are scaled by the selected or default
portion. Catalog provenance and calculation assumptions are described in
[`NUTRITION.md`](NUTRITION.md). The model and nutrition layer do not infer portion
mass or exact recipe composition from pixels.

## Python research and domain modules

The repository also contains dataset/model tooling and standalone Python
nutrition/meal-domain services. `app/storage.py` is a JSON adapter for those Python
services only; Flutter production persists to SQLite. Historical PyTorch
EfficientNet-B0 models remain for research context and are not used for production
mobile inference.
