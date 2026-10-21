# Nutrition and Calorie Data Layer

## Purpose and boundary

The local catalog maps each exact classifier label to a sourced representative
nutrition record, then scales it for a default serving or a portion the user selects.
It does not load the image model and has no PyTorch dependency.

The class identifiers must match the current classifier labels exactly:
`pizza`, `hamburger`, `sushi`, `steak`, `ice_cream`, `donuts`, `omelette`,
`french_fries`, `chicken_wings`, `spaghetti_bolognese`, `borek`, `kebap`, and `sarma`. Lookup uses the string
identifier, never the order of catalog entries or a guessed class index. Coverage
verification reads the existing class tuple from `ml/src/dataset.py` without
importing or executing the model code.

## USDA records and serving references

`nutrition/data/foods.json` is schema version 1, catalog version 1.1.0. Nutrients
are the USDA values per 100 g: energy in kcal and protein, carbohydrates, and fat
in grams. Every entry includes its exact FDC record description and ID, data type,
release, official record URL, retrieval date, default portion label and gram weight,
and the USDA portion row ID. `nutrition/data/usda_source_extracts.json` preserves
the selected four nutrient rows and one portion row per food from the official
USDA archives, plus archive URLs and SHA-256 hashes, for offline auditing.

| Class | FDC ID | USDA reference | Data type | Default serving |
| --- | ---: | --- | --- | ---: |
| `pizza` | 2708617 | Pizza, cheese, from restaurant or fast food, thick crust | FNDDS 2021–2023 | 1 piece, large pizza — 132 g |
| `hamburger` | 2706927 | Hamburger, on white bun, 1 small patty | FNDDS 2021–2023 | 1 hamburger — 115 g |
| `sushi` | 2708959 | Sushi, NFS | FNDDS 2021–2023 | 1 piece — 30 g |
| `steak` | 2705832 | Beef, steak, sirloin, NS as to fat eaten | FNDDS 2021–2023 | 1 regular — 160 g |
| `ice_cream` | 2705630 | Ice cream, vanilla | FNDDS 2021–2023 | 1 scoop — 120 g |
| `donuts` | 172758 | Doughnuts, yeast-leavened, glazed, enriched (includes honey buns) | SR Legacy | 1 doughnut — 63 g |
| `omelette` | 2707205 | Egg omelet or scrambled egg, no added fat | FNDDS 2021–2023 | 1 egg — 50 g |
| `french_fries` | 2709462 | Potato, french fries, restaurant | FNDDS 2021–2023 | 1 small fast food order — 110 g |
| `chicken_wings` | 2706057 | Chicken wing, baked, broiled, or roasted, from raw | FNDDS 2021–2023 | 1 wing, any size — 35 g |
| `spaghetti_bolognese` | 167680 | Restaurant, Italian, spaghetti with meat sauce | SR Legacy | 1 serving — 554 g |
| `borek` | TürKomp 06.02.0040 | Börek, peynirli, dondurulmuş (cheese-filled, frozen) | TürKomp | 100 g reference portion |
| `kebap` | 2706780 | Lamb shish kabob with vegetables, excluding potatoes | FNDDS 2021–2023 | 1 shishkabob — 202 g |
| `sarma` | 2706621 | Stuffed grape leaves with beef and rice | FNDDS 2021–2023 | 1 roll — 21 g |

The records were checked against USDA's official [FDC downloadable datasets](https://fdc.nal.usda.gov/download-datasets/): FNDDS 2021–2023 (published October 2024) and SR Legacy (April 2018, its final release). The local extract records the exact values and portion IDs used. FDC portion gram weights refer to edible material; for a bone-in wing, that means the portion weight is not the whole wing including refuse/bone. The named serving is a reference quantity, not a claim about the photographed food's weight.

The hamburger record was explicitly selected as a representative proxy for the classifier label. It means a white bun with one small patty, without cheese or added toppings; it is not a generic value for all burgers. The pizza reference is thick-crust cheese pizza, sushi is not further specified (NFS), steak is sirloin with fat eaten unspecified, and the doughnut record also includes honey buns. The remaining record-specific assumptions are in each catalog entry's `mapping_notes`.

Phase 5 adds three regional representative records. `kebap` uses USDA FNDDS record
2706780 for lamb shish kebab with vegetables; it is a defensible kebab example, not
an exact match for every kebap style. `sarma` uses USDA FNDDS record 2706621 for
beef-and-rice stuffed grape leaves; vegetarian and other fillings differ. Börek uses
TürKomp record 06.02.0040, a frozen cheese-filled börek. TürKomp does not publish a
household portion in the checked record, so its default is a transparent 100 g
reference portion. The source URL and retrieval record are kept in `foods.json` and
the selected evidence is in `phase5_source_extracts.json`.

The added per-100 g values are: börek 288 kcal, 6.73 g protein, 32.34 g
carbohydrates, 13.87 g fat; kebap 162 kcal, 12.0 g protein, 5.06 g carbohydrates,
10.3 g fat; sarma 228 kcal, 8.15 g protein, 11.3 g carbohydrates, 17.0 g fat.

## Portion calculation

For the default, the calculator uses that food's USDA household portion weight. A caller may instead pass an explicit edible weight in grams or a multiplier of the default serving. Grams and servings cannot be supplied together.

For each nutrient:

~~~text
portion amount = per_100g amount × portion grams / 100
~~~

For example, the pizza reference contains 271 kcal per 100 g. One default 132 g piece is estimated as 271 × 132 / 100 = 357.72 kcal; an entered 180 g portion is 487.8 kcal. The calculator keeps full floating-point precision and the CLI rounds only for display. The calorie value comes from USDA energy; it is not recalculated from the macros.

## Use

From the repository root:

~~~bash
python nutrition/src/estimate.py pizza
python nutrition/src/estimate.py pizza --servings 1.5
python nutrition/src/estimate.py pizza --grams 180
~~~

In Python, `nutrition.src.load_catalog(expected_classes=...)` loads and validates
the JSON, while `nutrition.src.estimate(food_id, grams=..., servings=...,
catalog=...)` returns the raw result and provenance. `expected_classes` can be the
classifier mapping's exact string keys. A custom catalog can be passed for repeated
lookups to avoid reading the file each time.

## Validation and limitations

The loader rejects malformed JSON, duplicate JSON keys and food IDs, missing
fields/nutrients, unsupported schema versions, invalid source references, and
negative or nonfinite nutrient values. Serving weights and requested portions must
be finite and positive. Catalog coverage validation reports missing or unexpected
class identifiers. Unknown or differently spelled labels fail clearly.

Food classification does **not** measure portion weight. The app can calculate for
the configured default serving or a portion the user chooses or enters. It does not
infer food mass from an image, and classifier confidence is not used to change
calories or macros. Confidence describes the classifier's class prediction; the
nutrition estimate is based on the reference food and portion assumption.

Broad classes cover different recipes, cuts, toppings, fillings, and cooking methods.
The mapped USDA item is therefore a representative nutrition estimate, not a
laboratory measurement of the photographed meal. One label also cannot describe a
mixed plate. USDA source portions vary by record and are references. The Phase 4
nutrition layer itself did not implement an API, mobile interface, food log, calorie
goal, dashboard, or local food class expansion. The Phase 5 image-to-nutrition
pipeline keeps classifier confidence informational: it never multiplies nutrition
by confidence and never estimates portion weight from the image. It uses the
configured default serving until a caller selects a portion through the Phase 6
domain services. The Phase 5
classifier has no unknown/out-of-distribution class, so unrelated images can still
be assigned to a supported label; confidence does not prove the food is supported.

## Phase 6 meal and dashboard services

Phase 6 reuses `nutrition/src/calculator.py` for all default, serving-multiplier, and
explicit-gram calculations. `app/meal_log.py` stores the selected portion grams and
the resulting calories/macros in a meal entry. Optional classifier confidence is
preserved only as prediction metadata and never changes nutrition or portion size.
The daily and weekly domain summaries aggregate persisted meal results; they do not
recalculate nutrition independently. UTC is the default calendar timezone, and a
weekly summary covers the inclusive seven days ending on its requested date. Phase 6
does not provide a mobile interface or infer weight from pixels. Those UI and
camera/gallery tasks belong to Phase 7.
