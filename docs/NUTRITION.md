# Nutrition Data and Calculations

## Catalog and class mapping

The canonical catalog is `nutrition/data/foods.json`; the Flutter app bundles a
verified mirror at `mobile/assets/nutrition/foods.json`. Lookup uses exact class ID,
not catalog order or fuzzy matching. The 13 IDs match the production classifier:
`pizza`, `hamburger`, `sushi`, `steak`, `ice_cream`, `donuts`, `omelette`,
`french_fries`, `chicken_wings`, `spaghetti_bolognese`, `borek`, `kebap`, `sarma`.

Nutrients are recorded per 100 g: energy in kcal and protein, carbohydrates, and
fat in grams. Each entry includes source and mapping metadata, retrieval details,
and a default portion. The selected USDA nutrient and portion rows are preserved in
`nutrition/data/usda_source_extracts.json` with source archive URLs and hashes.
Regional source evidence is in `nutrition/data/phase5_source_extracts.json`.

| Class | Reference food | Source | Default portion |
| --- | --- | --- | ---: |
| `pizza` | Cheese pizza, thick crust | USDA FNDDS 2021–2023 | 132 g |
| `hamburger` | White bun, one small patty | USDA FNDDS 2021–2023 | 115 g |
| `sushi` | Sushi, not further specified | USDA FNDDS 2021–2023 | 30 g |
| `steak` | Sirloin steak | USDA FNDDS 2021–2023 | 160 g |
| `ice_cream` | Vanilla ice cream | USDA FNDDS 2021–2023 | 120 g |
| `donuts` | Glazed yeast doughnut | USDA SR Legacy | 63 g |
| `omelette` | Omelet or scrambled egg, no added fat | USDA FNDDS 2021–2023 | 50 g |
| `french_fries` | Restaurant french fries | USDA FNDDS 2021–2023 | 110 g |
| `chicken_wings` | Roasted/baked chicken wing | USDA FNDDS 2021–2023 | 35 g edible portion |
| `spaghetti_bolognese` | Restaurant spaghetti with meat sauce | USDA SR Legacy | 554 g |
| `borek` | Frozen cheese-filled börek | TürKomp | 100 g reference |
| `kebap` | Lamb shish kebab with vegetables | USDA FNDDS 2021–2023 | 202 g |
| `sarma` | Beef-and-rice stuffed grape leaves | USDA FNDDS 2021–2023 | 21 g |

The USDA references are from the official [FoodData Central downloadable
datasets](https://fdc.nal.usda.gov/download-datasets/). They are representative
mappings: for example, the hamburger entry excludes cheese and toppings, and the
sarma entry represents beef-and-rice filling. The catalog's `mapping_notes` record
specific assumptions. The Börek reference is a 100 g quantity because the checked
TürKomp record does not publish a household serving.

## Portion calculation

For each nutrient, the calculator applies:

```text
portion amount = per-100-g amount × portion grams / 100
```

A user can use the catalog default, enter grams, or select a multiplier of the
default serving. Grams and serving multipliers cannot be supplied together. For
example, the pizza reference has 271 kcal per 100 g: the 132 g default estimates
357.72 kcal, while 180 g estimates 487.8 kcal. The calorie value uses the source
energy field and is not recalculated from macronutrients. Full precision is retained
internally; display formatting may round values.

Classifier confidence does not scale nutrients. The image does not determine portion
weight; the user can edit grams in the app. Meal records store calculated values as
snapshots, so later catalog updates do not alter historical meals.

## Validation and use

The catalog loader validates schema, provenance, exact IDs, required nutrients, and
finite nonnegative values. Portion weights and requested amounts must be finite and
positive. Nutrition verification checks all 13 catalog entries and the parity of the
Flutter mirror.

```bash
python nutrition/src/estimate.py pizza
python nutrition/src/estimate.py pizza --servings 1.5
python nutrition/src/estimate.py pizza --grams 180
python nutrition/src/verify_nutrition.py
```

In Python, `nutrition.src.load_catalog()` loads the catalog and
`nutrition.src.estimate(food_id, grams=..., servings=..., catalog=...)` returns the
calculation with provenance.

## Limitations

Food categories include many recipes, toppings, cuts, fillings, and preparation
methods. A mapped reference is an estimate, not a measurement of the photographed
meal, and a single class cannot describe a mixed plate. Food classification does not
measure mass. See [`ML_PIPELINE.md`](ML_PIPELINE.md) for the classifier's closed-set
limitations and confidence policy.
