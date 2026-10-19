"""Estimate nutrition for a food class and a default or user-selected portion."""

import argparse

if __package__:
    from .calculator import estimate
else:
    from calculator import estimate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("food_id", help="Exact classifier label, e.g. pizza")
    portion = parser.add_mutually_exclusive_group()
    portion.add_argument("--grams", type=float, help="Edible portion weight in grams")
    portion.add_argument("--servings", type=float, help="Multiplier of the default serving")
    arguments = parser.parse_args()
    try:
        result = estimate(arguments.food_id, grams=arguments.grams, servings=arguments.servings)
    except ValueError as error:
        parser.exit(1, f"Nutrition estimate failed: {error}\n")
    nutrients = result["nutrients"]
    default = result["default_serving"]
    print(f"Food: {result['display_name']} ({result['food_id']})")
    print(f"Reference: {result['reference_description']}")
    print(f"Default serving: {default['label']} ({default['grams']:g} g)")
    print(f"Portion: {result['portion_grams']:g} g ({result['portion_mode']})")
    print(f"Calories: {nutrients['energy_kcal']:.2f} kcal")
    print(f"Protein: {nutrients['protein_g']:.2f} g")
    print(f"Carbohydrates: {nutrients['carbohydrates_g']:.2f} g")
    print(f"Fat: {nutrients['fat_g']:.2f} g")
    print(f"Source: USDA FoodData Central {result['nutrition_source']['fdc_id']}; catalog {result['catalog_version']}")
    print("Representative estimate; portion weight is selected, not measured from an image.")


if __name__ == "__main__":
    main()
