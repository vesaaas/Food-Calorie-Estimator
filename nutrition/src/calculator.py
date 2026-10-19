"""Portion scaling from per-100g references; no model confidence or image inputs."""

if __package__:
    from .catalog import NUTRIENTS, NutritionCatalog, finite_number, load_catalog
else:
    from catalog import NUTRIENTS, NutritionCatalog, finite_number, load_catalog


def scale_nutrients(per_100g: dict, grams: float) -> dict:
    """Pure calculation; retain precision and reject nonfinite results."""
    grams = finite_number(grams, "grams", positive=True)
    result = {}
    for name in NUTRIENTS:
        value = finite_number(per_100g.get(name), name)
        result[name] = finite_number(value * (grams / 100), f"calculated {name}")
    return result


def estimate(food_id: str, grams=None, servings=None, *, catalog: NutritionCatalog | None = None) -> dict:
    """Use a default serving, explicit edible grams, or a serving multiplier.

    Pass a loaded catalog to reuse it without file I/O. Returned data is independent
    of the catalog. No rounding occurs here; energy_kcal means dietary calories.
    """
    if grams is not None and servings is not None:
        raise ValueError("Supply either grams or servings, not both")
    catalog = load_catalog() if catalog is None else catalog
    food = catalog.lookup(food_id)
    default = food["default_serving"]
    if grams is not None:
        portion_grams = finite_number(grams, "grams", positive=True)
        mode = "grams"
    elif servings is not None:
        servings = finite_number(servings, "servings", positive=True)
        portion_grams = finite_number(default["grams"] * servings, "portion grams", positive=True)
        mode = "servings"
    else:
        portion_grams = float(default["grams"])
        mode = "default"
    return {
        "food_id": food_id, "display_name": food["display_name"],
        "reference_description": food["reference_description"],
        "mapping_notes": food["mapping_notes"], "catalog_version": catalog.catalog_version,
        "portion_grams": portion_grams, "portion_mode": mode,
        "default_serving": default,
        "nutrients": scale_nutrients(food["per_100g"], portion_grams),
        "nutrition_source": food["nutrition_source"],
    }
