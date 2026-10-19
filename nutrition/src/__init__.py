"""Local USDA nutrition lookup and portion calculation, with no ML dependencies."""

from .catalog import NutritionCatalog, load_catalog
from .calculator import estimate, scale_nutrients

__all__ = ["NutritionCatalog", "load_catalog", "estimate", "scale_nutrients"]
