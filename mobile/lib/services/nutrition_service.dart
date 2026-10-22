import 'dart:convert';

import 'package:flutter/services.dart';

import 'food_classification_service.dart' show foodClassNames;

const nutritionCatalogAssetPath = 'assets/nutrition/foods.json';

const _nutrientKeys = <String>[
  'energy_kcal',
  'protein_g',
  'carbohydrates_g',
  'fat_g',
];

abstract interface class FoodNutritionService {
  Future<NutritionEstimate> estimate(
    String foodId, {
    double? grams,
    double? servings,
  });
}

class NutritionFacts {
  const NutritionFacts({
    required this.energyKcal,
    required this.proteinGrams,
    required this.carbohydratesGrams,
    required this.fatGrams,
  });

  final double energyKcal;
  final double proteinGrams;
  final double carbohydratesGrams;
  final double fatGrams;

  factory NutritionFacts.fromJson(Map<String, dynamic> json, String field) {
    return NutritionFacts(
      energyKcal: _finiteNumber(json['energy_kcal'], '$field.energy_kcal'),
      proteinGrams: _finiteNumber(json['protein_g'], '$field.protein_g'),
      carbohydratesGrams: _finiteNumber(
        json['carbohydrates_g'],
        '$field.carbohydrates_g',
      ),
      fatGrams: _finiteNumber(json['fat_g'], '$field.fat_g'),
    );
  }

  NutritionFacts scaled(double factor) => NutritionFacts(
    energyKcal: energyKcal * factor,
    proteinGrams: proteinGrams * factor,
    carbohydratesGrams: carbohydratesGrams * factor,
    fatGrams: fatGrams * factor,
  );
}

class DefaultServing {
  const DefaultServing({
    required this.label,
    required this.grams,
    required this.basis,
    required this.sourcePortionId,
  });

  final String label;
  final double grams;
  final String basis;
  final int? sourcePortionId;

  factory DefaultServing.fromJson(Map<String, dynamic> json, String field) {
    final sourcePortionId = json['source_portion_id'];
    if (sourcePortionId != null &&
        (sourcePortionId is! int || sourcePortionId <= 0)) {
      throw FormatException(
        '$field.source_portion_id must be a positive integer or null.',
      );
    }
    return DefaultServing(
      label: _nonEmptyString(json['label'], '$field.label'),
      grams: _finiteNumber(json['grams'], '$field.grams', positive: true),
      basis: _nonEmptyString(json['basis'], '$field.basis'),
      sourcePortionId: sourcePortionId as int?,
    );
  }
}

class NutritionFood {
  const NutritionFood({
    required this.foodId,
    required this.displayName,
    required this.referenceDescription,
    required this.mappingNotes,
    required this.per100g,
    required this.defaultServing,
    required this.nutritionSource,
  });

  final String foodId;
  final String displayName;
  final String referenceDescription;
  final String mappingNotes;
  final NutritionFacts per100g;
  final DefaultServing defaultServing;
  final Map<String, dynamic> nutritionSource;

  factory NutritionFood.fromJson(Map<String, dynamic> json) {
    final foodId = _nonEmptyString(json['food_id'], 'food.food_id');
    if (!RegExp(r'^[a-z][a-z0-9_]*$').hasMatch(foodId)) {
      throw FormatException('Invalid canonical food_id: $foodId');
    }
    final per100g = _jsonObject(json['per_100g'], '$foodId.per_100g');
    _requireKeys(per100g, _nutrientKeys, '$foodId.per_100g');
    final serving = _jsonObject(
      json['default_serving'],
      '$foodId.default_serving',
    );
    final source = _jsonObject(
      json['nutrition_source'],
      '$foodId.nutrition_source',
    );
    for (final key in [
      'provider',
      'record_description',
      'url',
      'retrieved_on',
      'release',
    ]) {
      _nonEmptyString(source[key], '$foodId.nutrition_source.$key');
    }
    if (source['record_description'] != json['reference_description']) {
      throw FormatException(
        '$foodId source and reference descriptions differ.',
      );
    }
    final provider = source['provider'];
    if (provider == 'USDA FoodData Central') {
      if (serving['source_portion_id'] == null ||
          source['fdc_id'] is! int ||
          (source['fdc_id'] as int) <= 0) {
        throw FormatException('$foodId has invalid USDA source identifiers.');
      }
      _nonEmptyString(
        source['data_type'],
        '$foodId.nutrition_source.data_type',
      );
      _nonEmptyString(
        source['download_url'],
        '$foodId.nutrition_source.download_url',
      );
    } else if (provider == 'TürKomp Turkish Food Composition Database') {
      _nonEmptyString(
        source['data_type'],
        '$foodId.nutrition_source.data_type',
      );
      _nonEmptyString(
        source['source_record_id'],
        '$foodId.nutrition_source.source_record_id',
      );
    } else {
      throw FormatException('$foodId has an unsupported nutrition provider.');
    }
    return NutritionFood(
      foodId: foodId,
      displayName: _nonEmptyString(
        json['display_name'],
        '$foodId.display_name',
      ),
      referenceDescription: _nonEmptyString(
        json['reference_description'],
        '$foodId.reference_description',
      ),
      mappingNotes: _nonEmptyString(
        json['mapping_notes'],
        '$foodId.mapping_notes',
      ),
      per100g: NutritionFacts.fromJson(per100g, '$foodId.per_100g'),
      defaultServing: DefaultServing.fromJson(
        serving,
        '$foodId.default_serving',
      ),
      nutritionSource: Map.unmodifiable(source),
    );
  }
}

class NutritionEstimate {
  const NutritionEstimate({
    required this.food,
    required this.catalogVersion,
    required this.portionGrams,
    required this.portionMode,
    required this.nutrients,
  });

  final NutritionFood food;
  final String catalogVersion;
  final double portionGrams;
  final String portionMode;
  final NutritionFacts nutrients;
}

class NutritionCatalog {
  NutritionCatalog._({required this.catalogVersion, required this.foods});

  final String catalogVersion;
  final Map<String, NutritionFood> foods;

  factory NutritionCatalog.fromJsonString(String source) {
    final root = _jsonObject(jsonDecode(source), 'catalog');
    if (root['schema_version'] != 1) {
      throw const FormatException('Unsupported nutrition catalog schema.');
    }
    final catalogVersion = _nonEmptyString(
      root['catalog_version'],
      'catalog.catalog_version',
    );
    final rawFoods = root['foods'];
    if (rawFoods is! List || rawFoods.isEmpty) {
      throw const FormatException('catalog.foods must be a nonempty array.');
    }
    final foods = <String, NutritionFood>{};
    for (final rawFood in rawFoods) {
      final food = NutritionFood.fromJson(_jsonObject(rawFood, 'catalog.food'));
      if (foods.containsKey(food.foodId)) {
        throw FormatException('Duplicate food_id: ${food.foodId}');
      }
      foods[food.foodId] = food;
    }
    final expectedIds = foodClassNames.toSet();
    final actualIds = foods.keys.toSet();
    if (actualIds.length != expectedIds.length ||
        actualIds.difference(expectedIds).isNotEmpty ||
        expectedIds.difference(actualIds).isNotEmpty) {
      throw FormatException(
        'Nutrition catalog class coverage mismatch: '
        'missing=${expectedIds.difference(actualIds).toList()..sort()}, '
        'unexpected=${actualIds.difference(expectedIds).toList()..sort()}',
      );
    }
    return NutritionCatalog._(
      catalogVersion: catalogVersion,
      foods: Map.unmodifiable(foods),
    );
  }

  List<String> get foodIds => foods.keys.toList(growable: false);

  NutritionFood lookup(String foodId) {
    final food = foods[foodId];
    if (food == null) {
      throw ArgumentError.value(foodId, 'foodId', 'Unknown canonical food ID.');
    }
    return food;
  }

  NutritionEstimate estimate(String foodId, {double? grams, double? servings}) {
    if (grams != null && servings != null) {
      throw ArgumentError('Supply either grams or servings, not both.');
    }
    final food = lookup(foodId);
    late final double portionGrams;
    late final String portionMode;
    if (grams != null) {
      portionGrams = _finiteNumber(grams, 'grams', positive: true);
      portionMode = 'grams';
    } else if (servings != null) {
      final count = _finiteNumber(servings, 'servings', positive: true);
      portionGrams = _finiteNumber(
        food.defaultServing.grams * count,
        'portion grams',
        positive: true,
      );
      portionMode = 'servings';
    } else {
      portionGrams = food.defaultServing.grams;
      portionMode = 'default';
    }
    final nutrients = food.per100g.scaled(portionGrams / 100);
    for (final value in [
      nutrients.energyKcal,
      nutrients.proteinGrams,
      nutrients.carbohydratesGrams,
      nutrients.fatGrams,
    ]) {
      _finiteNumber(value, 'calculated nutrient');
    }
    return NutritionEstimate(
      food: food,
      catalogVersion: catalogVersion,
      portionGrams: portionGrams,
      portionMode: portionMode,
      nutrients: nutrients,
    );
  }
}

class AssetFoodNutritionService implements FoodNutritionService {
  AssetFoodNutritionService({
    AssetBundle? bundle,
    this.assetPath = nutritionCatalogAssetPath,
  }) : _bundle = bundle ?? rootBundle;

  final AssetBundle _bundle;
  final String assetPath;
  Future<NutritionCatalog>? _catalog;

  Future<NutritionCatalog> _loadCatalog() => _catalog ??= _bundle
      .loadString(assetPath)
      .then(NutritionCatalog.fromJsonString);

  @override
  Future<NutritionEstimate> estimate(
    String foodId, {
    double? grams,
    double? servings,
  }) async =>
      (await _loadCatalog()).estimate(foodId, grams: grams, servings: servings);
}

Map<String, dynamic> _jsonObject(Object? value, String field) {
  if (value is! Map<String, dynamic>) {
    throw FormatException('$field must be an object.');
  }
  return value;
}

String _nonEmptyString(Object? value, String field) {
  if (value is! String || value.trim().isEmpty) {
    throw FormatException('$field must be a nonempty string.');
  }
  return value;
}

double _finiteNumber(Object? value, String field, {bool positive = false}) {
  if (value is! num || !value.isFinite || (positive ? value <= 0 : value < 0)) {
    final bound = positive ? 'positive' : 'nonnegative';
    throw FormatException('$field must be a finite $bound number.');
  }
  return value.toDouble();
}

void _requireKeys(
  Map<String, dynamic> object,
  List<String> keys,
  String field,
) {
  for (final key in keys) {
    if (!object.containsKey(key)) {
      throw FormatException('$field.$key is required.');
    }
  }
}
