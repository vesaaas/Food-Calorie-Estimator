import 'dart:convert';
import 'dart:io';

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/services/food_classification_service.dart';
import 'package:food_calorie_estimator/services/nutrition_service.dart';

void main() {
  late NutritionCatalog catalog;
  late String catalogJson;

  setUpAll(() async {
    catalogJson = await File(nutritionCatalogAssetPath).readAsString();
    catalog = NutritionCatalog.fromJsonString(catalogJson);
  });

  test('catalog covers exactly the classifier food IDs', () {
    expect(catalog.foodIds.toSet(), foodClassNames.toSet());
    expect(catalog.foodIds, hasLength(13));
  });

  test(
    'canonical records retain source values, mappings, and serving defaults',
    () {
      final pizza = catalog.lookup('pizza');
      expect(pizza.displayName, 'Pizza');
      expect(
        pizza.referenceDescription,
        'Pizza, cheese, from restaurant or fast food, thick crust',
      );
      expect(pizza.per100g.energyKcal, 271);
      expect(pizza.per100g.proteinGrams, 10.8);
      expect(pizza.per100g.carbohydratesGrams, 33.2);
      expect(pizza.per100g.fatGrams, 10.5);
      expect(pizza.defaultServing.label, '1 piece, large pizza');
      expect(pizza.defaultServing.grams, 132);
      expect(pizza.defaultServing.sourcePortionId, 303695);
      expect(pizza.nutritionSource['fdc_id'], 2708617);

      final borek = catalog.lookup('borek');
      expect(borek.defaultServing.label, '100 g reference portion');
      expect(borek.defaultServing.grams, 100);
      expect(borek.defaultServing.sourcePortionId, isNull);
      expect(borek.nutritionSource['source_record_id'], '06.02.0040');
    },
  );

  test('default serving scales all nutrients using its canonical grams', () {
    for (final foodId in foodClassNames) {
      final food = catalog.lookup(foodId);
      final estimate = catalog.estimate(foodId);
      final factor = food.defaultServing.grams / 100;
      expect(estimate.portionMode, 'default', reason: foodId);
      expect(estimate.portionGrams, food.defaultServing.grams, reason: foodId);
      expect(
        estimate.nutrients.energyKcal,
        food.per100g.energyKcal * factor,
        reason: foodId,
      );
      expect(
        estimate.nutrients.proteinGrams,
        food.per100g.proteinGrams * factor,
        reason: foodId,
      );
      expect(
        estimate.nutrients.carbohydratesGrams,
        food.per100g.carbohydratesGrams * factor,
        reason: foodId,
      );
      expect(
        estimate.nutrients.fatGrams,
        food.per100g.fatGrams * factor,
        reason: foodId,
      );
    }
  });

  test('explicit grams calculate from the per-100-gram reference', () {
    final estimate = catalog.estimate('pizza', grams: 180);
    expect(estimate.portionMode, 'grams');
    expect(estimate.portionGrams, 180);
    expect(estimate.nutrients.energyKcal, closeTo(487.8, 1e-10));
    expect(estimate.nutrients.proteinGrams, closeTo(19.44, 1e-10));
    expect(estimate.nutrients.carbohydratesGrams, closeTo(59.76, 1e-10));
    expect(estimate.nutrients.fatGrams, closeTo(18.9, 1e-10));
  });

  test('fractional and multiple servings scale the default serving', () {
    final half = catalog.estimate('pizza', servings: 0.5);
    expect(half.portionMode, 'servings');
    expect(half.portionGrams, 66);
    expect(half.nutrients.energyKcal, closeTo(178.86, 1e-10));

    final doubleServing = catalog.estimate('pizza', servings: 2);
    expect(doubleServing.portionGrams, 264);
    expect(doubleServing.nutrients.energyKcal, closeTo(715.44, 1e-10));
  });

  test('invalid portions and simultaneous modes are rejected', () {
    for (final invalid in [
      0.0,
      -1.0,
      double.nan,
      double.infinity,
      double.negativeInfinity,
    ]) {
      expect(
        () => catalog.estimate('pizza', grams: invalid),
        throwsFormatException,
      );
      expect(
        () => catalog.estimate('pizza', servings: invalid),
        throwsFormatException,
      );
    }
    expect(
      () => catalog.estimate('pizza', grams: 100, servings: 1),
      throwsArgumentError,
    );
  });

  test('unknown or differently spelled food IDs do not resolve', () {
    for (final invalid in ['Pizza', 'pizza ', 'ice cream', 'unknown']) {
      expect(() => catalog.estimate(invalid), throwsArgumentError);
    }
  });

  test('classifier confidence is not part of nutrition calculation', () async {
    const lowConfidence = FoodClassification(food: 'sarma', confidence: 0.01);
    const highConfidence = FoodClassification(food: 'sarma', confidence: 0.99);
    final low = catalog.estimate(lowConfidence.food, grams: 125);
    final high = catalog.estimate(highConfidence.food, grams: 125);
    expect(low.portionGrams, high.portionGrams);
    expect(low.nutrients.energyKcal, high.nutrients.energyKcal);
    expect(low.nutrients.proteinGrams, high.nutrients.proteinGrams);
    expect(low.nutrients.carbohydratesGrams, high.nutrients.carbohydratesGrams);
    expect(low.nutrients.fatGrams, high.nutrients.fatGrams);
  });

  test(
    'asset service loads and caches the Flutter nutrition catalog',
    () async {
      final bundle = _MemoryAssetBundle(catalogJson);
      final service = AssetFoodNutritionService(bundle: bundle);
      final estimate = await service.estimate('french_fries');
      final secondEstimate = await service.estimate('pizza');
      expect(estimate.food.displayName, 'French fries');
      expect(estimate.portionGrams, 110);
      expect(estimate.portionMode, 'default');
      expect(secondEstimate.food.foodId, 'pizza');
      expect(bundle.loadCalls, 1);
    },
  );

  test('catalog parser rejects class coverage changes', () async {
    final json = jsonDecode(catalogJson) as Map<String, dynamic>;
    json['foods'] = (json['foods'] as List)
        .where((food) => (food as Map<String, dynamic>)['food_id'] != 'sarma')
        .toList();
    expect(
      () => NutritionCatalog.fromJsonString(jsonEncode(json)),
      throwsFormatException,
    );
  });
}

class _MemoryAssetBundle extends CachingAssetBundle {
  _MemoryAssetBundle(this.source);

  final String source;
  int loadCalls = 0;

  @override
  Future<ByteData> load(String key) async {
    loadCalls++;
    return ByteData.sublistView(Uint8List.fromList(utf8.encode(source)));
  }
}
