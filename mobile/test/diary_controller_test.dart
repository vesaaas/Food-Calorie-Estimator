import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/repositories/diary_repository.dart';
import 'package:food_calorie_estimator/services/diary_controller.dart';
import 'package:food_calorie_estimator/services/nutrition_service.dart';

void main() {
  late NutritionCatalog catalog;
  late MemoryDiaryRepository repository;
  late DiaryController controller;
  var nextId = 0;
  final now = DateTime(2026, 10, 22, 13, 30);

  setUpAll(() async {
    catalog = NutritionCatalog.fromJsonString(
      await File('assets/nutrition/foods.json').readAsString(),
    );
  });

  setUp(() async {
    repository = MemoryDiaryRepository();
    nextId = 0;
    controller = DiaryController(
      repository: repository,
      clock: () => now,
      mealIdFactory: () => 'meal-${++nextId}',
    );
    await controller.initialize();
  });

  tearDown(() => controller.dispose());

  test('adds exact food identity and a nutrition snapshot', () async {
    final estimate = catalog.estimate('french_fries', grams: 200);
    final meal = await controller.addMeal(
      estimate,
      classifierConfidence: 0.999,
    );

    expect(meal.foodId, 'french_fries');
    expect(meal.displayName, 'French fries');
    expect(meal.portionMode, 'grams');
    expect(meal.portionGrams, 200);
    expect(meal.caloriesKcal, estimate.nutrients.energyKcal);
    expect(meal.proteinGrams, estimate.nutrients.proteinGrams);
    expect(meal.carbohydratesGrams, estimate.nutrients.carbohydratesGrams);
    expect(meal.fatGrams, estimate.nutrients.fatGrams);
    expect(meal.classifierConfidence, 0.999);
    expect(repository.saved.meals.single.toJson(), meal.toJson());
  });

  test('confidence is metadata and does not change nutrition', () async {
    final estimate = catalog.estimate('pizza');
    final first = await controller.addMeal(estimate, classifierConfidence: 0.2);
    final second = await controller.addMeal(estimate, classifierConfidence: 1);
    expect(first.classifierConfidence, 0.2);
    expect(second.classifierConfidence, 1);
    expect(first.caloriesKcal, second.caloriesKcal);
    expect(first.proteinGrams, second.proteinGrams);
    expect(first.carbohydratesGrams, second.carbohydratesGrams);
    expect(first.fatGrams, second.fatGrams);
  });

  test('removal preserves other meals and the calorie goal', () async {
    final first = await controller.addMeal(catalog.estimate('pizza'));
    final second = await controller.addMeal(catalog.estimate('sushi'));
    await controller.setDailyCalorieGoal(1900);
    await controller.removeMeal(first.mealId);

    expect(controller.state.meals.map((meal) => meal.mealId), [second.mealId]);
    expect(controller.dailyCalorieGoal, 1900);
    expect(repository.saved.meals.single.foodId, 'sushi');
  });

  test(
    'today totals exclude meals on different local calendar dates',
    () async {
      final today = await controller.addMeal(catalog.estimate('pizza'));
      final yesterday = MealEntry(
        mealId: 'yesterday',
        foodId: 'sushi',
        displayName: 'Sushi',
        loggedAt: DateTime(2026, 10, 21, 23, 59),
        portionGrams: 100,
        portionMode: 'grams',
        caloriesKcal: 10,
        proteinGrams: 2,
        carbohydratesGrams: 3,
        fatGrams: 4,
      );
      final datedRepository = MemoryDiaryRepository()
        ..saved = DiaryState(meals: [today, yesterday]);
      final datedController = DiaryController(
        repository: datedRepository,
        clock: () => now,
      );
      await datedController.initialize();

      expect(
        datedController.mealsForDay(datedController.today).map((m) => m.mealId),
        [today.mealId],
      );
      final totals = datedController.totalsForDay(datedController.today);
      expect(totals.mealCount, 1);
      expect(totals.caloriesKcal, today.caloriesKcal);
      datedController.dispose();
    },
  );

  test('goal set, update, and reload preserve meals', () async {
    final meal = await controller.addMeal(catalog.estimate('pizza'));
    await controller.setDailyCalorieGoal(1800);
    await controller.setDailyCalorieGoal(2000.5);
    expect(controller.dailyCalorieGoal, 2000.5);
    expect(controller.state.meals.single.toJson(), meal.toJson());

    final reloaded = DiaryController(repository: repository, clock: () => now);
    await reloaded.initialize();
    expect(reloaded.dailyCalorieGoal, 2000.5);
    expect(reloaded.state.meals.single.toJson(), meal.toJson());
    reloaded.dispose();
  });

  test('overlapping writes are serialized without dropping updates', () async {
    final futures = <Future<dynamic>>[
      controller.addMeal(catalog.estimate('pizza')),
      controller.setDailyCalorieGoal(2200),
      controller.addMeal(catalog.estimate('sushi')),
    ];
    await Future.wait(futures);

    expect(controller.state.meals.map((meal) => meal.foodId), [
      'pizza',
      'sushi',
    ]);
    expect(controller.dailyCalorieGoal, 2200);
    expect(repository.saved.toJson(), controller.state.toJson());
  });

  test('invalid goals are rejected without changing diary state', () async {
    for (final invalid in [0.0, -1.0, double.nan, double.infinity]) {
      await expectLater(
        controller.setDailyCalorieGoal(invalid),
        throwsA(anything),
      );
    }
    expect(controller.dailyCalorieGoal, isNull);
  });
}

class MemoryDiaryRepository implements DiaryRepository {
  DiaryState saved = DiaryState.empty;

  @override
  Future<DiaryState> load() async => saved;

  @override
  Future<void> save(DiaryState state) async {
    saved = state;
  }
}
