import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/repositories/json_diary_repository.dart';

void main() {
  late Directory directory;
  late JsonDiaryRepository repository;

  setUp(() async {
    directory = await Directory.systemTemp.createTemp('diary_repository_test');
    repository = JsonDiaryRepository(directoryProvider: () async => directory);
  });

  tearDown(() async => directory.delete(recursive: true));

  test('missing file loads an empty diary', () async {
    expect((await repository.load()).meals, isEmpty);
    expect((await repository.load()).dailyCalorieGoal, isNull);
  });

  test(
    'JSON round trip and persisted reload preserve meal snapshots',
    () async {
      final state = DiaryState(meals: [_meal()], dailyCalorieGoal: 2100.5);
      await repository.save(state);

      final reloaded = await repository.load();
      expect(reloaded.toJson(), state.toJson());
      expect(reloaded.meals.single.foodId, 'french_fries');
      expect(reloaded.meals.single.displayName, 'French Fries');
      expect(reloaded.meals.single.portionGrams, 200);
      expect(reloaded.meals.single.caloriesKcal, 578);
      expect(reloaded.meals.single.proteinGrams, 7);
      expect(reloaded.meals.single.classifierConfidence, 0.999);
      expect(reloaded.dailyCalorieGoal, 2100.5);
    },
  );

  test(
    'malformed, duplicate-key, and unsupported state are rejected',
    () async {
      final file = File('${directory.path}/food_calorie_diary.json');
      await file.writeAsString('{broken');
      expect(repository.load(), throwsA(isA<DiaryPersistenceException>()));

      await file.writeAsString(
        '{"schema_version":1,"schema_version":1,"meals":[],"daily_calorie_goal":null}',
      );
      expect(repository.load(), throwsA(isA<DiaryPersistenceException>()));

      await file.writeAsString(
        '{"schema_version":3,"meals":[],"daily_calorie_goal":null}',
      );
      expect(repository.load(), throwsA(isA<DiaryPersistenceException>()));
    },
  );

  test('version 1 entries without a photo remain readable', () async {
    final file = File('${directory.path}/food_calorie_diary.json');
    await file.writeAsString('''
{"schema_version":1,"daily_calorie_goal":null,"meals":[{
"meal_id":"legacy-meal","food_id":"pizza","display_name":"Pizza",
"logged_at":"2026-10-22T11:30:00Z","portion_grams":132,
"portion_mode":"default","calories_kcal":355.08,"protein_g":15.048,
"carbohydrates_g":41.844,"fat_g":13.332
}]}
''');

    final meal = (await repository.load()).meals.single;
    expect(meal.photoReference, isNull);
    expect(meal.foodId, 'pizza');
    expect(meal.caloriesKcal, 355.08);
  });
}

MealEntry _meal({String id = 'meal-1', DateTime? at}) => MealEntry(
  mealId: id,
  foodId: 'french_fries',
  displayName: 'French Fries',
  loggedAt: at ?? DateTime.utc(2026, 10, 22, 11, 30),
  portionGrams: 200,
  portionMode: 'grams',
  caloriesKcal: 578,
  proteinGrams: 7,
  carbohydratesGrams: 74.4,
  fatGrams: 28,
  classifierConfidence: 0.999,
);
