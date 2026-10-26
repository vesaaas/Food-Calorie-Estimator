import 'dart:convert';
import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/repositories/json_diary_repository.dart';
import 'package:food_calorie_estimator/repositories/sqlite_diary_repository.dart';
import 'package:sqflite_common_ffi/sqflite_ffi.dart';

void main() {
  late Directory directory;

  setUpAll(() {
    sqfliteFfiInit();
    databaseFactory = databaseFactoryFfi;
  });

  setUp(() async {
    directory = await Directory.systemTemp.createTemp('sqlite_diary_test');
  });

  tearDown(() async => directory.delete(recursive: true));

  SqliteDiaryRepository createRepository() => SqliteDiaryRepository(
    databaseFactory: databaseFactoryFfi,
    databasePathProvider: () async => directory.path,
    legacyRepository: JsonDiaryRepository(
      directoryProvider: () async => directory,
    ),
  );

  test(
    'initializes an empty database with indexed timestamp and goal schema',
    () async {
      final repository = createRepository();
      expect((await repository.load()).toJson(), DiaryState.empty.toJson());
      final db = await databaseFactoryFfi.openDatabase(
        '${directory.path}/food_calorie_diary.sqlite',
      );
      final indexes = await db.rawQuery(
        "SELECT name FROM sqlite_master WHERE type = ? AND name = ?",
        ['index', 'idx_meals_logged_at_utc'],
      );
      final tables = await db.rawQuery(
        'SELECT name FROM sqlite_master WHERE type = ?',
        ['table'],
      );
      expect(indexes, hasLength(1));
      expect(
        tables.map((row) => row['name']),
        containsAll(['meals', 'goals', 'app_metadata']),
      );
      await db.close();
      await repository.close();
    },
  );

  test(
    'inserts, reads, updates, deletes, orders, and persists all model fields',
    () async {
      final repository = createRepository();
      final older = _meal(
        id: 'older',
        at: DateTime.utc(2026, 10, 20),
        photoReference: '0123456789abcdef0123456789abcdef.jpg',
      );
      final newer = _meal(id: 'newer', at: DateTime.utc(2026, 10, 22));
      await repository.save(
        DiaryState(meals: [newer, older], dailyCalorieGoal: 2050.5),
      );

      final loaded = await repository.load();
      expect(loaded.meals.map((meal) => meal.mealId), ['newer', 'older']);
      expect(loaded.meals.last.toJson(), older.toJson());
      expect(loaded.dailyCalorieGoal, 2050.5);

      final updated = _meal(
        id: 'newer',
        at: DateTime.utc(2026, 10, 24),
        calories: 500,
      );
      await repository.save(
        DiaryState(meals: [updated], dailyCalorieGoal: 2200),
      );
      final afterUpdate = await repository.load();
      expect(afterUpdate.meals.single.toJson(), updated.toJson());
      expect(afterUpdate.dailyCalorieGoal, 2200);

      await repository.save(DiaryState(dailyCalorieGoal: 2200));
      expect((await repository.load()).meals, isEmpty);
      await repository.close();
    },
  );

  test(
    'goal can be saved, updated, cleared, and read after recreation',
    () async {
      final first = createRepository();
      await first.save(DiaryState(dailyCalorieGoal: 1900));
      await first.close();
      final second = createRepository();
      expect((await second.load()).dailyCalorieGoal, 1900);
      await second.save(DiaryState(dailyCalorieGoal: 2150.25));
      await second.close();
      final third = createRepository();
      expect((await third.load()).dailyCalorieGoal, 2150.25);
      await third.save(DiaryState());
      expect((await third.load()).dailyCalorieGoal, isNull);
      await third.close();
    },
  );

  test('migrates legacy JSON meals, goal, nutrition, confidence, and photo reference', () async {
    final legacyFile = File('${directory.path}/food_calorie_diary.json');
    final expected = DiaryState(
      meals: [_meal(photoReference: 'abcdef0123456789abcdef0123456789.png')],
      dailyCalorieGoal: 2044.5,
    );
    await legacyFile.writeAsString('${jsonEncode(expected.toJson())}\n');

    final repository = createRepository();
    expect((await repository.load()).toJson(), expected.toJson());
    expect(await legacyFile.exists(), isTrue);
    await repository.close();
  });

  test('migration is idempotent after repository recreation', () async {
    final legacyFile = File('${directory.path}/food_calorie_diary.json');
    final original = DiaryState(meals: [_meal()], dailyCalorieGoal: 2000);
    await legacyFile.writeAsString(jsonEncode(original.toJson()));
    final first = createRepository();
    await first.load();
    await first.close();

    // A changed legacy file must not be re-imported once migration completed.
    await legacyFile.writeAsString(jsonEncode(DiaryState().toJson()));
    final second = createRepository();
    expect((await second.load()).toJson(), original.toJson());
    expect(await legacyFile.exists(), isTrue);
    await second.close();
  });

  test(
    'failed JSON migration leaves source and does not mark migration complete',
    () async {
      final legacyFile = File('${directory.path}/food_calorie_diary.json');
      await legacyFile.writeAsString('{invalid');
      final repository = createRepository();
      await expectLater(
        repository.load(),
        throwsA(isA<DiaryPersistenceException>()),
      );
      expect(await legacyFile.readAsString(), '{invalid');
      await repository.close();
      final db = await databaseFactoryFfi.openDatabase(
        '${directory.path}/food_calorie_diary.sqlite',
      );
      final markers = await db.query('app_metadata');
      expect(markers, isEmpty);
      await db.close();
    },
  );
}

MealEntry _meal({
  String id = 'meal-1',
  DateTime? at,
  String? photoReference,
  double calories = 578,
}) => MealEntry(
  mealId: id,
  foodId: 'french_fries',
  displayName: 'French Fries',
  loggedAt: at ?? DateTime.utc(2026, 10, 22, 11, 30),
  portionGrams: 200,
  portionMode: 'grams',
  caloriesKcal: calories,
  proteinGrams: 7.25,
  carbohydratesGrams: 74.4,
  fatGrams: 28.1,
  classifierConfidence: 0.999,
  photoReference: photoReference,
);
