import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/repositories/diary_repository.dart';
import 'package:food_calorie_estimator/repositories/json_diary_repository.dart';
import 'package:food_calorie_estimator/services/diary_controller.dart';
import 'package:food_calorie_estimator/services/meal_photo_store.dart';
import 'package:food_calorie_estimator/services/nutrition_service.dart';

void main() {
  late Directory root;
  late File source;
  late AppSupportMealPhotoStore photoStore;
  late NutritionCatalog catalog;

  setUp(() async {
    root = await Directory.systemTemp.createTemp('meal_photo_store_test');
    source = File('${root.path}/selected.PNG')..writeAsBytesSync(_onePixelPng);
    photoStore = AppSupportMealPhotoStore(directoryProvider: () async => root);
    catalog = NutritionCatalog.fromJsonString(
      await File('assets/nutrition/foods.json').readAsString(),
    );
  });

  tearDown(() async => root.delete(recursive: true));

  test('photo persists and diary reload retains its safe reference', () async {
    final repository = JsonDiaryRepository(directoryProvider: () async => root);
    final controller = DiaryController(
      repository: repository,
      photoStore: photoStore,
      mealIdFactory: () => 'photo-meal',
    );
    await controller.initialize();
    final meal = await controller.addMeal(
      catalog.estimate('pizza'),
      photoSourcePath: source.path,
      classifierConfidence: 0.9,
    );
    controller.dispose();

    expect(meal.photoReference, matches(RegExp(r'^[a-f0-9]{32}\.png$')));
    expect(
      jsonFileHasNoPhotoBytes(
        await File('${root.path}/food_calorie_diary.json').readAsString(),
      ),
      isTrue,
    );
    final storedPath = await photoStore.resolvePhotoPath(meal.photoReference);
    expect(storedPath, isNotNull);
    expect(storedPath, isNot(source.path));
    expect(storedPath, endsWith('/meal_photos/${meal.photoReference}'));
    expect(await File(storedPath!).readAsBytes(), _onePixelPng);

    final reloaded = DiaryController(
      repository: JsonDiaryRepository(directoryProvider: () async => root),
      photoStore: photoStore,
    );
    await reloaded.initialize();
    expect(reloaded.state.meals.single.photoReference, meal.photoReference);
    expect(
      await photoStore.resolvePhotoPath(
        reloaded.state.meals.single.photoReference,
      ),
      storedPath,
    );
    reloaded.dispose();
  });

  test(
    'each meal owns a distinct photo and removal retains other photos',
    () async {
      final repository = JsonDiaryRepository(
        directoryProvider: () async => root,
      );
      var nextId = 0;
      final controller = DiaryController(
        repository: repository,
        photoStore: photoStore,
        mealIdFactory: () => 'photo-meal-${++nextId}',
      );
      await controller.initialize();
      final first = await controller.addMeal(
        catalog.estimate('pizza'),
        photoSourcePath: source.path,
      );
      final second = await controller.addMeal(
        catalog.estimate('sushi'),
        photoSourcePath: source.path,
      );

      expect(first.photoReference, isNotNull);
      expect(second.photoReference, isNotNull);
      expect(first.photoReference, isNot(second.photoReference));
      final firstPath = await photoStore.resolvePhotoPath(first.photoReference);
      final secondPath = await photoStore.resolvePhotoPath(
        second.photoReference,
      );
      expect(firstPath, isNotNull);
      expect(secondPath, isNotNull);
      expect(firstPath, isNot(secondPath));

      await controller.removeMeal(first.mealId);

      expect(await photoStore.resolvePhotoPath(first.photoReference), isNull);
      expect(
        await photoStore.resolvePhotoPath(second.photoReference),
        secondPath,
      );
      expect(controller.state.meals.single.mealId, second.mealId);
      controller.dispose();
    },
  );

  test(
    'photo copy failure on diary save rolls back the copied photo',
    () async {
      final repository = _FailingSaveRepository();
      final controller = DiaryController(
        repository: repository,
        photoStore: photoStore,
        mealIdFactory: () => 'failed-photo-meal',
      );
      await controller.initialize();
      await expectLater(
        controller.addMeal(
          catalog.estimate('pizza'),
          photoSourcePath: source.path,
        ),
        throwsA(isA<StateError>()),
      );
      expect(controller.state.meals, isEmpty);
      final files = await Directory('${root.path}/meal_photos')
          .list(followLinks: false)
          .toList();
      expect(files, isEmpty);
      controller.dispose();
    },
  );

  test('removal cleanup failure preserves other meals and the goal', () async {
    final repository = JsonDiaryRepository(directoryProvider: () async => root);
    final failingCleanup = _DeleteFailingPhotoStore(photoStore);
    var nextId = 0;
    final controller = DiaryController(
      repository: repository,
      photoStore: failingCleanup,
      mealIdFactory: () => 'retained-photo-meal-${++nextId}',
    );
    await controller.initialize();
    await controller.setDailyCalorieGoal(2000);
    final first = await controller.addMeal(
      catalog.estimate('pizza'),
      photoSourcePath: source.path,
    );
    final second = await controller.addMeal(catalog.estimate('sushi'));
    controller.dispose();

    final next = DiaryController(
      repository: repository,
      photoStore: failingCleanup,
    );
    await next.initialize();
    await next.removeMeal(first.mealId);
    expect(next.state.meals.map((meal) => meal.mealId), [second.mealId]);
    expect(next.dailyCalorieGoal, 2000);
    expect((await repository.load()).meals.single.mealId, second.mealId);
    next.dispose();
  });

  test(
    'unsafe photo references cannot resolve or delete outside files',
    () async {
      final victim = File('${root.path}/victim.png')
        ..writeAsBytesSync([1, 2, 3]);
      expect(await photoStore.resolvePhotoPath('../victim.png'), isNull);
      await photoStore.deletePhoto('../victim.png');
      expect(await victim.exists(), isTrue);
    },
  );

  test(
    'orphan cleanup deletes only app-owned unreferenced photo files',
    () async {
      final keptReference = await photoStore.savePhoto(source.path);
      final orphanReference = await photoStore.savePhoto(source.path);
      await photoStore.removeUnreferencedPhotos({keptReference});

      expect(await photoStore.resolvePhotoPath(keptReference), isNotNull);
      expect(await photoStore.resolvePhotoPath(orphanReference), isNull);
    },
  );
}

bool jsonFileHasNoPhotoBytes(String json) =>
    !json.contains('data:image') && !json.contains('iVBOR');

class _FailingSaveRepository implements DiaryRepository {
  @override
  Future<DiaryState> load() async => DiaryState.empty;

  @override
  Future<void> save(DiaryState state) =>
      Future.error(StateError('Synthetic persistence failure'));
}

class _DeleteFailingPhotoStore implements MealPhotoStore {
  _DeleteFailingPhotoStore(this.delegate);

  final MealPhotoStore delegate;

  @override
  Future<void> deletePhoto(String? photoReference) =>
      Future.error(StateError('Synthetic cleanup failure'));

  @override
  Future<String?> resolvePhotoPath(String? photoReference) =>
      delegate.resolvePhotoPath(photoReference);

  @override
  Future<void> removeUnreferencedPhotos(Set<String> referencedPhotos) =>
      delegate.removeUnreferencedPhotos(referencedPhotos);

  @override
  Future<String> savePhoto(String sourcePath) => delegate.savePhoto(sourcePath);
}

const _onePixelPng = <int>[
  137,
  80,
  78,
  71,
  13,
  10,
  26,
  10,
  0,
  0,
  0,
  13,
  73,
  72,
  68,
  82,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  1,
  8,
  6,
  0,
  0,
  0,
  31,
  21,
  196,
  137,
  0,
  0,
  0,
  13,
  73,
  68,
  65,
  84,
  8,
  215,
  99,
  248,
  207,
  192,
  240,
  31,
  0,
  5,
  0,
  1,
  255,
  137,
  153,
  61,
  29,
  0,
  0,
  0,
  0,
  73,
  69,
  78,
  68,
  174,
  66,
  96,
  130,
];
