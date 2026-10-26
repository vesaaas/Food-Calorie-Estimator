import 'dart:async';
import 'dart:math';

import 'package:flutter/foundation.dart';

import '../models/diary_state.dart';
import '../models/meal_entry.dart';
import '../repositories/diary_repository.dart';
import '../repositories/sqlite_diary_repository.dart';
import 'meal_photo_store.dart';
import 'nutrition_service.dart';

typedef DiaryClock = DateTime Function();
typedef MealIdFactory = String Function();

class DailyNutritionTotals {
  const DailyNutritionTotals({
    required this.caloriesKcal,
    required this.proteinGrams,
    required this.carbohydratesGrams,
    required this.fatGrams,
    required this.mealCount,
  });

  final double caloriesKcal;
  final double proteinGrams;
  final double carbohydratesGrams;
  final double fatGrams;
  final int mealCount;

  factory DailyNutritionTotals.fromMeals(Iterable<MealEntry> meals) {
    final entries = meals.toList(growable: false);
    return DailyNutritionTotals(
      caloriesKcal: _safeSum(entries.map((meal) => meal.caloriesKcal)),
      proteinGrams: _safeSum(entries.map((meal) => meal.proteinGrams)),
      carbohydratesGrams: _safeSum(
        entries.map((meal) => meal.carbohydratesGrams),
      ),
      fatGrams: _safeSum(entries.map((meal) => meal.fatGrams)),
      mealCount: entries.length,
    );
  }
}

/// Shared app state and serialized diary operations for Scan, Diary, and Goal.
class DiaryController extends ChangeNotifier {
  DiaryController({
    required this.repository,
    MealPhotoStore? photoStore,
    DiaryClock? clock,
    MealIdFactory? mealIdFactory,
  }) : photoStore = photoStore ?? const NoopMealPhotoStore(),
       _clock = clock ?? DateTime.now,
       _mealIdFactory = mealIdFactory ?? _newMealId {
    _lastKnownLocalDay = _localCalendarDay(_clock());
  }

  factory DiaryController.local({
    DiaryClock? clock,
    MealIdFactory? mealIdFactory,
  }) => DiaryController(
    repository: SqliteDiaryRepository(),
    photoStore: AppSupportMealPhotoStore(),
    clock: clock,
    mealIdFactory: mealIdFactory,
  );

  final DiaryRepository repository;
  final MealPhotoStore photoStore;
  final DiaryClock _clock;
  final MealIdFactory _mealIdFactory;
  Future<void> _operationQueue = Future<void>.value();
  DiaryState? _state;
  bool _isLoading = false;
  Object? _loadError;
  late DateTime _lastKnownLocalDay;

  bool get isInitialized => _state != null;
  bool get isLoading => _isLoading;
  Object? get loadError => _loadError;
  DiaryState get state => _state ?? DiaryState.empty;
  double? get dailyCalorieGoal => state.dailyCalorieGoal;

  DateTime get today {
    final day = _localCalendarDay(_clock());
    _lastKnownLocalDay = day;
    return day;
  }

  /// Notifies date-dependent views after the local day changes, without
  /// mutating or persisting the diary state.
  bool refreshIfLocalDayChanged() {
    final currentDay = _localCalendarDay(_clock());
    if (_sameDay(currentDay, _lastKnownLocalDay)) return false;
    _lastKnownLocalDay = currentDay;
    notifyListeners();
    return true;
  }

  Future<void> initialize() => _serialize(() async {
    if (isInitialized) return;
    _isLoading = true;
    _loadError = null;
    notifyListeners();
    try {
      _state = await repository.load();
      try {
        await photoStore.removeUnreferencedPhotos(
          _state!.meals
              .map((meal) => meal.photoReference)
              .whereType<String>()
              .toSet(),
        );
      } on Object {
        // An orphan cleanup failure must not make a valid diary unavailable.
      }
    } on Object catch (error) {
      _loadError = error;
    } finally {
      _isLoading = false;
      _lastKnownLocalDay = _localCalendarDay(_clock());
      notifyListeners();
    }
  });

  List<MealEntry> mealsForDay(DateTime day) {
    final localDay = day.toLocal();
    final meals =
        state.meals
            .where((meal) => _sameDay(meal.loggedAt.toLocal(), localDay))
            .toList(growable: false)
          ..sort((left, right) => right.loggedAt.compareTo(left.loggedAt));
    return List.unmodifiable(meals);
  }

  DailyNutritionTotals totalsForDay(DateTime day) =>
      DailyNutritionTotals.fromMeals(mealsForDay(day));

  Future<MealEntry> addMeal(
    NutritionEstimate estimate, {
    double? classifierConfidence,
    String? photoSourcePath,
  }) => _serialize(() async {
    final current = _requireInitialized();
    var mealId = _mealIdFactory();
    while (current.meals.any((meal) => meal.mealId == mealId)) {
      mealId = _mealIdFactory();
    }
    final photoReference = photoSourcePath == null
        ? null
        : await photoStore.savePhoto(photoSourcePath);
    try {
      final meal = MealEntry(
        mealId: mealId,
        foodId: estimate.food.foodId,
        displayName: estimate.food.displayName,
        loggedAt: _clock().toUtc(),
        portionGrams: estimate.portionGrams,
        portionMode: estimate.portionMode,
        caloriesKcal: estimate.nutrients.energyKcal,
        proteinGrams: estimate.nutrients.proteinGrams,
        carbohydratesGrams: estimate.nutrients.carbohydratesGrams,
        fatGrams: estimate.nutrients.fatGrams,
        classifierConfidence: classifierConfidence,
        photoReference: photoReference,
      );
      await _save(
        DiaryState(
          meals: [...current.meals, meal],
          dailyCalorieGoal: current.dailyCalorieGoal,
        ),
      );
      return meal;
    } on Object {
      if (photoReference != null) {
        try {
          await photoStore.deletePhoto(photoReference);
        } on Object {
          try {
            await photoStore.removeUnreferencedPhotos(
              current.meals
                  .map((meal) => meal.photoReference)
                  .whereType<String>()
                  .toSet(),
            );
          } on Object {
            // Retry failed photo rollback during the next successful load.
          }
        }
      }
      rethrow;
    }
  });

  Future<void> removeMeal(String mealId) => _serialize(() async {
    final current = _requireInitialized();
    final removed = current.meals.where((meal) => meal.mealId == mealId);
    final remaining = current.meals
        .where((meal) => meal.mealId != mealId)
        .toList(growable: false);
    if (remaining.length == current.meals.length) {
      throw ArgumentError.value(mealId, 'mealId', 'Meal not found.');
    }
    await _save(
      DiaryState(meals: remaining, dailyCalorieGoal: current.dailyCalorieGoal),
    );
    try {
      await photoStore.deletePhoto(removed.first.photoReference);
    } on Object {
      // Diary removal is committed independently; an undeletable file is safe
      // and will be retried by orphan cleanup on a later initialization.
    }
  });

  Future<void> setDailyCalorieGoal(double calories) => _serialize(() async {
    final current = _requireInitialized();
    await _save(DiaryState(meals: current.meals, dailyCalorieGoal: calories));
  });

  DiaryState _requireInitialized() {
    final current = _state;
    if (current == null) {
      throw StateError('The saved diary is unavailable until it loads.');
    }
    return current;
  }

  Future<void> _save(DiaryState next) async {
    await repository.save(next);
    _state = next;
    _loadError = null;
    _lastKnownLocalDay = _localCalendarDay(_clock());
    notifyListeners();
  }

  Future<T> _serialize<T>(Future<T> Function() operation) {
    final completer = Completer<T>();
    _operationQueue = _operationQueue.then((_) async {
      try {
        completer.complete(await operation());
      } on Object catch (error, stackTrace) {
        completer.completeError(error, stackTrace);
      }
    });
    return completer.future;
  }
}

bool _sameDay(DateTime left, DateTime right) =>
    left.year == right.year &&
    left.month == right.month &&
    left.day == right.day;

DateTime _localCalendarDay(DateTime time) {
  final local = time.toLocal();
  return DateTime(local.year, local.month, local.day);
}

double _safeSum(Iterable<double> values) {
  var total = 0.0;
  for (final value in values) {
    total += value;
    if (!total.isFinite) {
      throw StateError('A daily nutrition total is not finite.');
    }
  }
  return total;
}

String _newMealId() {
  final random = Random.secure();
  final bytes = List<int>.generate(16, (_) => random.nextInt(256));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  final hex = bytes.map((value) => value.toRadixString(16).padLeft(2, '0'));
  final value = hex.join();
  return '${value.substring(0, 8)}-${value.substring(8, 12)}-'
      '${value.substring(12, 16)}-${value.substring(16, 20)}-'
      '${value.substring(20)}';
}
