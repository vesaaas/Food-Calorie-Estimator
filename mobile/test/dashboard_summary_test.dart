import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/models/dashboard_summary.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/repositories/diary_repository.dart';
import 'package:food_calorie_estimator/services/diary_controller.dart';
import 'package:food_calorie_estimator/services/dashboard_summary_service.dart';

void main() {
  final deviceDay = DateTime(2026, 10, 22);
  late _MemoryRepository repository;
  late DiaryController controller;

  setUp(() async {
    repository = _MemoryRepository();
    controller = DiaryController(
      repository: repository,
      clock: () => DateTime(2026, 10, 22, 13),
      mealIdFactory: () => 'unused',
    );
    await controller.initialize();
  });

  tearDown(() => controller.dispose());

  test('empty day and week use real zero values without a goal', () {
    final summary = DashboardSummaryService.summarize(
      controller,
      today: deviceDay,
    );

    expect(summary.dailyTotals.caloriesKcal, 0);
    expect(summary.dailyTotals.proteinGrams, 0);
    expect(summary.dailyTotals.carbohydratesGrams, 0);
    expect(summary.dailyTotals.fatGrams, 0);
    expect(summary.dailyTotals.mealCount, 0);
    expect(summary.dailyGoal, isNull);
    expect(summary.progress, isNull);
    expect(summary.remainingCalories, isNull);
    expect(summary.week, hasLength(7));
    expect(summary.week.every((day) => day.totals.caloriesKcal == 0), isTrue);
    expect(summary.weeklyCalories, 0);
    expect(summary.weeklyProteinGrams, 0);
    expect(summary.weeklyCarbohydratesGrams, 0);
    expect(summary.weeklyFatGrams, 0);
    expect(summary.weeklyMacroEnergyKcal, 0);
    expect(summary.weeklyMacroEnergyDistribution, isNull);
    expect(summary.weeklyAverageCalories, 0);
    expect(summary.weeklyMealCount, 0);
  });

  test(
    'today sums saved nutrition snapshots and ignores photo/confidence',
    () async {
      await repository.save(
        DiaryState(
          meals: [
            _meal('manual', _localTime(2026, 10, 22, 9), calories: 300),
            _meal(
              'photo',
              _localTime(2026, 10, 22, 12),
              calories: 450,
              protein: 22,
              carbohydrates: 48,
              fat: 18,
              confidence: 0.99,
              photoReference: 'a' * 32 + '.jpg',
            ),
            _meal('yesterday', _localTime(2026, 10, 21, 23), calories: 900),
          ],
          dailyCalorieGoal: 1000,
        ),
      );
      controller.dispose();
      controller = DiaryController(
        repository: repository,
        clock: () => DateTime(2026, 10, 22, 13),
      );
      await controller.initialize();

      final summary = DashboardSummaryService.summarize(
        controller,
        today: deviceDay,
      );
      expect(summary.dailyTotals.caloriesKcal, 750);
      expect(summary.dailyTotals.proteinGrams, 22);
      expect(summary.dailyTotals.carbohydratesGrams, 48);
      expect(summary.dailyTotals.fatGrams, 18);
      expect(summary.dailyTotals.mealCount, 2);
      expect(summary.progress, 0.75);
      expect(summary.remainingCalories, 250);
      expect(summary.exceededCalories, isNull);
    },
  );

  test('exact goal has zero remaining and over-goal reports excess', () async {
    await repository.save(
      DiaryState(
        meals: [_meal('exact', _localTime(2026, 10, 22), calories: 500)],
        dailyCalorieGoal: 500,
      ),
    );
    controller.dispose();
    controller = DiaryController(repository: repository);
    await controller.initialize();
    var summary = DashboardSummaryService.summarize(
      controller,
      today: deviceDay,
    );
    expect(summary.progress, 1);
    expect(summary.remainingCalories, 0);
    expect(summary.exceededCalories, isNull);

    await repository.save(
      DiaryState(
        meals: [_meal('exceeded', _localTime(2026, 10, 22), calories: 650)],
        dailyCalorieGoal: 500,
      ),
    );
    controller.dispose();
    controller = DiaryController(repository: repository);
    await controller.initialize();
    summary = DashboardSummaryService.summarize(controller, today: deviceDay);
    expect(summary.progress, 1.3);
    expect(summary.remainingCalories, 0);
    expect(summary.exceededCalories, 150);
  });

  test(
    'weekly summary includes exactly seven local dates and zero days',
    () async {
      final start = DateTime(2026, 10, 16);
      await repository.save(
        DiaryState(
          meals: [
            _meal(
              'start',
              _localTime(2026, 10, 16, 1),
              calories: 100,
              protein: 10,
            ),
            _meal(
              'middle',
              _localTime(2026, 10, 19, 23),
              calories: 350,
              protein: 4,
              carbohydrates: 48,
            ),
            _meal(
              'end',
              _localTime(2026, 10, 22, 2),
              calories: 250,
              protein: 8,
              fat: 18,
            ),
            _meal('old', _localTime(2026, 10, 15, 23), calories: 999),
            _meal('future', _localTime(2026, 10, 23, 1), calories: 999),
          ],
        ),
      );
      controller.dispose();
      controller = DiaryController(repository: repository);
      await controller.initialize();

      final summary = DashboardSummaryService.summarize(
        controller,
        today: deviceDay,
      );
      expect(summary.week.map((day) => day.day), [
        for (var offset = 0; offset < 7; offset++)
          DateTime(start.year, start.month, start.day + offset),
      ]);
      expect(summary.week.map((day) => day.totals.caloriesKcal), [
        100,
        0,
        0,
        350,
        0,
        0,
        250,
      ]);
      expect(summary.weeklyCalories, 700);
      expect(summary.weeklyProteinGrams, 22);
      expect(summary.weeklyCarbohydratesGrams, 48);
      expect(summary.weeklyFatGrams, 18);
      expect(summary.weeklyAverageCalories, 100);
      expect(summary.weeklyMealCount, 3);
    },
  );

  test('weekly macro-energy distribution uses 4/4/9 kcal per gram', () async {
    await repository.save(
      DiaryState(
        meals: [
          _meal(
            'macros',
            _localTime(2026, 10, 20),
            calories: 999,
            protein: 10,
            carbohydrates: 20,
            fat: 10,
          ),
        ],
      ),
    );
    controller.dispose();
    controller = DiaryController(repository: repository);
    await controller.initialize();

    final summary = DashboardSummaryService.summarize(
      controller,
      today: deviceDay,
    );
    final distribution = summary.weeklyMacroEnergyDistribution!;
    expect(summary.weeklyMacroEnergyKcal, 210);
    expect(distribution.protein, closeTo(40 / 210, 1e-12));
    expect(distribution.carbohydrates, closeTo(80 / 210, 1e-12));
    expect(distribution.fat, closeTo(90 / 210, 1e-12));
    expect(distribution.total, closeTo(1, 1e-12));
  });

  test('very large macro totals keep energy shares finite and normalized', () {
    final max = double.maxFinite;
    final multiplicationOverflow = _summaryWithMacros(
      protein: max,
      carbohydrates: 0,
      fat: 0,
    );
    final singleShare = multiplicationOverflow.weeklyMacroEnergyDistribution!;
    expect(multiplicationOverflow.weeklyMacroEnergyKcal, isNull);
    expect(singleShare.protein, 1);
    expect(singleShare.carbohydrates, 0);
    expect(singleShare.fat, 0);
    _expectValidShares(singleShare);

    final summationOverflow = _summaryWithMacros(
      protein: max,
      carbohydrates: max,
      fat: max,
    );
    final balanced = summationOverflow.weeklyMacroEnergyDistribution!;
    expect(summationOverflow.weeklyMacroEnergyKcal, isNull);
    expect(balanced.protein, closeTo(4 / 17, 1e-12));
    expect(balanced.carbohydrates, closeTo(4 / 17, 1e-12));
    expect(balanced.fat, closeTo(9 / 17, 1e-12));
    _expectValidShares(balanced);
  });

  test(
    'removing a meal updates weekly macro totals and distribution',
    () async {
      await repository.save(
        DiaryState(
          meals: [
            _meal('keep', _localTime(2026, 10, 22), protein: 10, fat: 2),
            _meal(
              'remove',
              _localTime(2026, 10, 21),
              carbohydrates: 30,
              fat: 8,
            ),
          ],
        ),
      );
      controller.dispose();
      controller = DiaryController(repository: repository);
      await controller.initialize();
      await controller.removeMeal('remove');

      final summary = DashboardSummaryService.summarize(
        controller,
        today: deviceDay,
      );
      expect(summary.weeklyProteinGrams, 10);
      expect(summary.weeklyCarbohydratesGrams, 0);
      expect(summary.weeklyFatGrams, 2);
      expect(summary.weeklyMacroEnergyKcal, 58);
    },
  );

  test(
    'UTC timestamps near midnight are assigned to local calendar dates',
    () async {
      final localBeforeMidnight = DateTime(2026, 10, 21, 23, 30);
      final localAfterMidnight = DateTime(2026, 10, 22, 0, 30);
      await repository.save(
        DiaryState(
          meals: [
            _meal(
              'before',
              localBeforeMidnight.toUtc(),
              calories: 75,
              protein: 7,
            ),
            _meal(
              'after',
              localAfterMidnight.toUtc(),
              calories: 125,
              carbohydrates: 11,
            ),
          ],
        ),
      );
      controller.dispose();
      controller = DiaryController(repository: repository);
      await controller.initialize();

      final summary = DashboardSummaryService.summarize(
        controller,
        today: deviceDay,
      );
      expect(summary.dailyTotals.caloriesKcal, 125);
      expect(summary.dailyTotals.carbohydratesGrams, 11);
      expect(summary.weeklyCalories, 200);
      expect(summary.weeklyProteinGrams, 7);
      expect(summary.weeklyCarbohydratesGrams, 11);
      expect(summary.weeklyMealCount, 2);
    },
  );
}

DashboardSummary _summaryWithMacros({
  required double protein,
  required double carbohydrates,
  required double fat,
}) => DashboardSummary(
  today: DateTime(2026, 10, 22),
  dailyTotals: const DashboardTotals(
    caloriesKcal: 0,
    proteinGrams: 0,
    carbohydratesGrams: 0,
    fatGrams: 0,
    mealCount: 0,
  ),
  dailyGoal: null,
  week: const [],
  weeklyCalories: 0,
  weeklyProteinGrams: protein,
  weeklyCarbohydratesGrams: carbohydrates,
  weeklyFatGrams: fat,
  weeklyMealCount: 0,
);

void _expectValidShares(MacroEnergyDistribution distribution) {
  for (final share in [
    distribution.protein,
    distribution.carbohydrates,
    distribution.fat,
  ]) {
    expect(share.isFinite, isTrue);
    expect(share, inInclusiveRange(0, 1));
  }
  expect(distribution.total, closeTo(1, 1e-12));
}

DateTime _localTime(int year, int month, int day, [int hour = 12]) =>
    DateTime(year, month, day, hour).toUtc();

MealEntry _meal(
  String id,
  DateTime loggedAt, {
  double calories = 0,
  double protein = 0,
  double carbohydrates = 0,
  double fat = 0,
  double? confidence,
  String? photoReference,
}) => MealEntry(
  mealId: id,
  foodId: 'pizza',
  displayName: 'Pizza',
  loggedAt: loggedAt,
  portionGrams: 100,
  portionMode: 'grams',
  caloriesKcal: calories,
  proteinGrams: protein,
  carbohydratesGrams: carbohydrates,
  fatGrams: fat,
  classifierConfidence: confidence,
  photoReference: photoReference,
);

class _MemoryRepository implements DiaryRepository {
  DiaryState state = DiaryState.empty;

  @override
  Future<DiaryState> load() async => state;

  @override
  Future<void> save(DiaryState next) async => state = next;
}
