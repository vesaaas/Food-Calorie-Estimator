import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/app.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/repositories/diary_repository.dart';
import 'package:food_calorie_estimator/services/diary_controller.dart';

void main() {
  late _MutableClock clock;
  late DiaryController controller;

  setUp(() async {
    clock = _MutableClock(DateTime(2026, 10, 22, 23, 59));
    controller = DiaryController(
      repository: _MemoryRepository(_twoDayState),
      clock: () => clock.now,
    );
    await controller.initialize();
  });

  tearDown(() => controller.dispose());

  testWidgets('Diary rolls its today list forward after app resume', (
    tester,
  ) async {
    await _launch(tester, controller);
    await tester.tap(find.text('Diary').last);
    await tester.pumpAndSettle();
    expect(find.text('Meal A'), findsOneWidget);
    expect(find.text('Meal B'), findsNothing);
    expect(find.text('100.0 kcal'), findsNWidgets(2));

    clock.now = DateTime(2026, 10, 23, 0, 1);
    await _resumeApp(tester);

    expect(find.text('Meal A'), findsNothing);
    expect(find.text('Meal B'), findsOneWidget);
    expect(find.text('250.0 kcal'), findsNWidgets(2));
  });

  testWidgets(
    'Dashboard Today rolls date and totals forward after app resume',
    (tester) async {
      await _launch(tester, controller);
      await tester.tap(find.text('Dashboard').last);
      await tester.pumpAndSettle();
      expect(find.textContaining('10/22/2026'), findsOneWidget);
      expect(find.text('100.0 kcal'), findsOneWidget);

      clock.now = DateTime(2026, 10, 23, 0, 1);
      await _resumeApp(tester);

      expect(find.textContaining('10/23/2026'), findsOneWidget);
      expect(find.text('250.0 kcal'), findsOneWidget);
    },
  );

  testWidgets('Dashboard Weekly moves its seven-day window after app resume', (
    tester,
  ) async {
    await _launch(tester, controller);
    await tester.tap(find.text('Dashboard').last);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Weekly'));
    await tester.pumpAndSettle();
    expect(
      find.byKey(const Key('dashboard-weekday-2026-10-16')),
      findsOneWidget,
    );
    expect(
      find.byKey(const Key('dashboard-weekday-2026-10-22')),
      findsOneWidget,
    );

    clock.now = DateTime(2026, 10, 23, 0, 1);
    await _resumeApp(tester);

    expect(find.byKey(const Key('dashboard-weekday-2026-10-16')), findsNothing);
    expect(
      find.byKey(const Key('dashboard-weekday-2026-10-17')),
      findsOneWidget,
    );
    expect(
      find.byKey(const Key('dashboard-weekday-2026-10-23')),
      findsOneWidget,
    );
  });

  testWidgets('same-day resume does not notify or mutate diary state', (
    tester,
  ) async {
    await _launch(tester, controller);
    final originalState = controller.state;
    var notifications = 0;
    controller.addListener(() => notifications++);

    clock.now = DateTime(2026, 10, 22, 23, 59);
    await _resumeApp(tester);

    expect(notifications, 0);
    expect(controller.state, same(originalState));
    expect((controller.repository as _MemoryRepository).saveCalls, 0);
  });
}

Future<void> _launch(WidgetTester tester, DiaryController controller) async {
  await tester.pumpWidget(FoodCalorieEstimatorApp(diaryController: controller));
  await tester.pumpAndSettle();
}

Future<void> _resumeApp(WidgetTester tester) async {
  tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.inactive);
  tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.hidden);
  tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.paused);
  await tester.pump();
  tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.hidden);
  tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.inactive);
  tester.binding.handleAppLifecycleStateChanged(AppLifecycleState.resumed);
  await tester.pumpAndSettle();
}

final _twoDayState = DiaryState(
  meals: [
    MealEntry(
      mealId: 'day-a',
      foodId: 'pizza',
      displayName: 'Meal A',
      loggedAt: DateTime(2026, 10, 22, 12),
      portionGrams: 100,
      portionMode: 'grams',
      caloriesKcal: 100,
      proteinGrams: 10,
      carbohydratesGrams: 20,
      fatGrams: 5,
    ),
    MealEntry(
      mealId: 'day-b',
      foodId: 'sushi',
      displayName: 'Meal B',
      loggedAt: DateTime(2026, 10, 23, 12),
      portionGrams: 150,
      portionMode: 'grams',
      caloriesKcal: 250,
      proteinGrams: 15,
      carbohydratesGrams: 30,
      fatGrams: 8,
    ),
  ],
);

class _MutableClock {
  _MutableClock(this.now);

  DateTime now;
}

class _MemoryRepository implements DiaryRepository {
  _MemoryRepository(this.state);

  final DiaryState state;
  var saveCalls = 0;

  @override
  Future<DiaryState> load() async => state;

  @override
  Future<void> save(DiaryState state) async {
    saveCalls++;
  }
}
