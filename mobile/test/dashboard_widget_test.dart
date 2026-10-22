import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/repositories/diary_repository.dart';
import 'package:food_calorie_estimator/screens/dashboard_screen.dart';
import 'package:food_calorie_estimator/services/diary_controller.dart';

void main() {
  testWidgets('Dashboard updates when persisted diary loading completes', (
    tester,
  ) async {
    final pendingState = Completer<DiaryState>();
    final controller = DiaryController(
      repository: _DeferredRepository(pendingState.future),
      clock: () => DateTime(2026, 10, 22, 13),
    );
    addTearDown(controller.dispose);
    unawaited(controller.initialize());

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(body: DashboardScreen(diaryController: controller)),
      ),
    );
    expect(find.text('Loading your saved meals and goal…'), findsOneWidget);

    pendingState.complete(
      DiaryState(
        meals: [
          MealEntry(
            mealId: 'restored-meal',
            foodId: 'pizza',
            displayName: 'Pizza',
            loggedAt: DateTime.utc(2026, 10, 22, 10),
            portionGrams: 100,
            portionMode: 'grams',
            caloriesKcal: 321,
            proteinGrams: 14,
            carbohydratesGrams: 39,
            fatGrams: 12,
          ),
        ],
      ),
    );
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('dashboard-today-calories')), findsOneWidget);
    expect(find.text('321.0 kcal'), findsOneWidget);
    expect(find.text('1'), findsOneWidget);
  });
}

class _DeferredRepository implements DiaryRepository {
  const _DeferredRepository(this.state);

  final Future<DiaryState> state;

  @override
  Future<DiaryState> load() => state;

  @override
  Future<void> save(DiaryState next) async {}
}
