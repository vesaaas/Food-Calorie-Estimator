import '../models/dashboard_summary.dart';
import 'diary_controller.dart';

/// Derives dashboard values from saved meal snapshots in the shared controller.
class DashboardSummaryService {
  const DashboardSummaryService._();

  static DashboardSummary summarize(
    DiaryController controller, {
    required DateTime today,
  }) {
    final localToday = today.toLocal();
    final day = DateTime(localToday.year, localToday.month, localToday.day);
    final weekStart = DateTime(day.year, day.month, day.day - 6);
    final week = List<DashboardDaySummary>.generate(7, (index) {
      final currentDay = DateTime(
        weekStart.year,
        weekStart.month,
        weekStart.day + index,
      );
      return DashboardDaySummary(
        day: currentDay,
        totals: _dashboardTotals(controller.totalsForDay(currentDay)),
      );
    }, growable: false);
    return DashboardSummary(
      today: day,
      dailyTotals: _dashboardTotals(controller.totalsForDay(day)),
      dailyGoal: controller.dailyCalorieGoal,
      week: List.unmodifiable(week),
      weeklyCalories: _finiteSum(week.map((item) => item.totals.caloriesKcal)),
      weeklyMealCount: week.fold<int>(
        0,
        (count, item) => count + item.totals.mealCount,
      ),
    );
  }
}

DashboardTotals _dashboardTotals(DailyNutritionTotals totals) =>
    DashboardTotals(
      caloriesKcal: totals.caloriesKcal,
      proteinGrams: totals.proteinGrams,
      carbohydratesGrams: totals.carbohydratesGrams,
      fatGrams: totals.fatGrams,
      mealCount: totals.mealCount,
    );

double _finiteSum(Iterable<double> values) {
  var sum = 0.0;
  for (final value in values) {
    sum += value;
    if (!sum.isFinite) {
      throw StateError('A weekly calorie total is not finite.');
    }
  }
  return sum;
}
