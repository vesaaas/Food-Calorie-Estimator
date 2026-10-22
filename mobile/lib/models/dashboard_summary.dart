class DashboardTotals {
  const DashboardTotals({
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
}

class DashboardDaySummary {
  const DashboardDaySummary({required this.day, required this.totals});

  final DateTime day;
  final DashboardTotals totals;
}

class DashboardSummary {
  const DashboardSummary({
    required this.today,
    required this.dailyTotals,
    required this.dailyGoal,
    required this.week,
    required this.weeklyCalories,
    required this.weeklyMealCount,
  });

  final DateTime today;
  final DashboardTotals dailyTotals;
  final double? dailyGoal;
  final List<DashboardDaySummary> week;
  final double weeklyCalories;
  final int weeklyMealCount;

  double? get progress =>
      dailyGoal == null ? null : dailyTotals.caloriesKcal / dailyGoal!;

  double? get remainingCalories => dailyGoal == null
      ? null
      : (dailyGoal! - dailyTotals.caloriesKcal).clamp(0, double.infinity);

  double? get exceededCalories => dailyGoal == null || progress! <= 1
      ? null
      : dailyTotals.caloriesKcal - dailyGoal!;

  double get weeklyAverageCalories => weeklyCalories / 7;
}
