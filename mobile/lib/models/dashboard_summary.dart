import 'dart:math' as math;

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
    required this.weeklyProteinGrams,
    required this.weeklyCarbohydratesGrams,
    required this.weeklyFatGrams,
    required this.weeklyMealCount,
  });

  final DateTime today;
  final DashboardTotals dailyTotals;
  final double? dailyGoal;
  final List<DashboardDaySummary> week;
  final double weeklyCalories;
  final double weeklyProteinGrams;
  final double weeklyCarbohydratesGrams;
  final double weeklyFatGrams;
  final int weeklyMealCount;

  /// Energy estimated from the logged macros, excluding other calorie sources.
  /// Returns the estimate when it can be represented as a finite double;
  /// otherwise returns null to avoid reporting an infinite numeric result.
  double? get weeklyMacroEnergyKcal {
    final scale = math.max(
      weeklyProteinGrams,
      math.max(weeklyCarbohydratesGrams, weeklyFatGrams),
    );
    if (scale == 0) return 0;

    final scaledEnergy =
        (weeklyProteinGrams / scale) * 4 +
        (weeklyCarbohydratesGrams / scale) * 4 +
        (weeklyFatGrams / scale) * 9;
    if (scaledEnergy > double.maxFinite / scale) return null;
    return scaledEnergy * scale;
  }

  /// Shares of estimated energy from the three logged macros, or null if zero.
  MacroEnergyDistribution? get weeklyMacroEnergyDistribution {
    final scale = math.max(
      weeklyProteinGrams,
      math.max(weeklyCarbohydratesGrams, weeklyFatGrams),
    );
    if (scale == 0) return null;

    // Scale grams before applying energy factors so neither the factors nor
    // their sum can overflow for finite, nonnegative macro totals.
    final proteinEnergy = (weeklyProteinGrams / scale) * 4;
    final carbohydratesEnergy = (weeklyCarbohydratesGrams / scale) * 4;
    final fatEnergy = (weeklyFatGrams / scale) * 9;
    final totalEnergy = proteinEnergy + carbohydratesEnergy + fatEnergy;
    return MacroEnergyDistribution(
      protein: (proteinEnergy / totalEnergy).clamp(0, 1),
      carbohydrates: (carbohydratesEnergy / totalEnergy).clamp(0, 1),
      fat: (fatEnergy / totalEnergy).clamp(0, 1),
    );
  }

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

class MacroEnergyDistribution {
  const MacroEnergyDistribution({
    required this.protein,
    required this.carbohydrates,
    required this.fat,
  });

  final double protein;
  final double carbohydrates;
  final double fat;

  double get total => protein + carbohydrates + fat;
}
