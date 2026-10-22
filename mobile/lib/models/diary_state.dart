import 'meal_entry.dart';

const diarySchemaVersion = 2;

class DiaryState {
  DiaryState({Iterable<MealEntry> meals = const [], this.dailyCalorieGoal})
    : meals = List.unmodifiable(meals) {
    final ids = this.meals.map((meal) => meal.mealId).toSet();
    if (ids.length != this.meals.length) {
      throw const FormatException('Meal IDs must be unique.');
    }
    final photoReferences = this.meals
        .map((meal) => meal.photoReference)
        .whereType<String>()
        .toSet();
    if (photoReferences.length !=
        this.meals.where((meal) => meal.photoReference != null).length) {
      throw const FormatException('A meal photo can belong to only one meal.');
    }
    if (dailyCalorieGoal != null) {
      _positiveNumber(dailyCalorieGoal!, 'daily_calorie_goal');
    }
  }

  final List<MealEntry> meals;
  final double? dailyCalorieGoal;

  static final empty = DiaryState();

  factory DiaryState.fromJson(Map<String, dynamic> json) {
    if (json.keys.toSet().difference({
          'schema_version',
          'meals',
          'daily_calorie_goal',
        }).isNotEmpty ||
        !json.keys.toSet().containsAll({
          'schema_version',
          'meals',
          'daily_calorie_goal',
        })) {
      throw const FormatException('Diary state fields are invalid.');
    }
    final version = json['schema_version'];
    if (version is! int || (version != 1 && version != diarySchemaVersion)) {
      throw const FormatException('Unsupported diary schema version.');
    }
    final rawMeals = json['meals'];
    if (rawMeals is! List) {
      throw const FormatException('Diary meals must be an array.');
    }
    final meals = rawMeals
        .map((rawMeal) {
          if (rawMeal is! Map<String, dynamic>) {
            throw const FormatException('Each diary meal must be an object.');
          }
          return MealEntry.fromJson(rawMeal);
        })
        .toList(growable: false);
    final rawGoal = json['daily_calorie_goal'];
    return DiaryState(
      meals: meals,
      dailyCalorieGoal: rawGoal == null
          ? null
          : _positiveNumber(rawGoal, 'daily_calorie_goal'),
    );
  }

  Map<String, Object?> toJson() => {
    'schema_version': diarySchemaVersion,
    'meals': meals.map((meal) => meal.toJson()).toList(growable: false),
    'daily_calorie_goal': dailyCalorieGoal,
  };
}

double _positiveNumber(Object? value, String field) {
  if (value is! num || !value.isFinite || value <= 0) {
    throw FormatException('$field must be finite and positive.');
  }
  return value.toDouble();
}
