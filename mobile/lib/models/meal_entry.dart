import '../services/food_classification_service.dart' show foodClassNames;
import '../services/nutrition_service.dart';

const mealPortionModes = {'default', 'grams', 'servings'};
final _mealPhotoReferencePattern = RegExp(
  r'^[a-f0-9]{32}\.(jpg|jpeg|png|webp|heic)$',
);

class MealEntry {
  MealEntry({
    required this.mealId,
    required this.foodId,
    required this.displayName,
    required DateTime loggedAt,
    required this.portionGrams,
    required this.portionMode,
    required this.caloriesKcal,
    required this.proteinGrams,
    required this.carbohydratesGrams,
    required this.fatGrams,
    this.classifierConfidence,
    this.photoReference,
  }) : loggedAt = loggedAt.toUtc() {
    _validate();
  }

  final String mealId;
  final String foodId;
  final String displayName;
  final DateTime loggedAt;
  final double portionGrams;
  final String portionMode;
  final double caloriesKcal;
  final double proteinGrams;
  final double carbohydratesGrams;
  final double fatGrams;
  final double? classifierConfidence;
  final String? photoReference;

  factory MealEntry.fromEstimate({
    required String mealId,
    required NutritionEstimate estimate,
    required DateTime loggedAt,
    double? classifierConfidence,
  }) {
    final facts = estimate.nutrients;
    return MealEntry(
      mealId: mealId,
      foodId: estimate.food.foodId,
      displayName: estimate.food.displayName,
      loggedAt: loggedAt,
      portionGrams: estimate.portionGrams,
      portionMode: estimate.portionMode,
      caloriesKcal: facts.energyKcal,
      proteinGrams: facts.proteinGrams,
      carbohydratesGrams: facts.carbohydratesGrams,
      fatGrams: facts.fatGrams,
      classifierConfidence: classifierConfidence,
    );
  }

  factory MealEntry.fromJson(Map<String, dynamic> json) {
    const required = {
      'meal_id',
      'food_id',
      'display_name',
      'logged_at',
      'portion_grams',
      'portion_mode',
      'calories_kcal',
      'protein_g',
      'carbohydrates_g',
      'fat_g',
    };
    const allowed = {...required, 'classifier_confidence', 'photo_reference'};
    _requireExactFields(json, required, allowed, 'meal');
    final timestamp = json['logged_at'];
    if (timestamp is! String || !_hasExplicitTimezone(timestamp)) {
      throw const FormatException('meal.logged_at must include a timezone.');
    }
    DateTime loggedAt;
    try {
      loggedAt = DateTime.parse(timestamp);
    } on FormatException {
      throw const FormatException(
        'meal.logged_at must be an ISO-8601 timestamp.',
      );
    }
    return MealEntry(
      mealId: _string(json['meal_id'], 'meal.meal_id'),
      foodId: _string(json['food_id'], 'meal.food_id'),
      displayName: _string(json['display_name'], 'meal.display_name'),
      loggedAt: loggedAt,
      portionGrams: _number(
        json['portion_grams'],
        'meal.portion_grams',
        positive: true,
      ),
      portionMode: _string(json['portion_mode'], 'meal.portion_mode'),
      caloriesKcal: _number(json['calories_kcal'], 'meal.calories_kcal'),
      proteinGrams: _number(json['protein_g'], 'meal.protein_g'),
      carbohydratesGrams: _number(
        json['carbohydrates_g'],
        'meal.carbohydrates_g',
      ),
      fatGrams: _number(json['fat_g'], 'meal.fat_g'),
      classifierConfidence: json.containsKey('classifier_confidence')
          ? _nullableNumber(
              json['classifier_confidence'],
              'meal.classifier_confidence',
            )
          : null,
      photoReference: json.containsKey('photo_reference')
          ? _nullableString(json['photo_reference'], 'meal.photo_reference')
          : null,
    );
  }

  Map<String, Object?> toJson() => {
    'meal_id': mealId,
    'food_id': foodId,
    'display_name': displayName,
    'logged_at': loggedAt.toUtc().toIso8601String(),
    'portion_grams': portionGrams,
    'portion_mode': portionMode,
    'calories_kcal': caloriesKcal,
    'protein_g': proteinGrams,
    'carbohydrates_g': carbohydratesGrams,
    'fat_g': fatGrams,
    'classifier_confidence': classifierConfidence,
    'photo_reference': photoReference,
  };

  void _validate() {
    if (mealId.trim().isEmpty) {
      throw const FormatException('meal_id must be a nonempty string.');
    }
    if (!foodClassNames.contains(foodId)) {
      throw FormatException('Unknown canonical food_id: $foodId');
    }
    if (displayName.trim().isEmpty) {
      throw const FormatException('display_name must be a nonempty string.');
    }
    if (photoReference != null &&
        !_mealPhotoReferencePattern.hasMatch(photoReference!)) {
      throw const FormatException('photo_reference is not a safe photo ID.');
    }
    _number(portionGrams, 'portion_grams', positive: true);
    if (!mealPortionModes.contains(portionMode)) {
      throw FormatException('Unsupported portion_mode: $portionMode');
    }
    _number(caloriesKcal, 'calories_kcal');
    _number(proteinGrams, 'protein_g');
    _number(carbohydratesGrams, 'carbohydrates_g');
    _number(fatGrams, 'fat_g');
    if (classifierConfidence case final confidence?) {
      _number(confidence, 'classifier_confidence');
      if (confidence > 1) {
        throw const FormatException(
          'classifier_confidence must be between zero and one.',
        );
      }
    }
  }
}

String? _nullableString(Object? value, String field) {
  if (value == null) return null;
  if (value is! String || !_mealPhotoReferencePattern.hasMatch(value)) {
    throw FormatException('$field must be a safe app-owned photo ID.');
  }
  return value;
}

double? _nullableNumber(Object? value, String field) =>
    value == null ? null : _number(value, field);

double _number(Object? value, String field, {bool positive = false}) {
  if (value is! num || !value.isFinite || (positive ? value <= 0 : value < 0)) {
    throw FormatException(
      '$field must be a finite ${positive ? 'positive' : 'nonnegative'} number.',
    );
  }
  return value.toDouble();
}

String _string(Object? value, String field) {
  if (value is! String || value.trim().isEmpty) {
    throw FormatException('$field must be a nonempty string.');
  }
  return value;
}

bool _hasExplicitTimezone(String value) =>
    value.endsWith('Z') || RegExp(r'[+-]\d{2}:\d{2}$').hasMatch(value);

void _requireExactFields(
  Map<String, dynamic> json,
  Set<String> required,
  Set<String> allowed,
  String field,
) {
  final missing = required.difference(json.keys.toSet());
  final extra = json.keys.toSet().difference(allowed);
  if (missing.isNotEmpty || extra.isNotEmpty) {
    throw FormatException(
      '$field fields invalid: missing=${missing.toList()..sort()}, '
      'unexpected=${extra.toList()..sort()}',
    );
  }
}
