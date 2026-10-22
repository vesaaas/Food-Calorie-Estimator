import 'dart:io';

import 'package:flutter/material.dart';

import '../models/meal_entry.dart';
import '../services/diary_controller.dart';
import '../services/meal_photo_store.dart';
import '../widgets/screen_header.dart';

class DiaryScreen extends StatelessWidget {
  const DiaryScreen({super.key, required this.diaryController});

  final DiaryController diaryController;

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: diaryController,
      builder: (context, _) {
        final today = diaryController.today;
        final meals = diaryController.mealsForDay(today);
        final totals = diaryController.totalsForDay(today);
        return ListView(
          padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
          children: [
            const ScreenHeader(
              title: 'Meal diary',
              subtitle: 'Your meals, all in one place.',
            ),
            const SizedBox(height: 20),
            if (!diaryController.isInitialized) ...[
              _DiaryLoadState(controller: diaryController),
            ] else ...[
              _DailyTotalsCard(totals: totals),
              const SizedBox(height: 18),
              Text(
                'Today’s meals',
                style: Theme.of(context).textTheme.titleLarge
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 10),
              if (meals.isEmpty)
                const Card(
                  child: Padding(
                    padding: EdgeInsets.all(22),
                    child: Center(child: Text('No meals logged today')),
                  ),
                )
              else
                for (final meal in meals)
                  _MealCard(
                    meal: meal,
                    photoStore: diaryController.photoStore,
                    onTap: () => _showMealDetails(context, meal),
                    onRemove: () => _removeMeal(context, meal),
                  ),
            ],
          ],
        );
      },
    );
  }

  Future<void> _showMealDetails(BuildContext context, MealEntry meal) =>
      showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        builder: (context) =>
            MealDetailSheet(meal: meal, photoStore: diaryController.photoStore),
      );

  Future<void> _removeMeal(BuildContext context, MealEntry meal) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Remove meal?'),
        content: Text('Remove ${meal.displayName} from today’s diary?'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            key: const Key('confirm-remove-meal'),
            onPressed: () => Navigator.pop(context, true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );
    if (confirmed != true || !context.mounted) return;
    try {
      await diaryController.removeMeal(meal.mealId);
    } on Object {
      if (!context.mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Could not remove this meal. Please try again.'),
        ),
      );
    }
  }
}

class _DiaryLoadState extends StatelessWidget {
  const _DiaryLoadState({required this.controller});

  final DiaryController controller;

  @override
  Widget build(BuildContext context) {
    if (controller.isLoading) {
      return const Padding(
        padding: EdgeInsets.all(28),
        child: Center(child: CircularProgressIndicator()),
      );
    }
    if (controller.loadError != null) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            children: [
              const Text('Saved diary data could not be loaded.'),
              const SizedBox(height: 12),
              OutlinedButton(
                onPressed: controller.initialize,
                child: const Text('Try again'),
              ),
            ],
          ),
        ),
      );
    }
    return const Padding(
      padding: EdgeInsets.all(28),
      child: Center(child: CircularProgressIndicator()),
    );
  }
}

class _DailyTotalsCard extends StatelessWidget {
  const _DailyTotalsCard({required this.totals});

  final DailyNutritionTotals totals;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Today', style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 12),
            _TotalLine(
              key: const Key('diary-total-calories'),
              label: 'Calories',
              value: '${totals.caloriesKcal.toStringAsFixed(1)} kcal',
            ),
            _TotalLine(
              key: const Key('diary-total-protein'),
              label: 'Protein',
              value: '${totals.proteinGrams.toStringAsFixed(1)} g',
            ),
            _TotalLine(
              key: const Key('diary-total-carbohydrates'),
              label: 'Carbohydrates',
              value: '${totals.carbohydratesGrams.toStringAsFixed(1)} g',
            ),
            _TotalLine(
              key: const Key('diary-total-fat'),
              label: 'Fat',
              value: '${totals.fatGrams.toStringAsFixed(1)} g',
            ),
          ],
        ),
      ),
    );
  }
}

class _TotalLine extends StatelessWidget {
  const _TotalLine({super.key, required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 3),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [Text(label), Text(value)],
    ),
  );
}

class _MealCard extends StatelessWidget {
  const _MealCard({
    required this.meal,
    required this.photoStore,
    required this.onTap,
    required this.onRemove,
  });

  final MealEntry meal;
  final MealPhotoStore photoStore;
  final VoidCallback onTap;
  final VoidCallback onRemove;

  @override
  Widget build(BuildContext context) {
    final time = meal.loggedAt.toLocal();
    final timeLabel =
        '${time.hour.toString().padLeft(2, '0')}:'
        '${time.minute.toString().padLeft(2, '0')}';
    return Card(
      key: ValueKey('meal-${meal.mealId}'),
      child: Row(
        children: [
          Expanded(
            child: InkWell(
              key: Key('open-meal-${meal.mealId}'),
              onTap: onTap,
              borderRadius: BorderRadius.circular(12),
              child: Padding(
                padding: const EdgeInsets.fromLTRB(12, 12, 8, 12),
                child: Row(
                  children: [
                    _MealThumbnail(meal: meal, photoStore: photoStore),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Expanded(
                                child: Text(
                                  meal.displayName,
                                  style: Theme.of(context).textTheme.titleMedium
                                      ?.copyWith(fontWeight: FontWeight.w700),
                                ),
                              ),
                              Text(timeLabel),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text('Portion: ${_formatGrams(meal.portionGrams)} g'),
                          const SizedBox(height: 4),
                          Text('${meal.caloriesKcal.toStringAsFixed(1)} kcal'),
                          const SizedBox(height: 2),
                          Text(
                            'Protein ${meal.proteinGrams.toStringAsFixed(1)} g  ·  '
                            'Carbs ${meal.carbohydratesGrams.toStringAsFixed(1)} g  ·  '
                            'Fat ${meal.fatGrams.toStringAsFixed(1)} g',
                            style: Theme.of(context).textTheme.bodySmall,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          IconButton(
            key: Key('remove-meal-${meal.mealId}'),
            tooltip: 'Remove meal',
            onPressed: onRemove,
            icon: const Icon(Icons.delete_outline),
          ),
        ],
      ),
    );
  }
}

class _MealThumbnail extends StatelessWidget {
  const _MealThumbnail({required this.meal, required this.photoStore});

  final MealEntry meal;
  final MealPhotoStore photoStore;

  @override
  Widget build(BuildContext context) => ClipRRect(
    borderRadius: BorderRadius.circular(10),
    child: SizedBox(
      key: Key('meal-thumbnail-${meal.mealId}'),
      width: 60,
      height: 60,
      child: FutureBuilder<String?>(
        future: photoStore.resolvePhotoPath(meal.photoReference),
        builder: (context, snapshot) {
          final path = snapshot.data;
          if (snapshot.connectionState == ConnectionState.waiting ||
              path == null) {
            return _MealThumbnailPlaceholder(mealId: meal.mealId);
          }
          return Image.file(
            File(path),
            key: Key('meal-thumbnail-photo-${meal.mealId}'),
            cacheWidth: 180,
            fit: BoxFit.cover,
            errorBuilder: (_, _, _) =>
                _MealThumbnailPlaceholder(mealId: meal.mealId),
          );
        },
      ),
    ),
  );
}

class _MealThumbnailPlaceholder extends StatelessWidget {
  const _MealThumbnailPlaceholder({required this.mealId});

  final String mealId;

  @override
  Widget build(BuildContext context) => Container(
    key: Key('meal-thumbnail-placeholder-$mealId'),
    color: const Color(0xFFEAF3ED),
    alignment: Alignment.center,
    child: Icon(
      Icons.ramen_dining_outlined,
      size: 28,
      color: Theme.of(context).colorScheme.primary,
    ),
  );
}

class MealDetailSheet extends StatelessWidget {
  const MealDetailSheet({
    super.key,
    required this.meal,
    required this.photoStore,
  });

  final MealEntry meal;
  final MealPhotoStore photoStore;

  @override
  Widget build(BuildContext context) {
    final loggedAt = meal.loggedAt.toLocal();
    final dateLabel =
        '${loggedAt.year.toString().padLeft(4, '0')}-'
        '${loggedAt.month.toString().padLeft(2, '0')}-'
        '${loggedAt.day.toString().padLeft(2, '0')}  '
        '${loggedAt.hour.toString().padLeft(2, '0')}:'
        '${loggedAt.minute.toString().padLeft(2, '0')}';
    return SafeArea(
      child: ConstrainedBox(
        constraints: BoxConstraints(
          maxHeight: MediaQuery.sizeOf(context).height * 0.9,
        ),
        child: SingleChildScrollView(
          padding: const EdgeInsets.fromLTRB(22, 4, 22, 26),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _MealPhoto(
                photoStore: photoStore,
                photoReference: meal.photoReference,
              ),
              const SizedBox(height: 18),
              Text(
                meal.displayName,
                key: const Key('meal-detail-name'),
                style: Theme.of(context).textTheme.headlineSmall
                    ?.copyWith(fontWeight: FontWeight.w700),
              ),
              const SizedBox(height: 8),
              _DetailLine('Portion', '${_formatGrams(meal.portionGrams)} g'),
              _DetailLine(
                'Calories',
                '${meal.caloriesKcal.toStringAsFixed(1)} kcal',
              ),
              _DetailLine(
                'Protein',
                '${meal.proteinGrams.toStringAsFixed(1)} g',
              ),
              _DetailLine(
                'Carbohydrates',
                '${meal.carbohydratesGrams.toStringAsFixed(1)} g',
              ),
              _DetailLine('Fat', '${meal.fatGrams.toStringAsFixed(1)} g'),
              _DetailLine('Logged', dateLabel),
              if (meal.classifierConfidence case final confidence?)
                _DetailLine(
                  'Classifier confidence',
                  '${(confidence * 100).toStringAsFixed(1)}%',
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _MealPhoto extends StatelessWidget {
  const _MealPhoto({required this.photoStore, required this.photoReference});

  final MealPhotoStore photoStore;
  final String? photoReference;

  @override
  Widget build(BuildContext context) => ClipRRect(
    borderRadius: BorderRadius.circular(18),
    child: SizedBox(
      height: 220,
      width: double.infinity,
      child: FutureBuilder<String?>(
        future: photoStore.resolvePhotoPath(photoReference),
        builder: (context, snapshot) {
          final path = snapshot.data;
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (path == null) return const _MissingMealPhoto();
          return Image.file(
            File(path),
            key: const Key('meal-detail-photo'),
            fit: BoxFit.cover,
            errorBuilder: (_, _, _) => const _MissingMealPhoto(),
          );
        },
      ),
    ),
  );
}

class _MissingMealPhoto extends StatelessWidget {
  const _MissingMealPhoto();

  @override
  Widget build(BuildContext context) => Container(
    key: const Key('meal-detail-photo-placeholder'),
    color: const Color(0xFFEAF3ED),
    child: const Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Icon(Icons.ramen_dining_outlined, size: 48),
        SizedBox(height: 8),
        Text('Meal photo unavailable'),
      ],
    ),
  );
}

class _DetailLine extends StatelessWidget {
  const _DetailLine(this.label, this.value);

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => Padding(
    padding: const EdgeInsets.symmetric(vertical: 7),
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label),
        const SizedBox(width: 12),
        Flexible(child: Text(value, textAlign: TextAlign.end)),
      ],
    ),
  );
}

String _formatGrams(double grams) =>
    grams == grams.roundToDouble() ? grams.toStringAsFixed(0) : '$grams';
