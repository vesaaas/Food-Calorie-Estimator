import 'dart:math' as math;

import 'package:flutter/material.dart';

import '../models/dashboard_summary.dart';
import '../services/diary_controller.dart';
import '../services/dashboard_summary_service.dart';
import '../widgets/screen_header.dart';

enum _DashboardPeriod { today, week }

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({required this.diaryController, super.key});

  final DiaryController diaryController;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  _DashboardPeriod _period = _DashboardPeriod.today;

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.diaryController,
      builder: (context, _) {
        final controller = widget.diaryController;
        return ListView(
          keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
          padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
          children: [
            const ScreenHeader(
              title: 'Dashboard',
              subtitle: 'Your food tracking at a glance.',
            ),
            const SizedBox(height: 22),
            SegmentedButton<_DashboardPeriod>(
              segments: const [
                ButtonSegment(
                  value: _DashboardPeriod.today,
                  label: Text('Today'),
                  icon: Icon(Icons.today_outlined),
                ),
                ButtonSegment(
                  value: _DashboardPeriod.week,
                  label: Text('Weekly'),
                  icon: Icon(Icons.bar_chart_rounded),
                ),
              ],
              selected: {_period},
              onSelectionChanged: (selection) =>
                  setState(() => _period = selection.first),
            ),
            const SizedBox(height: 20),
            if (controller.isLoading || !controller.isInitialized)
              _LoadingCard(
                hasError: controller.loadError != null,
                onRetry: controller.initialize,
              )
            else ...[
              Builder(
                builder: (context) {
                  final summary = DashboardSummaryService.summarize(
                    controller,
                    today: controller.today,
                  );
                  return _period == _DashboardPeriod.today
                      ? _TodayContent(summary: summary)
                      : _WeeklyContent(summary: summary);
                },
              ),
            ],
          ],
        );
      },
    );
  }
}

class _LoadingCard extends StatelessWidget {
  const _LoadingCard({required this.hasError, required this.onRetry});

  final bool hasError;
  final Future<void> Function() onRetry;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            if (!hasError) const CircularProgressIndicator(),
            const SizedBox(height: 8),
            Text(
              hasError
                  ? 'Your saved dashboard data could not be loaded.'
                  : 'Loading your saved meals and goal…',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium
                  ?.copyWith(color: colors.onSurfaceVariant),
            ),
            if (hasError) ...[
              const SizedBox(height: 10),
              TextButton(onPressed: onRetry, child: const Text('Try again')),
            ],
          ],
        ),
      ),
    );
  }
}

class _TodayContent extends StatelessWidget {
  const _TodayContent({required this.summary});

  final DashboardSummary summary;

  @override
  Widget build(BuildContext context) {
    final totals = summary.dailyTotals;
    final colors = Theme.of(context).colorScheme;
    final goal = summary.dailyGoal;
    final progress = summary.progress;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          _fullDate(summary.today),
          style: Theme.of(context).textTheme.titleMedium
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 10),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Calories consumed',
                  style: Theme.of(context).textTheme.titleSmall,
                ),
                const SizedBox(height: 3),
                FittedBox(
                  fit: BoxFit.scaleDown,
                  alignment: Alignment.centerLeft,
                  child: Text(
                    '${_number(totals.caloriesKcal)} kcal',
                    key: const Key('dashboard-today-calories'),
                    style: Theme.of(context).textTheme.headlineMedium
                        ?.copyWith(fontWeight: FontWeight.w800),
                  ),
                ),
                if (goal != null) ...[
                  const SizedBox(height: 16),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Daily goal'),
                      Flexible(
                        child: Text(
                          '${_number(goal)} kcal',
                          key: const Key('dashboard-daily-goal'),
                          textAlign: TextAlign.end,
                          style: const TextStyle(fontWeight: FontWeight.w700),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 9),
                  Semantics(
                    label: 'Progress toward daily calorie goal',
                    value: progress! >= 1
                        ? '100 percent or more'
                        : '${(progress * 100).round()} percent',
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: LinearProgressIndicator(
                        key: const Key('dashboard-goal-progress'),
                        value: progress.clamp(0, 1).toDouble(),
                        minHeight: 10,
                      ),
                    ),
                  ),
                  const SizedBox(height: 9),
                  if (summary.exceededCalories case final exceeded?)
                    Text(
                      'Goal exceeded by ${_number(exceeded)} kcal',
                      key: const Key('dashboard-goal-exceeded'),
                      style: TextStyle(
                        color: colors.error,
                        fontWeight: FontWeight.w600,
                      ),
                    )
                  else
                    Text(
                      '${_number(summary.remainingCalories!)} kcal remaining',
                      key: const Key('dashboard-remaining'),
                      style: TextStyle(color: colors.onSurfaceVariant),
                    ),
                ] else ...[
                  const SizedBox(height: 12),
                  Text(
                    'Set a daily calorie goal on the Goal screen to track progress.',
                    key: const Key('dashboard-no-goal'),
                    style: TextStyle(color: colors.onSurfaceVariant),
                  ),
                ],
                const Divider(height: 28),
                Wrap(
                  spacing: 18,
                  runSpacing: 14,
                  children: [
                    _Metric(
                      label: 'Protein',
                      value: '${_number(totals.proteinGrams)} g',
                    ),
                    _Metric(
                      label: 'Carbs',
                      value: '${_number(totals.carbohydratesGrams)} g',
                    ),
                    _Metric(
                      label: 'Fat',
                      value: '${_number(totals.fatGrams)} g',
                    ),
                    _Metric(label: 'Meals', value: '${totals.mealCount}'),
                  ],
                ),
              ],
            ),
          ),
        ),
        if (totals.mealCount == 0) ...[
          const SizedBox(height: 12),
          _EmptyNote(text: 'No meals logged today. Your totals are zero.'),
        ],
      ],
    );
  }
}

class _WeeklyContent extends StatelessWidget {
  const _WeeklyContent({required this.summary});

  final DashboardSummary summary;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'This week',
          style: Theme.of(context).textTheme.titleLarge
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 6),
        Text(
          '${_shortDate(summary.week.first.day)} – ${_shortDate(summary.week.last.day)}',
          style: Theme.of(context).textTheme.titleMedium
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 10),
        Card(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(14, 16, 14, 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Expanded(
                      child: _WeeklyMetric(
                        label: 'Weekly total',
                        value: '${_number(summary.weeklyCalories)} kcal',
                        testKey: 'dashboard-weekly-total',
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: _WeeklyMetric(
                        label: 'Daily average',
                        value: '${_number(summary.weeklyAverageCalories)} kcal',
                        testKey: 'dashboard-weekly-average',
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  '${summary.weeklyMealCount} meals in 7 days',
                  key: const Key('dashboard-weekly-meal-count'),
                  style: Theme.of(context).textTheme.bodySmall
                      ?.copyWith(color: colors.onSurfaceVariant),
                ),
                const SizedBox(height: 20),
                _WeeklyBarChart(days: summary.week),
              ],
            ),
          ),
        ),
        if (summary.weeklyMealCount == 0) ...[
          const SizedBox(height: 12),
          const _EmptyNote(text: 'No meals logged in the past 7 days.'),
        ],
      ],
    );
  }
}

class _WeeklyBarChart extends StatelessWidget {
  const _WeeklyBarChart({required this.days});

  final List<DashboardDaySummary> days;

  @override
  Widget build(BuildContext context) {
    final colors = Theme.of(context).colorScheme;
    final maxCalories = days.fold<double>(
      0,
      (maximum, day) => math.max(maximum, day.totals.caloriesKcal),
    );
    final scale = maxCalories == 0 ? 1.0 : maxCalories;

    return Column(
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text('0 kcal', style: Theme.of(context).textTheme.labelSmall),
            Text(
              '${_number(maxCalories)} kcal max',
              key: const Key('dashboard-chart-scale'),
              style: Theme.of(context).textTheme.labelSmall,
            ),
          ],
        ),
        const SizedBox(height: 6),
        SizedBox(
          height: 160,
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              for (final day in days)
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 2),
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.end,
                      children: [
                        SizedBox(
                          height: 22,
                          child: FittedBox(
                            fit: BoxFit.scaleDown,
                            child: Text(
                              _number(day.totals.caloriesKcal),
                              key: Key(
                                'dashboard-day-value-${_dateKey(day.day)}',
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 5),
                        Expanded(
                          child: Align(
                            alignment: Alignment.bottomCenter,
                            child: FractionallySizedBox(
                              heightFactor: maxCalories == 0
                                  ? 0
                                  : (day.totals.caloriesKcal / scale).clamp(
                                      0,
                                      1,
                                    ),
                              widthFactor: 0.62,
                              child: DecoratedBox(
                                key: Key(
                                  'dashboard-day-bar-${_dateKey(day.day)}',
                                ),
                                decoration: BoxDecoration(
                                  color: day.totals.caloriesKcal == 0
                                      ? colors.surfaceContainerHighest
                                      : colors.primary,
                                  borderRadius: const BorderRadius.vertical(
                                    top: Radius.circular(6),
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          _weekday(day.day),
                          key: Key('dashboard-weekday-${_dateKey(day.day)}'),
                          style: Theme.of(context).textTheme.labelSmall,
                        ),
                      ],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );
  }
}

class _Metric extends StatelessWidget {
  const _Metric({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) => SizedBox(
    width: 120,
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodySmall),
        const SizedBox(height: 3),
        Text(
          value,
          style: Theme.of(context).textTheme.titleSmall
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
      ],
    ),
  );
}

class _WeeklyMetric extends StatelessWidget {
  const _WeeklyMetric({
    required this.label,
    required this.value,
    required this.testKey,
  });

  final String label;
  final String value;
  final String testKey;

  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(label, style: Theme.of(context).textTheme.bodySmall),
      const SizedBox(height: 3),
      FittedBox(
        fit: BoxFit.scaleDown,
        alignment: Alignment.centerLeft,
        child: Text(
          value,
          key: Key(testKey),
          style: Theme.of(context).textTheme.titleMedium
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
      ),
    ],
  );
}

class _EmptyNote extends StatelessWidget {
  const _EmptyNote({required this.text});

  final String text;

  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(14),
    decoration: BoxDecoration(
      color: Theme.of(context).colorScheme.surfaceContainerLow,
      borderRadius: BorderRadius.circular(16),
    ),
    child: Text(
      text,
      textAlign: TextAlign.center,
      style: Theme.of(context).textTheme.bodyMedium,
    ),
  );
}

String _number(double value) => value.toStringAsFixed(1);

String _fullDate(DateTime day) =>
    '${_weekdayLong(day)}, ${day.month}/${day.day}/${day.year}';

String _shortDate(DateTime day) => '${day.month}/${day.day}';

String _weekday(DateTime day) =>
    const ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][day.weekday - 1];

String _weekdayLong(DateTime day) => const [
  'Monday',
  'Tuesday',
  'Wednesday',
  'Thursday',
  'Friday',
  'Saturday',
  'Sunday',
][day.weekday - 1];

String _dateKey(DateTime day) =>
    '${day.year.toString().padLeft(4, '0')}-'
    '${day.month.toString().padLeft(2, '0')}-'
    '${day.day.toString().padLeft(2, '0')}';
