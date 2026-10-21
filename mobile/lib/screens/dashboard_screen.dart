import 'package:flutter/material.dart';

import '../widgets/screen_header.dart';

class DashboardScreen extends StatelessWidget {
  const DashboardScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
      children: [
        const ScreenHeader(
          title: 'Dashboard',
          subtitle: 'A simple view of your food tracking.',
        ),
        const SizedBox(height: 24),
        Text(
          'Today',
          style: Theme.of(context).textTheme.titleLarge
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 12),
        const _DailyPlaceholderCard(),
        const SizedBox(height: 22),
        Text(
          'This week',
          style: Theme.of(context).textTheme.titleLarge
              ?.copyWith(fontWeight: FontWeight.w700),
        ),
        const SizedBox(height: 12),
        const _WeeklyPlaceholderCard(),
      ],
    );
  }
}

class _DailyPlaceholderCard extends StatelessWidget {
  const _DailyPlaceholderCard();

  @override
  Widget build(BuildContext context) {
    const labels = [
      'Calories consumed',
      'Daily calorie goal',
      'Remaining calories',
    ];
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          children: [
            for (var i = 0; i < labels.length; i++) ...[
              if (i > 0) const Divider(height: 22),
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    labels[i],
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  Text(
                    '— kcal',
                    style: Theme.of(context).textTheme.titleMedium
                        ?.copyWith(fontWeight: FontWeight.w700),
                  ),
                ],
              ),
            ],
            const SizedBox(height: 14),
            Text(
              'No logged data yet',
              style: Theme.of(context).textTheme.labelMedium?.copyWith(
                color: Theme.of(context).colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _WeeklyPlaceholderCard extends StatelessWidget {
  const _WeeklyPlaceholderCard();

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          children: [
            Icon(
              Icons.bar_chart_rounded,
              size: 42,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 10),
            Text(
              'Weekly calorie chart',
              style: Theme.of(context).textTheme.titleMedium
                  ?.copyWith(fontWeight: FontWeight.w700),
            ),
            const SizedBox(height: 6),
            Text(
              'A seven-day summary will appear when meal tracking is connected.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium,
            ),
            const SizedBox(height: 16),
            Container(
              height: 78,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.surfaceContainerLow,
                borderRadius: BorderRadius.circular(14),
              ),
              alignment: Alignment.center,
              child: Text(
                'No logged data yet',
                style: Theme.of(context).textTheme.labelMedium?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
