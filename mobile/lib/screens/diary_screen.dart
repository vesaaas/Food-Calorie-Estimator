import 'package:flutter/material.dart';

import '../widgets/screen_header.dart';

class DiaryScreen extends StatelessWidget {
  const DiaryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
      children: [
        const ScreenHeader(
          title: 'Meal diary',
          subtitle: 'Your meals, all in one place.',
        ),
        const SizedBox(height: 30),
        Card(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 42),
            child: Column(
              children: [
                Icon(
                  Icons.menu_book_outlined,
                  size: 52,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(height: 16),
                Text(
                  'Your diary is ready',
                  style: Theme.of(context).textTheme.titleLarge
                      ?.copyWith(fontWeight: FontWeight.w700),
                ),
                const SizedBox(height: 8),
                Text(
                  'Meals you add will appear here with their portions and nutrition estimates.',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium
                      ?.copyWith(height: 1.45),
                ),
                const SizedBox(height: 14),
                Text(
                  'No meals logged yet',
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: Theme.of(context).colorScheme.onSurfaceVariant,
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
