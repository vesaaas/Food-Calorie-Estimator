import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/app.dart';

void main() {
  Future<void> launchApp(WidgetTester tester) async {
    await tester.pumpWidget(const FoodCalorieEstimatorApp());
    await tester.pumpAndSettle();
  }

  testWidgets('app launches on the Scan screen', (tester) async {
    await launchApp(tester);

    expect(find.text('Food Calorie Estimator'), findsOneWidget);
    expect(find.text('Take Photo'), findsOneWidget);
    expect(find.text('Choose from Gallery'), findsOneWidget);
    expect(find.byType(NavigationBar), findsOneWidget);
  });

  testWidgets('bottom navigation reaches Diary, Dashboard, and Goal', (
    tester,
  ) async {
    await launchApp(tester);

    await tester.tap(find.text('Diary').last);
    await tester.pumpAndSettle();
    expect(find.text('Your diary is ready'), findsOneWidget);
    expect(find.text('No meals logged yet'), findsOneWidget);

    await tester.tap(find.text('Dashboard').last);
    await tester.pumpAndSettle();
    expect(find.text('Calories consumed'), findsOneWidget);
    expect(find.text('Daily calorie goal'), findsOneWidget);
    expect(find.text('Remaining calories'), findsOneWidget);
    expect(find.text('Weekly calorie chart'), findsOneWidget);

    await tester.tap(find.text('Goal').last);
    await tester.pumpAndSettle();
    expect(find.text('Set a personal calorie target.'), findsOneWidget);
    expect(find.byKey(const Key('daily-goal-field')), findsOneWidget);
  });

  testWidgets('photo actions show safe placeholder messages', (tester) async {
    await launchApp(tester);

    await tester.tap(find.text('Take Photo'));
    await tester.pumpAndSettle();
    expect(find.text('Camera integration coming next.'), findsOneWidget);

    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    expect(find.text('Gallery integration coming next.'), findsOneWidget);
  });

  testWidgets('goal form validates positive numeric input locally', (
    tester,
  ) async {
    await launchApp(tester);
    await tester.tap(find.text('Goal').last);
    await tester.pumpAndSettle();

    await tester.tap(find.text('Save goal'));
    await tester.pumpAndSettle();
    expect(find.text('Enter a number greater than zero'), findsOneWidget);

    await tester.enterText(find.byKey(const Key('daily-goal-field')), '2000');
    await tester.tap(find.text('Save goal'));
    await tester.pumpAndSettle();
    expect(
      find.text(
        'Goal entry is valid. Saving will be available in a later phase.',
      ),
      findsOneWidget,
    );
  });
}
