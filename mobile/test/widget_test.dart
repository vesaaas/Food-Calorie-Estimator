import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:food_calorie_estimator/app.dart';
import 'package:food_calorie_estimator/services/image_acquisition_service.dart';
import 'package:food_calorie_estimator/services/food_classification_service.dart';
import 'package:food_calorie_estimator/services/nutrition_service.dart';
import 'package:food_calorie_estimator/services/diary_controller.dart';
import 'package:food_calorie_estimator/repositories/diary_repository.dart';
import 'package:food_calorie_estimator/models/diary_state.dart';
import 'package:food_calorie_estimator/models/meal_entry.dart';
import 'package:food_calorie_estimator/services/meal_photo_store.dart';

void main() {
  late Directory imageDirectory;
  late XFile testImage;
  late NutritionCatalog testNutritionCatalog;
  late MemoryDiaryRepository diaryRepository;
  late DiaryController diaryController;

  setUpAll(() async {
    imageDirectory = await Directory.systemTemp.createTemp('meal_photo_test');
    final catalogJson = await File('assets/nutrition/foods.json')
        .readAsString();
    testNutritionCatalog = NutritionCatalog.fromJsonString(catalogJson);
    final imageFile = File('${imageDirectory.path}/meal.png');
    await imageFile.writeAsBytes(_onePixelPng);
    testImage = XFile(imageFile.path);
  });

  tearDownAll(() async {
    await imageDirectory.delete(recursive: true);
  });

  Future<void> launchApp(
    WidgetTester tester, {
    ImageAcquisitionService? imageAcquisitionService,
    FoodClassificationService? foodClassificationService,
    FoodNutritionService? foodNutritionService,
    MealPhotoStore? mealPhotoStore,
    DiaryState? initialDiaryState,
  }) async {
    diaryRepository = MemoryDiaryRepository(initialDiaryState);
    diaryController = DiaryController(
      repository: diaryRepository,
      photoStore: mealPhotoStore ?? FakeMealPhotoStore(testImage.path),
      clock: () => DateTime(2026, 10, 22, 13),
      mealIdFactory: () => 'widget-meal',
    );
    await tester.pumpWidget(
      FoodCalorieEstimatorApp(
        imageAcquisitionService: imageAcquisitionService,
        foodClassificationService:
            foodClassificationService ?? FakeFoodClassificationService(),
        foodNutritionService:
            foodNutritionService ??
            CatalogBackedFakeNutritionService(testNutritionCatalog),
        diaryController: diaryController,
      ),
    );
    await tester.pumpAndSettle();
  }

  Future<void> pumpUntilVisible(WidgetTester tester, Finder finder) async {
    for (var attempt = 0; attempt < 100; attempt++) {
      await tester.pump(const Duration(milliseconds: 50));
      if (finder.evaluate().isNotEmpty) return;
    }
    final visibleTexts = tester
        .widgetList<Text>(find.byType(Text))
        .map((text) => text.data)
        .whereType<String>()
        .toList();
    fail('Timed out waiting for $finder. Texts: $visibleTexts');
  }

  Future<void> scrollToTop(WidgetTester tester) async {
    await tester.drag(find.byType(ListView).first, const Offset(0, 1000));
    await tester.pump(const Duration(milliseconds: 100));
  }

  testWidgets('app launches on the Scan screen', (tester) async {
    await launchApp(tester);

    expect(find.text('Food Calorie Estimator'), findsOneWidget);
    expect(find.text('Scan one of the supported foods'), findsOneWidget);
    expect(find.text('Take Photo'), findsOneWidget);
    expect(find.text('Choose from Gallery'), findsOneWidget);
    expect(find.byType(NavigationBar), findsOneWidget);
  });

  testWidgets('supported foods list shows the classifier classes', (
    tester,
  ) async {
    await launchApp(tester);

    await tester.tap(find.byKey(const Key('view-supported-foods')));
    await tester.pumpAndSettle();
    expect(find.text('Supported foods'), findsOneWidget);
    expect(find.text('Pizza'), findsOneWidget);
    expect(find.text('Hamburger'), findsOneWidget);
    expect(find.text('Ice Cream'), findsOneWidget);
    await tester.scrollUntilVisible(
      find.text('French Fries'),
      160,
      scrollable: find.descendant(
        of: find.byType(AlertDialog),
        matching: find.byType(Scrollable),
      ),
    );
    expect(find.text('French Fries'), findsOneWidget);
    await tester.scrollUntilVisible(
      find.text('Sarma'),
      160,
      scrollable: find.descendant(
        of: find.byType(AlertDialog),
        matching: find.byType(Scrollable),
      ),
    );
    expect(find.text('Sarma'), findsOneWidget);
  });

  testWidgets('bottom navigation reaches Diary, Dashboard, and Goal', (
    tester,
  ) async {
    await launchApp(tester);

    await tester.tap(find.text('Diary').last);
    await tester.pumpAndSettle();
    expect(find.text('Meal diary'), findsOneWidget);
    expect(find.text('No meals logged today'), findsOneWidget);

    await tester.tap(find.text('Dashboard').last);
    await tester.pumpAndSettle();
    expect(find.text('Calories consumed'), findsOneWidget);
    expect(find.byKey(const Key('dashboard-no-goal')), findsOneWidget);
    await tester.tap(find.text('Weekly'));
    await tester.pumpAndSettle();
    expect(find.text('This week'), findsOneWidget);

    await tester.tap(find.text('Goal').last);
    await tester.pumpAndSettle();
    expect(find.text('Set a personal calorie target.'), findsOneWidget);
    expect(find.byKey(const Key('daily-goal-field')), findsOneWidget);
  });

  testWidgets('Dashboard reacts to meal add, removal, and goal updates', (
    tester,
  ) async {
    await launchApp(tester);
    await tester.tap(find.text('Dashboard').last);
    await tester.pumpAndSettle();
    expect(
      find.text(
        'Set a daily calorie goal on the Goal screen to track progress.',
      ),
      findsOneWidget,
    );
    expect(
      find.text('No meals logged today. Your totals are zero.'),
      findsOneWidget,
    );

    final estimate = testNutritionCatalog.estimate('pizza', grams: 100);
    final meal = await diaryController.addMeal(estimate);
    await tester.pumpAndSettle();
    expect(
      find.text('${meal.caloriesKcal.toStringAsFixed(1)} kcal'),
      findsOneWidget,
    );
    expect(find.text('1'), findsOneWidget);

    await diaryController.setDailyCalorieGoal(meal.caloriesKcal / 2);
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('dashboard-goal-exceeded')), findsOneWidget);
    expect(find.text('0.0 kcal remaining'), findsNothing);

    await diaryController.setDailyCalorieGoal(meal.caloriesKcal + 100);
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('dashboard-remaining')), findsOneWidget);
    await diaryController.removeMeal(meal.mealId);
    await tester.pumpAndSettle();
    expect(find.text('0.0 kcal'), findsOneWidget);
    expect(
      find.text('No meals logged today. Your totals are zero.'),
      findsOneWidget,
    );
    expect(find.byKey(const Key('dashboard-daily-goal')), findsOneWidget);
  });

  testWidgets('Dashboard weekly selection renders all seven daily bars', (
    tester,
  ) async {
    await launchApp(tester);
    await tester.tap(find.text('Dashboard').last);
    await tester.pumpAndSettle();
    await tester.tap(find.text('Weekly'));
    await tester.pumpAndSettle();

    expect(find.text('This week'), findsOneWidget);
    expect(find.byKey(const Key('dashboard-weekly-total')), findsOneWidget);
    expect(find.byKey(const Key('dashboard-weekly-average')), findsOneWidget);
    expect(
      find.byKey(const Key('dashboard-weekly-meal-count')),
      findsOneWidget,
    );
    expect(find.byKey(const Key('dashboard-chart-scale')), findsOneWidget);
    expect(
      find.byKey(const Key('dashboard-day-bar-2026-10-16')),
      findsOneWidget,
    );
    expect(
      find.byKey(const Key('dashboard-day-bar-2026-10-22')),
      findsOneWidget,
    );
    expect(
      find.byKey(const Key('dashboard-weekday-2026-10-22')),
      findsOneWidget,
    );
  });

  testWidgets('Dashboard caps progress and fits very large calorie totals', (
    tester,
  ) async {
    final largeMeal = MealEntry(
      mealId: 'large-meal',
      foodId: 'pizza',
      displayName: 'Pizza',
      loggedAt: DateTime.utc(2026, 10, 22, 10),
      portionGrams: 100,
      portionMode: 'grams',
      caloriesKcal: 1e30,
      proteinGrams: 0,
      carbohydratesGrams: 0,
      fatGrams: 0,
    );
    await launchApp(
      tester,
      initialDiaryState: DiaryState(meals: [largeMeal], dailyCalorieGoal: 1),
    );
    await tester.tap(find.text('Dashboard').last);
    await tester.pumpAndSettle();

    final progress = tester.widget<LinearProgressIndicator>(
      find.byKey(const Key('dashboard-goal-progress')),
    );
    expect(progress.value, 1);
    expect(find.byKey(const Key('dashboard-goal-exceeded')), findsOneWidget);
    expect(tester.takeException(), isNull);

    await tester.tap(find.text('Weekly'));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('dashboard-chart-scale')), findsOneWidget);
    expect(tester.takeException(), isNull);
  });

  testWidgets('Take Photo invokes camera and previews the selected image', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(cameraResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);

    await tester.ensureVisible(find.text('Take Photo'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Take Photo'));
    await tester.pumpAndSettle();
    expect(picker.cameraCalls, 1);
    expect(picker.galleryCalls, 0);
    expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
    expect(find.text('Photo ready for analysis'), findsOneWidget);
    expect(find.text('Predicted food: pizza'), findsOneWidget);
    expect(find.text('Confidence: 91.0%'), findsOneWidget);
    expect(find.byKey(const Key('nutrition-details')), findsOneWidget);
  });

  testWidgets('Choose from Gallery invokes gallery and previews the image', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await scrollToTop(tester);
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));
    expect(picker.galleryCalls, 1);
    expect(picker.cameraCalls, 0);
    expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
    expect(find.text('Photo ready for analysis'), findsOneWidget);
    expect(find.text('Predicted food: pizza'), findsOneWidget);
    expect(
      find.text('Default serving: 1 piece, large pizza (132 g)'),
      findsOneWidget,
    );
    expect(find.byKey(const Key('value-Calories')), findsOneWidget);
  });

  testWidgets('low-confidence prediction is rejected without nutrition', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    final nutrition = CountingFoodNutritionService(testNutritionCatalog);
    await launchApp(
      tester,
      imageAcquisitionService: picker,
      foodClassificationService: FakeFoodClassificationService([
        const FoodClassification(food: 'donuts', confidence: 0.549),
      ]),
      foodNutritionService: nutrition,
    );

    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await pumpUntilVisible(
      tester,
      find.byKey(const Key('low-confidence-rejection')),
    );
    expect(
      find.text(
        'Unable to confidently identify this food. Please try another photo.',
      ),
      findsOneWidget,
    );
    expect(find.text('Predicted food: donuts'), findsNothing);
    expect(find.byKey(const Key('nutrition-details')), findsNothing);
    expect(find.byKey(const Key('add-meal-button')), findsNothing);
    expect(nutrition.calls, 0);
    expect(find.text('Take Photo'), findsOneWidget);
    expect(find.text('Choose from Gallery'), findsOneWidget);
  });

  testWidgets('threshold prediction is accepted and rejected scan can retry', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    final nutrition = CountingFoodNutritionService(testNutritionCatalog);
    final classifier = FakeFoodClassificationService([
      const FoodClassification(food: 'donuts', confidence: 0.54),
      const FoodClassification(food: 'pizza', confidence: 0.55),
    ]);
    await launchApp(
      tester,
      imageAcquisitionService: picker,
      foodClassificationService: classifier,
      foodNutritionService: nutrition,
    );

    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await pumpUntilVisible(
      tester,
      find.byKey(const Key('low-confidence-rejection')),
    );
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));
    expect(find.text('Predicted food: pizza'), findsOneWidget);
    expect(nutrition.calls, 1);
    expect(find.byKey(const Key('add-meal-button')), findsOneWidget);
  });

  testWidgets('Reset Scan clears scan state and preserves Diary and Goal', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    final savedMeal = MealEntry(
      mealId: 'existing-meal',
      foodId: 'sushi',
      displayName: 'Sushi',
      loggedAt: DateTime.utc(2026, 10, 22, 10),
      portionGrams: 150,
      portionMode: 'default',
      caloriesKcal: 210,
      proteinGrams: 12,
      carbohydratesGrams: 28,
      fatGrams: 5,
    );
    final initialState = DiaryState(meals: [savedMeal], dailyCalorieGoal: 1850);
    await launchApp(
      tester,
      imageAcquisitionService: picker,
      initialDiaryState: initialState,
    );
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await scrollToTop(tester);
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));
    expect(find.byKey(const Key('reset-scan-button')), findsOneWidget);

    await tester.tap(find.byKey(const Key('reset-scan-button')));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('selected-meal-image')), findsNothing);
    expect(find.text('Predicted food: pizza'), findsNothing);
    expect(find.byKey(const Key('nutrition-details')), findsNothing);
    expect(find.byKey(const Key('portion-grams')), findsNothing);
    expect(find.byKey(const Key('low-confidence-rejection')), findsNothing);
    expect(find.byKey(const Key('reset-scan-button')), findsNothing);
    expect(find.text('Your meal photo will appear here'), findsOneWidget);
    expect(diaryRepository.saved.meals, [savedMeal]);
    expect(diaryRepository.saved.dailyCalorieGoal, 1850);
  });

  testWidgets('Reset Scan clears a low-confidence result', (tester) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(
      tester,
      imageAcquisitionService: picker,
      foodClassificationService: FakeFoodClassificationService([
        const FoodClassification(food: 'donuts', confidence: 0.54),
      ]),
    );
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await pumpUntilVisible(
      tester,
      find.byKey(const Key('low-confidence-rejection')),
    );

    await scrollToTop(tester);
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('reset-scan-button')));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('low-confidence-rejection')), findsNothing);
    expect(find.byKey(const Key('selected-meal-image')), findsNothing);
    expect(find.byKey(const Key('reset-scan-button')), findsNothing);
    expect(find.text('Your meal photo will appear here'), findsOneWidget);
  });

  testWidgets(
    'manual food selection loads default nutrition and saves without photo',
    (tester) async {
      final picker = FakeImageAcquisitionService();
      final classifier = FakeFoodClassificationService();
      await launchApp(
        tester,
        imageAcquisitionService: picker,
        foodClassificationService: classifier,
      );

      await tester.ensureVisible(find.byKey(const Key('add-manually-button')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('add-manually-button')));
      await tester.pumpAndSettle();
      expect(find.text('Choose a supported food'), findsNWidgets(2));
      for (final food in foodClassNames) {
        final option = find.byKey(Key('manual-food-$food'));
        await tester.scrollUntilVisible(
          option,
          160,
          scrollable: find.descendant(
            of: find.byType(BottomSheet),
            matching: find.byType(Scrollable),
          ),
        );
        expect(option, findsOneWidget);
      }
      expect(find.byKey(const Key('manual-food-cake')), findsNothing);
      await tester.scrollUntilVisible(
        find.byKey(const Key('manual-food-french_fries')),
        160,
        scrollable: find.descendant(
          of: find.byType(BottomSheet),
          matching: find.byType(Scrollable),
        ),
      );
      await tester.tap(find.byKey(const Key('manual-food-french_fries')));
      await tester.pumpAndSettle();
      await pumpUntilVisible(
        tester,
        find.byKey(const Key('nutrition-details')),
      );

      expect(find.text('Selected food: French Fries'), findsOneWidget);
      expect(
        find.text('Default serving: 1 small fast food order (110 g)'),
        findsOneWidget,
      );
      expect(find.text('317.9 kcal'), findsOneWidget);
      expect(find.text('Predicted food: french_fries'), findsNothing);
      expect(find.textContaining('Confidence:'), findsNothing);
      expect(classifier.calls, 0);
      expect(picker.cameraCalls, 0);
      expect(picker.galleryCalls, 0);

      await tester.ensureVisible(find.byKey(const Key('portion-grams')));
      await tester.pumpAndSettle();
      await tester.enterText(find.byKey(const Key('portion-grams')), '200');
      await tester.pumpAndSettle();
      expect(find.text('578.0 kcal'), findsOneWidget);
      await tester.ensureVisible(find.byKey(const Key('add-meal-button')));
      await tester.pumpAndSettle();
      await tester.tap(find.byKey(const Key('add-meal-button')));
      await tester.pumpAndSettle();

      final savedMeal = diaryRepository.saved.meals.single;
      expect(savedMeal.foodId, 'french_fries');
      expect(savedMeal.displayName, 'French fries');
      expect(savedMeal.portionGrams, 200);
      expect(savedMeal.caloriesKcal, 578);
      expect(savedMeal.classifierConfidence, isNull);
      expect(savedMeal.photoReference, isNull);
      expect(find.byKey(const Key('manual-food-selection')), findsNothing);
      expect(find.byKey(const Key('nutrition-details')), findsNothing);
      expect(find.text('Your meal photo will appear here'), findsOneWidget);

      await tester.tap(find.text('Diary').last);
      await tester.pumpAndSettle();
      expect(
        find.byKey(const Key('meal-thumbnail-placeholder-widget-meal')),
        findsOneWidget,
      );
      await tester.tap(
        find.byKey(const Key('meal-thumbnail-placeholder-widget-meal')),
      );
      await tester.pumpAndSettle();
      expect(find.text('French fries'), findsNWidgets(2));
      expect(
        find.byKey(const Key('meal-detail-photo-placeholder')),
        findsOneWidget,
      );
    },
  );

  testWidgets('manual Reset Scan clears a manual selection', (tester) async {
    await launchApp(tester);
    await tester.ensureVisible(find.byKey(const Key('add-manually-button')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('add-manually-button')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('manual-food-pizza')));
    await tester.pumpAndSettle();
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));

    await scrollToTop(tester);
    await tester.ensureVisible(find.byKey(const Key('reset-scan-button')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('reset-scan-button')));
    await tester.pumpAndSettle();

    expect(find.byKey(const Key('manual-food-selection')), findsNothing);
    expect(find.byKey(const Key('nutrition-details')), findsNothing);
    expect(find.byKey(const Key('portion-grams')), findsNothing);
    expect(find.byKey(const Key('reset-scan-button')), findsNothing);
    expect(diaryRepository.saved.meals, isEmpty);
  });

  testWidgets(
    'cancelled acquisition preserves the previous result without lookup',
    (tester) async {
      final picker = FakeImageAcquisitionService(
        galleryResult: testImage,
        cameraResult: null,
      );
      final nutrition = CountingFoodNutritionService(testNutritionCatalog);
      await launchApp(
        tester,
        imageAcquisitionService: picker,
        foodNutritionService: nutrition,
      );

      await tester.ensureVisible(find.text('Choose from Gallery'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Choose from Gallery'));
      await scrollToTop(tester);
      await pumpUntilVisible(
        tester,
        find.byKey(const Key('nutrition-details')),
      );
      expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
      expect(find.text('Predicted food: pizza'), findsOneWidget);
      expect(nutrition.calls, 1);

      await tester.ensureVisible(find.text('Take Photo'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Take Photo'));
      await tester.pumpAndSettle();
      expect(picker.cameraCalls, 1);
      expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
      expect(find.text('Photo ready for analysis'), findsOneWidget);
      expect(find.text('Predicted food: pizza'), findsOneWidget);
      expect(
        find.text('Default serving: 1 piece, large pizza (132 g)'),
        findsOneWidget,
      );
      expect(nutrition.calls, 1);
    },
  );

  testWidgets('editing grams recalculates calories and macros', (tester) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await scrollToTop(tester);
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));

    await tester.ensureVisible(find.byKey(const Key('portion-grams')));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const Key('portion-grams')), '180');
    await tester.pumpAndSettle();
    expect(find.text('Edited portion: 180 g'), findsOneWidget);
    expect(find.text('487.8 kcal'), findsOneWidget);
    expect(find.text('19.4 g'), findsOneWidget);
    expect(find.text('59.8 g'), findsOneWidget);
    expect(find.text('18.9 g'), findsOneWidget);
  });

  testWidgets('new prediction replaces food and default nutrition', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(
      galleryResult: testImage,
      cameraResult: testImage,
    );
    final classifier = FakeFoodClassificationService([
      const FoodClassification(food: 'pizza', confidence: 0.6),
      const FoodClassification(food: 'french_fries', confidence: 0.999),
    ]);
    await launchApp(
      tester,
      imageAcquisitionService: picker,
      foodClassificationService: classifier,
    );

    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await scrollToTop(tester);
    await pumpUntilVisible(
      tester,
      find.text('Default serving: 1 piece, large pizza (132 g)'),
    );
    expect(
      find.text('Default serving: 1 piece, large pizza (132 g)'),
      findsOneWidget,
    );

    await tester.ensureVisible(find.text('Take Photo'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Take Photo'));
    await scrollToTop(tester);
    await pumpUntilVisible(
      tester,
      find.text('Default serving: 1 small fast food order (110 g)'),
    );
    expect(find.text('Predicted food: french fries'), findsOneWidget);
    expect(
      find.text('Default serving: 1 small fast food order (110 g)'),
      findsOneWidget,
    );
    expect(find.text('317.9 kcal'), findsOneWidget);
    expect(find.text('Edited portion: 132 g'), findsNothing);
  });

  testWidgets('picker errors show a message without crashing', (tester) async {
    final picker = FakeImageAcquisitionService(
      cameraError: Exception('failed'),
    );
    await launchApp(tester, imageAcquisitionService: picker);

    await tester.ensureVisible(find.text('Take Photo'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Take Photo'));
    await tester.pumpAndSettle();
    expect(
      find.text('Could not capture the photo. Please try again.'),
      findsOneWidget,
    );
    expect(find.byType(NavigationBar), findsOneWidget);
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
    expect(find.text('Daily goal saved.'), findsOneWidget);
    expect(diaryRepository.saved.dailyCalorieGoal, 2000);
  });

  testWidgets('tapping outside the Scan grams field dismisses keyboard', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await scrollToTop(tester);
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));

    final field = find.byKey(const Key('portion-grams'));
    await tester.ensureVisible(field);
    await tester.showKeyboard(field);
    await tester.enterText(field, '200');
    await tester.pumpAndSettle();
    final editable = tester.widget<EditableText>(
      find.descendant(of: field, matching: find.byType(EditableText)),
    );
    expect(editable.focusNode.hasFocus, isTrue);

    await tester.tapAt(const Offset(12, 12));
    await tester.pumpAndSettle();
    expect(editable.focusNode.hasFocus, isFalse);
    expect(tester.widget<TextField>(field).controller!.text, '200');
  });

  testWidgets('tapping outside the Goal field dismisses keyboard', (
    tester,
  ) async {
    await launchApp(tester);
    await tester.tap(find.text('Goal').last);
    await tester.pumpAndSettle();

    final field = find.byKey(const Key('daily-goal-field'));
    await tester.showKeyboard(field);
    await tester.enterText(field, '2100');
    await tester.pumpAndSettle();
    final editable = tester.widget<EditableText>(
      find.descendant(of: field, matching: find.byType(EditableText)),
    );
    expect(editable.focusNode.hasFocus, isTrue);

    await tester.tapAt(const Offset(12, 12));
    await tester.pumpAndSettle();
    expect(editable.focusNode.hasFocus, isFalse);
    expect(tester.widget<TextFormField>(field).controller!.text, '2100');
  });

  testWidgets('Scan Add Meal snapshot appears in Diary totals', (tester) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await scrollToTop(tester);
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));

    await tester.ensureVisible(find.byKey(const Key('portion-grams')));
    await tester.pumpAndSettle();
    await tester.enterText(find.byKey(const Key('portion-grams')), '200');
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const Key('add-meal-button')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('add-meal-button')));
    await tester.pumpAndSettle();
    expect(find.text('Pizza added to diary.'), findsOneWidget);
    expect(find.byKey(const Key('selected-meal-image')), findsNothing);
    expect(find.text('Predicted food: pizza'), findsNothing);
    expect(find.byKey(const Key('nutrition-details')), findsNothing);
    expect(find.text('Your meal photo will appear here'), findsOneWidget);
    expect(diaryRepository.saved.meals.single.photoReference, isNotNull);

    await tester.tap(find.text('Diary').last);
    await tester.pumpAndSettle();
    expect(find.text('Pizza'), findsOneWidget);
    expect(
      find.byKey(const Key('meal-thumbnail-photo-widget-meal')),
      findsOneWidget,
    );
    expect(find.text('Portion: 200 g'), findsOneWidget);
    expect(find.text('542.0 kcal'), findsNWidgets(2));
    await tester.tap(find.byKey(const Key('meal-thumbnail-photo-widget-meal')));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('meal-detail-photo')), findsOneWidget);
    expect(find.text('Logged'), findsOneWidget);
    expect(find.text('Classifier confidence'), findsOneWidget);
    expect(find.text('200 g'), findsOneWidget);
    expect(find.text('542.0 kcal'), findsNWidgets(3));
    expect(find.text('Protein'), findsNWidgets(2));
    expect(find.text('Carbohydrates'), findsNWidgets(2));
    expect(find.text('Fat'), findsNWidgets(2));
  });

  testWidgets('failed Add Meal keeps the current Scan result intact', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await pumpUntilVisible(tester, find.byKey(const Key('nutrition-details')));
    diaryRepository.failNextSave = true;

    await tester.ensureVisible(find.byKey(const Key('add-meal-button')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('add-meal-button')));
    await tester.pumpAndSettle();
    expect(
      find.text('Could not save this meal. Please try again.'),
      findsOneWidget,
    );
    expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
    expect(find.text('Predicted food: pizza'), findsOneWidget);
    expect(find.byKey(const Key('nutrition-details')), findsOneWidget);
    expect(diaryRepository.saved.meals, isEmpty);
  });

  testWidgets('Diary meal without a saved photo shows a placeholder', (
    tester,
  ) async {
    final legacyMeal = MealEntry(
      mealId: 'old-meal',
      foodId: 'pizza',
      displayName: 'Pizza',
      loggedAt: DateTime.utc(2026, 10, 22, 11),
      portionGrams: 132,
      portionMode: 'default',
      caloriesKcal: 355.08,
      proteinGrams: 15.048,
      carbohydratesGrams: 41.844,
      fatGrams: 13.332,
    );
    await launchApp(tester, initialDiaryState: DiaryState(meals: [legacyMeal]));
    await tester.tap(find.text('Diary').last);
    await tester.pumpAndSettle();
    await tester.ensureVisible(find.byKey(const Key('open-meal-old-meal')));
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('open-meal-old-meal')));
    await tester.pumpAndSettle();
    expect(
      find.byKey(const Key('meal-detail-photo-placeholder')),
      findsOneWidget,
    );
    expect(find.byKey(const Key('meal-detail-name')), findsOneWidget);
    expect(find.text('Calories'), findsNWidgets(2));
    expect(find.text('132 g'), findsOneWidget);
  });

  testWidgets('Diary meal with a missing saved photo shows a placeholder', (
    tester,
  ) async {
    final missingPhotoMeal = MealEntry(
      mealId: 'missing-photo-meal',
      foodId: 'pizza',
      displayName: 'Pizza',
      loggedAt: DateTime.utc(2026, 10, 22, 11),
      portionGrams: 132,
      portionMode: 'default',
      caloriesKcal: 355.08,
      proteinGrams: 15.048,
      carbohydratesGrams: 41.844,
      fatGrams: 13.332,
      photoReference: '${'a' * 32}.png',
    );
    await launchApp(
      tester,
      initialDiaryState: DiaryState(meals: [missingPhotoMeal]),
    );
    await tester.tap(find.text('Diary').last);
    await tester.pumpAndSettle();
    await tester.ensureVisible(
      find.byKey(const Key('open-meal-missing-photo-meal')),
    );
    await tester.pumpAndSettle();
    await tester.tap(find.byKey(const Key('open-meal-missing-photo-meal')));
    await tester.pumpAndSettle();

    expect(
      find.byKey(const Key('meal-detail-photo-placeholder')),
      findsOneWidget,
    );
    expect(find.byKey(const Key('meal-detail-name')), findsOneWidget);
  });
}

class MemoryDiaryRepository implements DiaryRepository {
  MemoryDiaryRepository([DiaryState? initialState])
    : saved = initialState ?? DiaryState.empty;

  DiaryState saved;
  bool failNextSave = false;

  @override
  Future<DiaryState> load() async => saved;

  @override
  Future<void> save(DiaryState state) async {
    if (failNextSave) {
      failNextSave = false;
      throw StateError('Synthetic diary save failure');
    }
    saved = state;
  }
}

class FakeMealPhotoStore implements MealPhotoStore {
  FakeMealPhotoStore(this.resolvedPath);

  final String? resolvedPath;
  String? savedReference;
  final deletedReferences = <String>[];

  @override
  Future<String> savePhoto(String sourcePath) async {
    final extension = sourcePath.toLowerCase().endsWith('.png') ? 'png' : 'jpg';
    savedReference = '${'a' * 32}.$extension';
    return savedReference!;
  }

  @override
  Future<String?> resolvePhotoPath(String? photoReference) async =>
      photoReference != null && photoReference == savedReference
      ? resolvedPath
      : null;

  @override
  Future<void> deletePhoto(String? photoReference) async {
    if (photoReference != null) deletedReferences.add(photoReference);
    if (photoReference == savedReference) savedReference = null;
  }

  @override
  Future<void> removeUnreferencedPhotos(Set<String> referencedPhotos) async {
    if (savedReference != null && !referencedPhotos.contains(savedReference)) {
      savedReference = null;
    }
  }
}

class FakeImageAcquisitionService implements ImageAcquisitionService {
  FakeImageAcquisitionService({
    this.cameraResult,
    this.galleryResult,
    this.cameraError,
    this.galleryError,
  });

  final XFile? cameraResult;
  final XFile? galleryResult;
  final Exception? cameraError;
  final Exception? galleryError;
  int cameraCalls = 0;
  int galleryCalls = 0;

  @override
  Future<XFile?> takePhoto() async {
    cameraCalls++;
    if (cameraError case final error?) throw error;
    return cameraResult;
  }

  @override
  Future<XFile?> chooseFromGallery() async {
    galleryCalls++;
    if (galleryError case final error?) throw error;
    return galleryResult;
  }
}

class FakeFoodClassificationService implements FoodClassificationService {
  FakeFoodClassificationService([List<FoodClassification>? predictions])
    : predictions =
          predictions ??
          [const FoodClassification(food: 'pizza', confidence: 0.91)];

  final List<FoodClassification> predictions;
  int calls = 0;

  @override
  Future<FoodClassification> classifyImage(String imagePath) async =>
      predictions[(calls++).clamp(0, predictions.length - 1)];
}

class CountingFoodNutritionService implements FoodNutritionService {
  CountingFoodNutritionService(this.catalog);

  final NutritionCatalog catalog;
  int calls = 0;

  @override
  Future<NutritionEstimate> estimate(
    String foodId, {
    double? grams,
    double? servings,
  }) {
    calls++;
    return Future.value(
      catalog.estimate(foodId, grams: grams, servings: servings),
    );
  }
}

class CatalogBackedFakeNutritionService implements FoodNutritionService {
  const CatalogBackedFakeNutritionService(this.catalog);

  final NutritionCatalog catalog;

  @override
  Future<NutritionEstimate> estimate(
    String foodId, {
    double? grams,
    double? servings,
  }) async => catalog.estimate(foodId, grams: grams, servings: servings);
}

final _onePixelPng = <int>[
  137,
  80,
  78,
  71,
  13,
  10,
  26,
  10,
  0,
  0,
  0,
  13,
  73,
  72,
  68,
  82,
  0,
  0,
  0,
  1,
  0,
  0,
  0,
  1,
  8,
  6,
  0,
  0,
  0,
  31,
  21,
  196,
  137,
  0,
  0,
  0,
  13,
  73,
  68,
  65,
  84,
  8,
  215,
  99,
  248,
  207,
  192,
  240,
  31,
  0,
  5,
  0,
  1,
  255,
  137,
  153,
  61,
  29,
  0,
  0,
  0,
  0,
  73,
  69,
  78,
  68,
  174,
  66,
  96,
  130,
];
