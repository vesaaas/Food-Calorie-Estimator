import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:food_calorie_estimator/app.dart';
import 'package:food_calorie_estimator/services/image_acquisition_service.dart';
import 'package:food_calorie_estimator/services/food_classification_service.dart';
import 'package:food_calorie_estimator/services/nutrition_service.dart';

void main() {
  late Directory imageDirectory;
  late XFile testImage;
  late NutritionCatalog testNutritionCatalog;

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
  }) async {
    await tester.pumpWidget(
      FoodCalorieEstimatorApp(
        imageAcquisitionService: imageAcquisitionService,
        foodClassificationService:
            foodClassificationService ?? FakeFoodClassificationService(),
        foodNutritionService:
            foodNutritionService ??
            CatalogBackedFakeNutritionService(testNutritionCatalog),
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

  testWidgets('Take Photo invokes camera and previews the selected image', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(cameraResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);

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
    expect(
      find.text(
        'Goal entry is valid. Saving will be available in a later phase.',
      ),
      findsOneWidget,
    );
  });
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
