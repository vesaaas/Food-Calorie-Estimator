import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:food_calorie_estimator/app.dart';
import 'package:food_calorie_estimator/services/image_acquisition_service.dart';
import 'package:food_calorie_estimator/services/food_classification_service.dart';

void main() {
  late Directory imageDirectory;
  late XFile testImage;

  setUpAll(() async {
    imageDirectory = await Directory.systemTemp.createTemp('meal_photo_test');
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
  }) async {
    await tester.pumpWidget(
      FoodCalorieEstimatorApp(
        imageAcquisitionService: imageAcquisitionService,
        foodClassificationService: FakeFoodClassificationService(),
      ),
    );
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
  });

  testWidgets('Choose from Gallery invokes gallery and previews the image', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(galleryResult: testImage);
    await launchApp(tester, imageAcquisitionService: picker);
    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    expect(picker.galleryCalls, 1);
    expect(picker.cameraCalls, 0);
    expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
    expect(find.text('Photo ready for analysis'), findsOneWidget);
  });

  testWidgets('cancelling acquisition keeps the previous image', (
    tester,
  ) async {
    final picker = FakeImageAcquisitionService(
      galleryResult: testImage,
      cameraResult: null,
    );
    await launchApp(tester, imageAcquisitionService: picker);

    await tester.ensureVisible(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Choose from Gallery'));
    await tester.pumpAndSettle();
    expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);

    await tester.ensureVisible(find.text('Take Photo'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Take Photo'));
    await tester.pumpAndSettle();
    expect(picker.cameraCalls, 1);
    expect(find.byKey(const Key('selected-meal-image')), findsOneWidget);
    expect(find.text('Photo ready for analysis'), findsOneWidget);
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
  @override
  Future<FoodClassification> classifyImage(String imagePath) async =>
      const FoodClassification(food: 'pizza', confidence: 0.91);
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
