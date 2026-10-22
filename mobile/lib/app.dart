import 'dart:async';

import 'package:flutter/material.dart';

import 'services/image_acquisition_service.dart';
import 'services/food_classification_service.dart';
import 'services/nutrition_service.dart';
import 'services/diary_controller.dart';
import 'services/meal_photo_store.dart';
import 'repositories/json_diary_repository.dart';
import 'screens/dashboard_screen.dart';
import 'screens/diary_screen.dart';
import 'screens/goal_screen.dart';
import 'screens/scan_screen.dart';
import 'theme/app_theme.dart';

class FoodCalorieEstimatorApp extends StatelessWidget {
  const FoodCalorieEstimatorApp({
    super.key,
    this.imageAcquisitionService,
    this.foodClassificationService,
    this.foodNutritionService,
    this.diaryController,
  });

  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;
  final FoodNutritionService? foodNutritionService;
  final DiaryController? diaryController;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Food Calorie Estimator',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      home: MainNavigationScreen(
        imageAcquisitionService: imageAcquisitionService,
        foodClassificationService: foodClassificationService,
        foodNutritionService: foodNutritionService,
        diaryController: diaryController,
      ),
    );
  }
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({
    super.key,
    this.imageAcquisitionService,
    this.foodClassificationService,
    this.foodNutritionService,
    this.diaryController,
  });

  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;
  final FoodNutritionService? foodNutritionService;
  final DiaryController? diaryController;

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _selectedIndex = 0;
  late final DiaryController _diaryController;
  late final bool _ownsDiaryController;
  late final List<Widget> _screens;

  @override
  void initState() {
    super.initState();
    _ownsDiaryController = widget.diaryController == null;
    _diaryController =
        widget.diaryController ??
        DiaryController(
          repository: JsonDiaryRepository(),
          photoStore: AppSupportMealPhotoStore(),
        );
    unawaited(_diaryController.initialize());
    _screens = [
      ScanScreen(
        imageAcquisitionService: widget.imageAcquisitionService,
        foodClassificationService: widget.foodClassificationService,
        foodNutritionService: widget.foodNutritionService,
        diaryController: _diaryController,
      ),
      DiaryScreen(diaryController: _diaryController),
      DashboardScreen(diaryController: _diaryController),
      GoalScreen(diaryController: _diaryController),
    ];
  }

  @override
  void dispose() {
    if (_ownsDiaryController) _diaryController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: IndexedStack(index: _selectedIndex, children: _screens),
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) =>
            setState(() => _selectedIndex = index),
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.document_scanner_outlined),
            selectedIcon: Icon(Icons.document_scanner),
            label: 'Scan',
          ),
          NavigationDestination(
            icon: Icon(Icons.menu_book_outlined),
            selectedIcon: Icon(Icons.menu_book),
            label: 'Diary',
          ),
          NavigationDestination(
            icon: Icon(Icons.insights_outlined),
            selectedIcon: Icon(Icons.insights),
            label: 'Dashboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.flag_outlined),
            selectedIcon: Icon(Icons.flag),
            label: 'Goal',
          ),
        ],
      ),
    );
  }
}
