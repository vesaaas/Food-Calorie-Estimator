import 'package:flutter/material.dart';

import 'services/image_acquisition_service.dart';
import 'services/food_classification_service.dart';
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
  });

  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Food Calorie Estimator',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      home: MainNavigationScreen(
        imageAcquisitionService: imageAcquisitionService,
        foodClassificationService: foodClassificationService,
      ),
    );
  }
}

class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({
    super.key,
    this.imageAcquisitionService,
    this.foodClassificationService,
  });

  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _selectedIndex = 0;

  late final List<Widget> _screens = [
    ScanScreen(
      imageAcquisitionService: widget.imageAcquisitionService,
      foodClassificationService: widget.foodClassificationService,
    ),
    const DiaryScreen(),
    const DashboardScreen(),
    const GoalScreen(),
  ];

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
