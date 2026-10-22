import 'package:flutter/material.dart';

import 'dart:io';

import 'package:image_picker/image_picker.dart';

import '../services/image_acquisition_service.dart';
import '../services/food_classification_service.dart';
import '../services/nutrition_service.dart';
import '../theme/app_theme.dart';
import '../widgets/screen_header.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({
    super.key,
    this.imageAcquisitionService,
    this.foodClassificationService,
    this.foodNutritionService,
  });

  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;
  final FoodNutritionService? foodNutritionService;

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  late final ImageAcquisitionService _imageAcquisitionService;
  late final FoodClassificationService _foodClassificationService;
  late final FoodNutritionService _foodNutritionService;
  final TextEditingController _portionGramsController = TextEditingController();
  XFile? _selectedImage;
  bool _isPickingImage = false;
  bool _isClassifying = false;
  bool _isLoadingNutrition = false;
  bool _portionInputInvalid = false;
  FoodClassification? _classification;
  NutritionEstimate? _nutrition;
  int _nutritionRequest = 0;

  @override
  void initState() {
    super.initState();
    _imageAcquisitionService =
        widget.imageAcquisitionService ?? ImagePickerAcquisitionService();
    _foodClassificationService =
        widget.foodClassificationService ?? OnDeviceFoodClassificationService();
    _foodNutritionService =
        widget.foodNutritionService ?? AssetFoodNutritionService();
  }

  @override
  void dispose() {
    _portionGramsController.dispose();
    super.dispose();
  }

  Future<void> _pickImage({required bool useCamera}) async {
    if (_isPickingImage) return;
    setState(() => _isPickingImage = true);
    try {
      final image = useCamera
          ? await _imageAcquisitionService.takePhoto()
          : await _imageAcquisitionService.chooseFromGallery();
      if (!mounted) return;
      if (image != null) {
        _nutritionRequest++;
        setState(() {
          _selectedImage = image;
          _classification = null;
          _nutrition = null;
          _portionGramsController.clear();
          _portionInputInvalid = false;
          _isLoadingNutrition = false;
          _isClassifying = true;
        });
        try {
          final result = await _foodClassificationService.classifyImage(
            image.path,
          );
          if (!mounted) return;
          final request = ++_nutritionRequest;
          setState(() {
            _classification = result;
            _isClassifying = false;
            _isLoadingNutrition = true;
          });
          try {
            final estimate = await _foodNutritionService.estimate(result.food);
            if (!mounted || request != _nutritionRequest) return;
            setState(() {
              _nutrition = estimate;
              _portionGramsController.text = _formatGrams(
                estimate.portionGrams,
              );
              _isLoadingNutrition = false;
            });
          } on Object {
            if (!mounted) return;
            setState(() => _isLoadingNutrition = false);
            ScaffoldMessenger.of(context)
              ..hideCurrentSnackBar()
              ..showSnackBar(
                const SnackBar(
                  content: Text('Nutrition information could not be loaded.'),
                ),
              );
          }
        } on Object {
          if (!mounted) return;
          ScaffoldMessenger.of(context)
            ..hideCurrentSnackBar()
            ..showSnackBar(
              const SnackBar(
                content: Text(
                  'Could not analyze this photo. Please try another image.',
                ),
              ),
            );
        } finally {
          if (mounted) setState(() => _isClassifying = false);
        }
      }
    } on Exception {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(
            content: Text(
              useCamera
                  ? 'Could not capture the photo. Please try again.'
                  : 'Could not select the image. Please try again.',
            ),
          ),
        );
    } finally {
      if (mounted) setState(() => _isPickingImage = false);
    }
  }

  Future<void> _updatePortion(String value) async {
    final grams = double.tryParse(value);
    if (grams == null || !grams.isFinite || grams <= 0) {
      _nutritionRequest++;
      setState(() => _portionInputInvalid = true);
      return;
    }
    final classification = _classification;
    if (classification == null) return;
    final request = ++_nutritionRequest;
    setState(() => _portionInputInvalid = false);
    try {
      final estimate = await _foodNutritionService.estimate(
        classification.food,
        grams: grams,
      );
      if (!mounted || request != _nutritionRequest) return;
      setState(() => _nutrition = estimate);
    } on Object {
      if (!mounted || request != _nutritionRequest) return;
      setState(() => _portionInputInvalid = true);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
      children: [
        const ScreenHeader(
          title: 'Food Calorie Estimator',
          subtitle: 'A clearer picture of what is on your plate.',
        ),
        const SizedBox(height: 26),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(22),
            child: Column(
              children: [
                Container(
                  height: 210,
                  width: double.infinity,
                  decoration: BoxDecoration(
                    color: const Color(0xFFEAF3ED),
                    borderRadius: BorderRadius.circular(18),
                  ),
                  clipBehavior: Clip.antiAlias,
                  child: _selectedImage == null
                      ? const Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.ramen_dining_outlined,
                              size: 54,
                              color: AppTheme.green,
                            ),
                            SizedBox(height: 12),
                            Text(
                              'Your meal photo will appear here',
                              style: TextStyle(fontWeight: FontWeight.w600),
                            ),
                          ],
                        )
                      : Image.file(
                          File(_selectedImage!.path),
                          key: const Key('selected-meal-image'),
                          width: double.infinity,
                          height: 210,
                          fit: BoxFit.cover,
                          errorBuilder: (context, error, stackTrace) =>
                              const Center(
                                child: Text(
                                  'This image could not be displayed.',
                                ),
                              ),
                        ),
                ),
                if (_selectedImage != null) ...[
                  const SizedBox(height: 10),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.check_circle_outline,
                        size: 18,
                        color: Theme.of(context).colorScheme.primary,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        'Photo ready for analysis',
                        style: Theme.of(context).textTheme.labelLarge?.copyWith(
                          color: Theme.of(context).colorScheme.primary,
                        ),
                      ),
                    ],
                  ),
                ],
                if (_isClassifying) ...[
                  const SizedBox(height: 14),
                  const CircularProgressIndicator(
                    key: Key('classification-loading'),
                  ),
                  const SizedBox(height: 8),
                  const Text('Analyzing photo…'),
                ],
                if (_classification case final result?) ...[
                  const SizedBox(height: 14),
                  Text(
                    'Predicted food: ${result.food.replaceAll('_', ' ')}',
                    key: const Key('food-prediction'),
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  Text(
                    'Confidence: ${(result.confidence * 100).toStringAsFixed(1)}%',
                  ),
                ],
                if (_isLoadingNutrition) ...[
                  const SizedBox(height: 12),
                  const LinearProgressIndicator(key: Key('nutrition-loading')),
                  const SizedBox(height: 8),
                  const Text('Loading nutrition…'),
                ],
                if (_nutrition case final estimate?) ...[
                  const SizedBox(height: 18),
                  _NutritionDetails(
                    estimate: estimate,
                    portionGramsController: _portionGramsController,
                    portionInputInvalid: _portionInputInvalid,
                    onPortionChanged: _updatePortion,
                  ),
                ],
                const SizedBox(height: 20),
                Text(
                  'Nutrition is estimated from a representative food and your selected portion. Photo weight is not measured.',
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.bodyMedium
                      ?.copyWith(height: 1.45),
                ),
                const SizedBox(height: 20),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _isPickingImage
                        ? null
                        : () => _pickImage(useCamera: true),
                    icon: const Icon(Icons.photo_camera_outlined),
                    label: const Text('Take Photo'),
                  ),
                ),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    onPressed: _isPickingImage
                        ? null
                        : () => _pickImage(useCamera: false),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Choose from Gallery'),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 18),
        const _InfoNote(),
      ],
    );
  }
}

String _formatGrams(double grams) {
  if (grams == grams.roundToDouble()) return grams.toStringAsFixed(0);
  return grams.toString();
}

class _NutritionDetails extends StatelessWidget {
  const _NutritionDetails({
    required this.estimate,
    required this.portionGramsController,
    required this.portionInputInvalid,
    required this.onPortionChanged,
  });

  final NutritionEstimate estimate;
  final TextEditingController portionGramsController;
  final bool portionInputInvalid;
  final ValueChanged<String> onPortionChanged;

  @override
  Widget build(BuildContext context) {
    final defaultServing = estimate.food.defaultServing;
    final nutrients = estimate.nutrients;
    final portionDescription = estimate.portionMode == 'default'
        ? 'Default serving'
        : 'Edited portion';
    return Container(
      key: const Key('nutrition-details'),
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFEAF3ED),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Nutrition estimate',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 5),
          Text(
            'Default serving: ${defaultServing.label} (${_formatGrams(defaultServing.grams)} g)',
          ),
          const SizedBox(height: 12),
          TextField(
            key: const Key('portion-grams'),
            controller: portionGramsController,
            onChanged: onPortionChanged,
            keyboardType: const TextInputType.numberWithOptions(decimal: true),
            decoration: InputDecoration(
              labelText: 'Current portion (g)',
              helperText:
                  '$portionDescription: ${_formatGrams(estimate.portionGrams)} g',
              errorText: portionInputInvalid
                  ? 'Enter a finite weight greater than zero.'
                  : null,
              isDense: true,
              filled: true,
              fillColor: Colors.white,
            ),
          ),
          const SizedBox(height: 14),
          _NutritionLine(
            key: const Key('nutrition-calories'),
            label: 'Calories',
            value: '${nutrients.energyKcal.toStringAsFixed(1)} kcal',
          ),
          _NutritionLine(
            key: const Key('nutrition-protein'),
            label: 'Protein',
            value: '${nutrients.proteinGrams.toStringAsFixed(1)} g',
          ),
          _NutritionLine(
            key: const Key('nutrition-carbohydrates'),
            label: 'Carbohydrates',
            value: '${nutrients.carbohydratesGrams.toStringAsFixed(1)} g',
          ),
          _NutritionLine(
            key: const Key('nutrition-fat'),
            label: 'Fat',
            value: '${nutrients.fatGrams.toStringAsFixed(1)} g',
          ),
          const SizedBox(height: 8),
          Text(
            estimate.food.mappingNotes,
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}

class _NutritionLine extends StatelessWidget {
  const _NutritionLine({super.key, required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 3),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(value, key: Key('value-$label')),
        ],
      ),
    );
  }
}

class _InfoNote extends StatelessWidget {
  const _InfoNote();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.7),
        borderRadius: BorderRadius.circular(18),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            Icons.info_outline,
            color: Theme.of(context).colorScheme.primary,
          ),
          const SizedBox(width: 12),
          const Expanded(
            child: Text(
              'Meal logging, personal goals, and progress charts are not connected yet.',
            ),
          ),
        ],
      ),
    );
  }
}
