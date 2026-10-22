import 'package:flutter/material.dart';

import 'dart:io';

import 'package:image_picker/image_picker.dart';

import '../services/image_acquisition_service.dart';
import '../services/food_classification_service.dart';
import '../theme/app_theme.dart';
import '../widgets/screen_header.dart';

class ScanScreen extends StatefulWidget {
  const ScanScreen({
    super.key,
    this.imageAcquisitionService,
    this.foodClassificationService,
  });

  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  late final ImageAcquisitionService _imageAcquisitionService;
  late final FoodClassificationService _foodClassificationService;
  XFile? _selectedImage;
  bool _isPickingImage = false;
  bool _isClassifying = false;
  FoodClassification? _classification;

  @override
  void initState() {
    super.initState();
    _imageAcquisitionService =
        widget.imageAcquisitionService ?? ImagePickerAcquisitionService();
    _foodClassificationService =
        widget.foodClassificationService ?? OnDeviceFoodClassificationService();
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
        setState(() {
          _selectedImage = image;
          _classification = null;
          _isClassifying = true;
        });
        try {
          final result = await _foodClassificationService.classifyImage(
            image.path,
          );
          if (mounted) setState(() => _classification = result);
        } on Exception {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
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
                const SizedBox(height: 20),
                Text(
                  'Choose a meal photo to identify the food. Nutrition estimates will be connected in a later phase.',
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
              'Photo analysis, portion review, and meal logging are not connected yet.',
            ),
          ),
        ],
      ),
    );
  }
}
