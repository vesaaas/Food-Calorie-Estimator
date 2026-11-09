import 'package:flutter/material.dart';

import 'dart:io';

import 'package:image_picker/image_picker.dart';

import '../services/image_acquisition_service.dart';
import '../services/food_classification_service.dart';
import '../services/nutrition_service.dart';
import '../services/diary_controller.dart';
import '../theme/app_theme.dart';
import '../widgets/screen_header.dart';

const _unsupportedFood = '__food_not_listed__';

class ScanScreen extends StatefulWidget {
  const ScanScreen({
    super.key,
    this.isActive = true,
    this.imageAcquisitionService,
    this.foodClassificationService,
    this.foodNutritionService,
    this.diaryController,
  });

  final bool isActive;
  final ImageAcquisitionService? imageAcquisitionService;
  final FoodClassificationService? foodClassificationService;
  final FoodNutritionService? foodNutritionService;
  final DiaryController? diaryController;

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
  bool _isUpdatingPortion = false;
  bool _isSavingMeal = false;
  bool _portionInputInvalid = false;
  bool _isNonFoodRejected = false;
  bool _isAwaitingFoodConfirmation = false;
  bool _isPredictionDeclined = false;
  bool _isUnsupportedFood = false;
  FoodClassification? _classification;
  NutritionEstimate? _nutrition;
  String? _manualFoodId;
  int _nutritionRequest = 0;
  int _scanRequest = 0;

  bool get _hasScanState =>
      _selectedImage != null ||
      _classification != null ||
      _nutrition != null ||
      _manualFoodId != null ||
      _isClassifying ||
      _isLoadingNutrition ||
      _isNonFoodRejected ||
      _isAwaitingFoodConfirmation ||
      _isPredictionDeclined ||
      _isUnsupportedFood ||
      _portionInputInvalid;

  void _clearScanState() {
    _scanRequest++;
    _nutritionRequest++;
    FocusManager.instance.primaryFocus?.unfocus();
    ScaffoldMessenger.of(context).hideCurrentSnackBar();
    _selectedImage = null;
    _classification = null;
    _nutrition = null;
    _manualFoodId = null;
    _portionGramsController.clear();
    _portionInputInvalid = false;
    _isNonFoodRejected = false;
    _isAwaitingFoodConfirmation = false;
    _isPredictionDeclined = false;
    _isUnsupportedFood = false;
    _isClassifying = false;
    _isLoadingNutrition = false;
    _isUpdatingPortion = false;
    _isPickingImage = false;
  }

  void _resetScan() {
    setState(_clearScanState);
  }

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
  void didUpdateWidget(covariant ScanScreen oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.isActive && !widget.isActive) _clearScanState();
  }

  @override
  void dispose() {
    _portionGramsController.dispose();
    super.dispose();
  }

  Future<void> _pickImage({required bool useCamera}) async {
    if (_isPickingImage || !widget.isActive) return;
    final acquisitionRequest = _scanRequest;
    setState(() => _isPickingImage = true);
    try {
      final image = useCamera
          ? await _imageAcquisitionService.takePhoto()
          : await _imageAcquisitionService.chooseFromGallery();
      if (!mounted || !widget.isActive || acquisitionRequest != _scanRequest) {
        return;
      }
      if (image != null) {
        final scanRequest = ++_scanRequest;
        _nutritionRequest++;
        setState(() {
          _selectedImage = image;
          _classification = null;
          _nutrition = null;
          _manualFoodId = null;
          _portionGramsController.clear();
          _portionInputInvalid = false;
          _isNonFoodRejected = false;
          _isAwaitingFoodConfirmation = false;
          _isPredictionDeclined = false;
          _isUnsupportedFood = false;
          _isLoadingNutrition = false;
          _isUpdatingPortion = false;
          _isClassifying = true;
        });
        try {
          final result = await _foodClassificationService.classifyImage(
            image.path,
          );
          if (!mounted || scanRequest != _scanRequest) return;
          if (result.isNonFood) {
            setState(() {
              _isNonFoodRejected = true;
              _isClassifying = false;
            });
            return;
          }
          setState(() {
            _classification = result;
            _isClassifying = false;
            _isAwaitingFoodConfirmation = true;
          });
        } on Object {
          if (!mounted || scanRequest != _scanRequest) return;
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
          if (mounted && scanRequest == _scanRequest) {
            setState(() => _isClassifying = false);
          }
        }
      }
    } on Exception {
      if (!mounted || !widget.isActive || acquisitionRequest != _scanRequest) {
        return;
      }
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

  void _declineFoodSelection() {
    if (!_isAwaitingFoodConfirmation) return;
    _nutritionRequest++;
    setState(() {
      _classification = null;
      _manualFoodId = null;
      _nutrition = null;
      _portionGramsController.clear();
      _portionInputInvalid = false;
      _isNonFoodRejected = false;
      _isAwaitingFoodConfirmation = false;
      _isPredictionDeclined = true;
      _isUnsupportedFood = false;
      _isLoadingNutrition = false;
      _isUpdatingPortion = false;
    });
  }

  void _declineUnsupportedFood() {
    _scanRequest++;
    _nutritionRequest++;
    setState(() {
      _classification = null;
      _manualFoodId = null;
      _nutrition = null;
      _portionGramsController.clear();
      _portionInputInvalid = false;
      _isNonFoodRejected = false;
      _isAwaitingFoodConfirmation = false;
      _isPredictionDeclined = false;
      _isUnsupportedFood = true;
      _isLoadingNutrition = false;
      _isUpdatingPortion = false;
    });
  }

  Future<void> _confirmFoodSelection() async {
    if (!_isAwaitingFoodConfirmation || !widget.isActive) return;
    final foodId = _manualFoodId ?? _classification?.food;
    if (foodId == null || !foodClassNames.contains(foodId)) {
      _declineUnsupportedFood();
      return;
    }
    final scanRequest = _scanRequest;
    final nutritionRequest = ++_nutritionRequest;
    setState(() {
      _isAwaitingFoodConfirmation = false;
      _isLoadingNutrition = true;
      _portionInputInvalid = false;
    });
    try {
      final estimate = await _foodNutritionService.estimate(foodId);
      if (!mounted ||
          !widget.isActive ||
          scanRequest != _scanRequest ||
          nutritionRequest != _nutritionRequest) {
        return;
      }
      setState(() {
        _nutrition = estimate;
        _portionGramsController.text = _formatGrams(estimate.portionGrams);
        _isLoadingNutrition = false;
      });
    } on Object {
      if (!mounted ||
          !widget.isActive ||
          scanRequest != _scanRequest ||
          nutritionRequest != _nutritionRequest) {
        return;
      }
      setState(() {
        _isLoadingNutrition = false;
        _isAwaitingFoodConfirmation = true;
      });
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          const SnackBar(
            content: Text('Nutrition information could not be loaded.'),
          ),
        );
    }
  }

  Future<void> _updatePortion(String value) async {
    final grams = double.tryParse(value);
    if (grams == null || !grams.isFinite || grams <= 0) {
      _nutritionRequest++;
      setState(() {
        _portionInputInvalid = true;
        _isUpdatingPortion = false;
      });
      return;
    }
    final foodId = _manualFoodId ?? _classification?.food;
    if (foodId == null) return;
    final request = ++_nutritionRequest;
    setState(() {
      _portionInputInvalid = false;
      _isUpdatingPortion = true;
    });
    try {
      final estimate = await _foodNutritionService.estimate(
        foodId,
        grams: grams,
      );
      if (!mounted || request != _nutritionRequest) return;
      setState(() {
        _nutrition = estimate;
        _isUpdatingPortion = false;
      });
    } on Object {
      if (!mounted || request != _nutritionRequest) return;
      setState(() {
        _portionInputInvalid = true;
        _isUpdatingPortion = false;
      });
    }
  }

  bool get _canAddMeal =>
      !_isSavingMeal &&
      !_isUpdatingPortion &&
      !_portionInputInvalid &&
      (_manualFoodId != null ||
          (_classification != null &&
              !_classification!.isNonFood &&
              _classification!.food != null)) &&
      _nutrition != null &&
      widget.diaryController != null;

  Future<void> _addMeal() async {
    if (!_canAddMeal) return;
    final classification = _classification;
    final estimate = _nutrition!;
    setState(() => _isSavingMeal = true);
    try {
      await widget.diaryController!.addMeal(
        estimate,
        classifierConfidence: classification?.confidence,
        photoSourcePath: _selectedImage?.path,
      );
      if (!mounted) return;
      FocusManager.instance.primaryFocus?.unfocus();
      setState(() {
        _nutritionRequest++;
        _selectedImage = null;
        _classification = null;
        _nutrition = null;
        _manualFoodId = null;
        _portionGramsController.clear();
        _portionInputInvalid = false;
        _isNonFoodRejected = false;
        _isLoadingNutrition = false;
        _isUpdatingPortion = false;
      });
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(
            content: Text('${estimate.food.displayName} added to diary.'),
          ),
        );
    } on Object {
      if (!mounted) return;
      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          const SnackBar(
            content: Text('Could not save this meal. Please try again.'),
          ),
        );
    } finally {
      if (mounted) setState(() => _isSavingMeal = false);
    }
  }

  Future<void> _addManually() async {
    if (_isSavingMeal || _isPickingImage || !widget.isActive) return;
    final selectionRequest = _scanRequest;
    final foodId = await showModalBottomSheet<String>(
      context: context,
      isScrollControlled: true,
      showDragHandle: true,
      builder: (context) => SafeArea(
        child: SizedBox(
          height: MediaQuery.sizeOf(context).height * 0.65,
          child: Column(
            children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
                child: Text(
                  'Choose a supported food',
                  style: Theme.of(context).textTheme.titleLarge,
                ),
              ),
              Expanded(
                child: ListView(
                  children: [
                    ListTile(
                      key: const Key('manual-food-not-listed'),
                      leading: const Icon(Icons.help_outline),
                      title: const Text("My food isn't listed"),
                      onTap: () => Navigator.of(context).pop(_unsupportedFood),
                    ),
                    const Divider(),
                    for (final food in foodClassNames)
                      ListTile(
                        key: Key('manual-food-$food'),
                        title: Text(_displayFoodName(food)),
                        onTap: () => Navigator.of(context).pop(food),
                      ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
    if (!mounted ||
        !widget.isActive ||
        selectionRequest != _scanRequest ||
        foodId == null) {
      return;
    }
    if (foodId == _unsupportedFood) {
      _declineUnsupportedFood();
      return;
    }

    _scanRequest++;
    _nutritionRequest++;
    FocusManager.instance.primaryFocus?.unfocus();
    setState(() {
      _selectedImage = null;
      _classification = null;
      _nutrition = null;
      _manualFoodId = foodId;
      _portionGramsController.clear();
      _portionInputInvalid = false;
      _isNonFoodRejected = false;
      _isAwaitingFoodConfirmation = true;
      _isPredictionDeclined = false;
      _isUnsupportedFood = false;
      _isClassifying = false;
      _isLoadingNutrition = false;
      _isUpdatingPortion = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      key: const Key('scan-screen-scroll'),
      keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
      padding: const EdgeInsets.fromLTRB(22, 24, 22, 28),
      children: [
        Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Expanded(
              child: ScreenHeader(
                title: 'Food Calorie Estimator',
                subtitle: 'A clearer picture of what is on your plate.',
              ),
            ),
            if (_hasScanState)
              IconButton(
                key: const Key('reset-scan-button'),
                tooltip: 'Reset scan',
                onPressed: _isSavingMeal ? null : _resetScan,
                icon: const Icon(Icons.refresh),
              ),
          ],
        ),
        const SizedBox(height: 26),
        Card(
          child: Padding(
            padding: const EdgeInsets.all(22),
            child: Column(
              children: [
                Text(
                  'Scan one of the supported foods',
                  style: Theme.of(context).textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
                TextButton(
                  key: const Key('view-supported-foods'),
                  onPressed: _showSupportedFoods,
                  child: const Text('View supported foods'),
                ),
                const SizedBox(height: 8),
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
                if (_isNonFoodRejected) ...[
                  const SizedBox(height: 14),
                  const Text(
                    'This photo was identified as a non-food object. Please photograph a meal and try again.',
                    key: Key('non-food-rejection'),
                    textAlign: TextAlign.center,
                  ),
                ],
                if (_classification case final result?
                    when !result.isNonFood) ...[
                  const SizedBox(height: 14),
                  Text(
                    'Predicted food: ${result.food!.replaceAll('_', ' ')}',
                    key: const Key('food-prediction'),
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  Text(
                    'Model confidence (informational): ${(result.confidence * 100).toStringAsFixed(1)}%',
                  ),
                ],
                if (_manualFoodId case final foodId?) ...[
                  const SizedBox(height: 14),
                  Text(
                    'User-selected food: ${_displayFoodName(foodId)}',
                    key: const Key('manual-food-selection'),
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ],
                if (_isAwaitingFoodConfirmation) ...[
                  const SizedBox(height: 14),
                  Text(
                    'Is this the correct food?',
                    key: const Key('food-confirmation-question'),
                    style: Theme.of(context).textTheme.titleMedium,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    alignment: WrapAlignment.center,
                    spacing: 10,
                    runSpacing: 8,
                    children: [
                      FilledButton(
                        key: const Key('confirm-food-button'),
                        onPressed: _confirmFoodSelection,
                        child: const Text('Yes, correct'),
                      ),
                      OutlinedButton(
                        key: const Key('decline-food-button'),
                        onPressed: _declineFoodSelection,
                        child: const Text('No, incorrect'),
                      ),
                    ],
                  ),
                ],
                if (_isPredictionDeclined) ...[
                  const SizedBox(height: 14),
                  const Text(
                    'Choose another photo or select a listed food manually. If your food is not listed, it is not currently supported.',
                    key: Key('prediction-declined-message'),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  _retryAndManualActions(),
                ],
                if (_isUnsupportedFood) ...[
                  const SizedBox(height: 14),
                  const Text(
                    'This food is not currently supported, so no nutrition estimate was created. Please try another meal photo.',
                    key: Key('unsupported-food-message'),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 8),
                  _retryPhotoActions(),
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
                if (_nutrition != null) ...[
                  const SizedBox(height: 14),
                  SizedBox(
                    width: double.infinity,
                    child: FilledButton.icon(
                      key: const Key('add-meal-button'),
                      onPressed: _canAddMeal ? _addMeal : null,
                      icon: _isSavingMeal
                          ? const SizedBox.square(
                              dimension: 18,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.add),
                      label: Text(_isSavingMeal ? 'Saving meal…' : 'Add Meal'),
                    ),
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
                Text(
                  'Scan with Photo',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 10),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    onPressed: _isPickingImage || _isSavingMeal
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
                    onPressed: _isPickingImage || _isSavingMeal
                        ? null
                        : () => _pickImage(useCamera: false),
                    icon: const Icon(Icons.photo_library_outlined),
                    label: const Text('Choose from Gallery'),
                  ),
                ),
                const SizedBox(height: 18),
                const Divider(),
                const SizedBox(height: 10),
                Text(
                  'Add Manually',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: OutlinedButton.icon(
                    key: const Key('add-manually-button'),
                    onPressed: _isPickingImage || _isSavingMeal
                        ? null
                        : _addManually,
                    icon: const Icon(Icons.edit_outlined),
                    label: const Text('Choose a supported food'),
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

  Future<void> _showSupportedFoods() => showDialog<void>(
    context: context,
    builder: (context) => AlertDialog(
      title: const Text('Supported foods'),
      content: SizedBox(
        width: double.maxFinite,
        height: 320,
        child: ListView(
          shrinkWrap: true,
          children: [
            for (final food in foodClassNames)
              ListTile(title: Text(_displayFoodName(food))),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Close'),
        ),
      ],
    ),
  );

  Widget _retryAndManualActions() => Wrap(
    alignment: WrapAlignment.center,
    spacing: 8,
    runSpacing: 8,
    children: [
      OutlinedButton.icon(
        key: const Key('retry-camera-button'),
        onPressed: _isPickingImage || _isSavingMeal
            ? null
            : () => _pickImage(useCamera: true),
        icon: const Icon(Icons.photo_camera_outlined),
        label: const Text('Try Camera'),
      ),
      OutlinedButton.icon(
        key: const Key('retry-gallery-button'),
        onPressed: _isPickingImage || _isSavingMeal
            ? null
            : () => _pickImage(useCamera: false),
        icon: const Icon(Icons.photo_library_outlined),
        label: const Text('Try Gallery'),
      ),
      FilledButton.tonal(
        key: const Key('declined-manual-selection-button'),
        onPressed: _isPickingImage || _isSavingMeal ? null : _addManually,
        child: const Text('Choose manually'),
      ),
      TextButton(
        key: const Key('decline-unsupported-food-button'),
        onPressed: _declineUnsupportedFood,
        child: const Text("My food isn't listed"),
      ),
    ],
  );

  Widget _retryPhotoActions() => Wrap(
    alignment: WrapAlignment.center,
    spacing: 8,
    runSpacing: 8,
    children: [
      OutlinedButton.icon(
        key: const Key('unsupported-retry-camera-button'),
        onPressed: _isPickingImage || _isSavingMeal
            ? null
            : () => _pickImage(useCamera: true),
        icon: const Icon(Icons.photo_camera_outlined),
        label: const Text('Try Camera'),
      ),
      OutlinedButton.icon(
        key: const Key('unsupported-retry-gallery-button'),
        onPressed: _isPickingImage || _isSavingMeal
            ? null
            : () => _pickImage(useCamera: false),
        icon: const Icon(Icons.photo_library_outlined),
        label: const Text('Try Gallery'),
      ),
    ],
  );
}

String _displayFoodName(String foodId) => foodId
    .split('_')
    .map((word) => '${word[0].toUpperCase()}${word.substring(1).toLowerCase()}')
    .join(' ');

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
            onTapOutside: (_) => FocusManager.instance.primaryFocus?.unfocus(),
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
              'Nutrition is estimated from the selected portion. Image weight is not inferred.',
            ),
          ),
        ],
      ),
    );
  }
}
