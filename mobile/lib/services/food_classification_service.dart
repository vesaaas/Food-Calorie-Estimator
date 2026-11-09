import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_litert/flutter_litert.dart';

const foodClassNames = <String>[
  'pizza',
  'hamburger',
  'sushi',
  'steak',
  'ice_cream',
  'donuts',
  'omelette',
  'french_fries',
  'chicken_wings',
  'spaghetti_bolognese',
  'borek',
  'kebap',
  'sarma',
];

const nonFoodClassName = 'non_food';
const foodModelClassNames = <String>[...foodClassNames, nonFoodClassName];

const _inputTensorShape = [1, 224, 224, 3];
const _outputTensorShape = [1, 14];

class FoodClassification {
  // A supported prediction requires a concrete catalog food ID; the only
  // result with no food ID is the explicit non-food outcome.
  const FoodClassification({required String food, required double confidence})
    : this._(food, confidence, false);

  const FoodClassification.nonFood({required double confidence})
    : this._(null, confidence, true);

  const FoodClassification._(this.food, this.confidence, this.isNonFood);

  final String? food;
  final double confidence;
  final bool isNonFood;
}

abstract interface class FoodClassificationService {
  Future<FoodClassification> classifyImage(String imagePath);
}

abstract interface class FoodModelRunner {
  Future<List<double>> run(Float32List input);
}

class TfliteFoodModelRunner implements FoodModelRunner {
  TfliteFoodModelRunner({
    this.assetPath = modelAssetPath,
    this.labelsAssetPath = defaultLabelsAssetPath,
  });

  static const modelAssetPath =
      'assets/models/mobilenet_v3_small_phase9_5_14class.tflite';
  static const defaultLabelsAssetPath =
      'assets/models/labels_phase9_5_14class.txt';

  final String assetPath;
  final String labelsAssetPath;
  Interpreter? _interpreter;

  @override
  Future<List<double>> run(Float32List input) async {
    if (input.length != 224 * 224 * 3 ||
        input.any((value) => !value.isFinite || value < 0 || value > 255)) {
      throw StateError('The food model input does not match its contract.');
    }
    final interpreter = _interpreter ?? await _loadInterpreter();
    _interpreter = interpreter;
    final output = Float32List(foodModelClassNames.length);
    interpreter.run(input, output);
    return output.toList(growable: false);
  }

  Future<Interpreter> _loadInterpreter() async {
    final labels = (await rootBundle.loadString(labelsAssetPath))
        .split('\n')
        .map((label) => label.trim())
        .where((label) => label.isNotEmpty)
        .toList(growable: false);
    if (!_sameLabels(labels, foodModelClassNames)) {
      throw StateError('The food model labels do not match their contract.');
    }
    final interpreter = await Interpreter.fromAsset(assetPath);
    final input = interpreter.getInputTensor(0);
    final output = interpreter.getOutputTensor(0);
    if (!_sameShape(input.shape, _inputTensorShape) ||
        input.type != TensorType.float32 ||
        !_sameShape(output.shape, _outputTensorShape) ||
        output.type != TensorType.float32) {
      interpreter.close();
      throw StateError(
        'The food model does not match its expected tensor contract.',
      );
    }
    return interpreter;
  }

  bool _sameShape(List<int> actual, List<int> expected) {
    if (actual.length != expected.length) return false;
    for (var i = 0; i < actual.length; i++) {
      if (actual[i] != expected[i]) return false;
    }
    return true;
  }

  bool _sameLabels(List<String> actual, List<String> expected) {
    if (actual.length != expected.length) return false;
    for (var i = 0; i < actual.length; i++) {
      if (actual[i] != expected[i]) return false;
    }
    return true;
  }
}

class OnDeviceFoodClassificationService implements FoodClassificationService {
  OnDeviceFoodClassificationService({FoodModelRunner? modelRunner})
    : _modelRunner = modelRunner ?? TfliteFoodModelRunner();

  final FoodModelRunner _modelRunner;

  @override
  Future<FoodClassification> classifyImage(String imagePath) async {
    final bytes = await File(imagePath).readAsBytes();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) throw const FormatException('Unsupported image.');

    final input = phase95FoodImageToFloatRgb(decoded);
    final scores = await _modelRunner.run(input);
    final sum = scores.fold<double>(0, (total, score) => total + score);
    if (scores.length != foodModelClassNames.length ||
        scores.any((score) => !score.isFinite || score < 0 || score > 1) ||
        !sum.isFinite ||
        (sum - 1).abs() > 0.001) {
      throw StateError('The food model returned invalid predictions.');
    }
    var bestIndex = 0;
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIndex]) bestIndex = i;
    }
    if (bestIndex == foodClassNames.length) {
      return FoodClassification.nonFood(confidence: scores[bestIndex]);
    }
    return FoodClassification(
      food: foodModelClassNames[bestIndex],
      confidence: scores[bestIndex],
    );
  }
}

/// Applies the Phase 9.5 Python geometry and returns raw RGB pixels.
///
/// EXIF orientation is baked before calculating dimensions so rotated inputs
/// match Pillow's `ImageOps.exif_transpose` before its resize step.
img.Image resizeAndCenterCropFoodImage(img.Image source) {
  final orientation = source.exif.imageIfd.orientation ?? 1;
  // Pillow's `convert('RGB')` drops alpha without compositing and produces
  // three interleaved 8-bit channels. `_orientedPixel` reads those RGB
  // channels in EXIF-corrected order without materializing a rotated copy.
  final swapAxes = orientation >= 5 && orientation <= 8;
  final orientedWidth = swapAxes ? source.height : source.width;
  final orientedHeight = swapAxes ? source.width : source.height;
  final scale =
      256 / (orientedWidth < orientedHeight ? orientedWidth : orientedHeight);
  // Python int(float) truncates positive dimensions toward zero.
  final width = (orientedWidth * scale).floor();
  final height = (orientedHeight * scale).floor();
  final resized = _pillowBilinearResize(source, orientation, width, height);
  return img.copyCrop(
    resized,
    x: _pythonRoundHalfToEven((width - 224) / 2.0),
    y: _pythonRoundHalfToEven((height - 224) / 2.0),
    width: 224,
    height: 224,
  );
}

img.Image _pillowBilinearResize(
  img.Image source,
  int orientation,
  int width,
  int height,
) {
  final sourceWidth = orientation >= 5 && orientation <= 8
      ? source.height
      : source.width;
  final sourceHeight = orientation >= 5 && orientation <= 8
      ? source.width
      : source.height;
  final horizontal = Uint8List(width * sourceHeight * 3);
  final horizontalWeights = _pillowBilinearWeights(sourceWidth, width);
  for (var y = 0; y < sourceHeight; y++) {
    for (var x = 0; x < width; x++) {
      final weights = horizontalWeights[x];
      for (var channel = 0; channel < 3; channel++) {
        var value = 0.0;
        for (var tap = 0; tap < weights.indices.length; tap++) {
          final pixel = _orientedPixel(
            source,
            orientation,
            weights.indices[tap],
            y,
          );
          final component = switch (channel) {
            0 => pixel.r,
            1 => pixel.g,
            _ => pixel.b,
          };
          value += component.toDouble().clamp(0, 255) * weights.values[tap];
        }
        horizontal[(y * width + x) * 3 + channel] = value
            .round()
            .clamp(0, 255)
            .toInt();
      }
    }
  }

  final output = Uint8List(width * height * 3);
  final verticalWeights = _pillowBilinearWeights(sourceHeight, height);
  for (var y = 0; y < height; y++) {
    final weights = verticalWeights[y];
    for (var x = 0; x < width; x++) {
      for (var channel = 0; channel < 3; channel++) {
        var value = 0.0;
        for (var tap = 0; tap < weights.indices.length; tap++) {
          final sourceIndex = (weights.indices[tap] * width + x) * 3 + channel;
          value += horizontal[sourceIndex] * weights.values[tap];
        }
        output[(y * width + x) * 3 + channel] = value
            .round()
            .clamp(0, 255)
            .toInt();
      }
    }
  }

  return img.Image.fromBytes(
    width: width,
    height: height,
    bytes: output.buffer,
    numChannels: 3,
    order: img.ChannelOrder.rgb,
  );
}

img.Pixel _orientedPixel(img.Image source, int orientation, int x, int y) {
  final sourceX = switch (orientation) {
    2 || 3 => source.width - 1 - x,
    5 || 6 => y,
    7 || 8 => source.width - 1 - y,
    _ => x,
  };
  final sourceY = switch (orientation) {
    3 || 4 => source.height - 1 - y,
    5 || 8 => x,
    6 || 7 => source.height - 1 - x,
    _ => y,
  };
  return source.getPixel(sourceX, sourceY);
}

List<_ResampleWeights> _pillowBilinearWeights(int sourceSize, int outputSize) {
  final scale = sourceSize / outputSize;
  final filterScale = scale < 1 ? 1.0 : scale;
  final support = filterScale;
  return List<_ResampleWeights>.generate(outputSize, (outputIndex) {
    final center = (outputIndex + 0.5) * scale;
    final first = (center - support + 0.5)
        .floor()
        .clamp(0, sourceSize - 1)
        .toInt();
    final last = (center + support + 0.5)
        .floor()
        .clamp(0, sourceSize - 1)
        .toInt();
    final indices = <int>[];
    final values = <double>[];
    var sum = 0.0;
    for (var inputIndex = first; inputIndex <= last; inputIndex++) {
      final distance = ((inputIndex + 0.5) - center).abs() / filterScale;
      final weight = (1.0 - distance).clamp(0.0, 1.0);
      if (weight > 0) {
        indices.add(inputIndex);
        values.add(weight);
        sum += weight;
      }
    }
    if (sum == 0) throw StateError('Empty Pillow bilinear filter support.');
    for (var i = 0; i < values.length; i++) {
      values[i] /= sum;
    }
    return _ResampleWeights(indices, values);
  });
}

class _ResampleWeights {
  const _ResampleWeights(this.indices, this.values);

  final List<int> indices;
  final List<double> values;
}

/// Converts an orientation-aware decoded image to flattened NHWC RGB values.
/// Values remain float32 `[0,255]`; MobileNetV3 scaling stays inside the model.
Float32List phase95FoodImageToFloatRgb(img.Image source) {
  final cropped = resizeAndCenterCropFoodImage(source);
  final input = Float32List(224 * 224 * 3);
  var offset = 0;
  for (var y = 0; y < 224; y++) {
    for (var x = 0; x < 224; x++) {
      final pixel = cropped.getPixel(x, y);
      input[offset++] = pixel.r.toDouble();
      input[offset++] = pixel.g.toDouble();
      input[offset++] = pixel.b.toDouble();
    }
  }
  return input;
}

int _pythonRoundHalfToEven(double value) {
  final lower = value.floor();
  final fraction = value - lower;
  if (fraction < 0.5) return lower;
  if (fraction > 0.5) return lower + 1;
  return lower.isEven ? lower : lower + 1;
}
