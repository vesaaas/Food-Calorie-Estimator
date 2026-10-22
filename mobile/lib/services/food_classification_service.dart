import 'dart:io';
import 'dart:typed_data';

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

const foodClassificationConfidenceThreshold = 0.55;

const _inputTensorShape = [1, 224, 224, 3];
const _outputTensorShape = [1, 13];

class FoodClassification {
  const FoodClassification({required this.food, required this.confidence});

  final String food;
  final double confidence;

  bool get meetsConfidenceThreshold =>
      confidence >= foodClassificationConfidenceThreshold;
}

abstract interface class FoodClassificationService {
  Future<FoodClassification> classifyImage(String imagePath);
}

abstract interface class FoodModelRunner {
  Future<List<double>> run(Float32List input);
}

class TfliteFoodModelRunner implements FoodModelRunner {
  TfliteFoodModelRunner({this.assetPath = modelAssetPath});

  static const modelAssetPath =
      'assets/models/mobilenet_v3_small_phase7_v2.tflite';

  final String assetPath;
  Interpreter? _interpreter;

  @override
  Future<List<double>> run(Float32List input) async {
    final interpreter = _interpreter ?? await _loadInterpreter();
    _interpreter = interpreter;
    final output = Float32List(foodClassNames.length);
    interpreter.run(input, output);
    return output.toList(growable: false);
  }

  Future<Interpreter> _loadInterpreter() async {
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

    final resized = _resizeAndCenterCrop(decoded);
    final input = Float32List(224 * 224 * 3);
    var offset = 0;
    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        input[offset++] = pixel.r.toDouble();
        input[offset++] = pixel.g.toDouble();
        input[offset++] = pixel.b.toDouble();
      }
    }

    final scores = await _modelRunner.run(input);
    if (scores.length != foodClassNames.length ||
        scores.any((score) => !score.isFinite)) {
      throw StateError('The food model returned invalid predictions.');
    }
    var bestIndex = 0;
    for (var i = 1; i < scores.length; i++) {
      if (scores[i] > scores[bestIndex]) bestIndex = i;
    }
    return FoodClassification(
      food: foodClassNames[bestIndex],
      confidence: scores[bestIndex],
    );
  }

  img.Image _resizeAndCenterCrop(img.Image source) {
    final scale =
        256 / (source.width < source.height ? source.width : source.height);
    final width = (source.width * scale).round();
    final height = (source.height * scale).round();
    final resized = img.copyResize(
      source,
      width: width,
      height: height,
      interpolation: img.Interpolation.linear,
    );
    return img.copyCrop(
      resized,
      x: (width - 224) ~/ 2,
      y: (height - 224) ~/ 2,
      width: 224,
      height: 224,
    );
  }
}
