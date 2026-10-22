import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:food_calorie_estimator/services/food_classification_service.dart';
import 'package:image/image.dart' as img;

void main() {
  late Directory directory;

  setUp(() async {
    directory = await Directory.systemTemp.createTemp(
      'food_classification_test',
    );
  });

  tearDown(() async {
    await directory.delete(recursive: true);
  });

  test(
    'decodes PNG and passes float32 RGB values in flattened NHWC order',
    () async {
      final image = img.Image(width: 256, height: 256);
      for (var y = 0; y < image.height; y++) {
        for (var x = 0; x < image.width; x++) {
          image.setPixelRgb(x, y, x, y, 200);
        }
      }
      final runner = _RecordingRunner(_scoresFor(0, 0.8));
      final result = await _classify(directory, image, runner);

      expect(result.food, 'pizza');
      expect(result.confidence, 0.8);
      final input = runner.input!;
      expect(input, isA<Float32List>());
      expect(input, hasLength(1 * 224 * 224 * 3));
      // The 256x256 image is center-cropped from (16,16); adjacent pixels are
      // stored as RGB, then rows follow in NHWC order.
      expect(input.take(6), [16, 16, 200, 17, 16, 200]);
      expect(input.skip(224 * 3).take(3), [16, 17, 200]);
      expect(input.skip((224 * 224 - 1) * 3).take(3), [239, 239, 200]);
      expect(input.every((value) => value >= 0 && value <= 255), isTrue);
    },
  );

  test('resizes the short edge and takes a centered crop', () async {
    final image = img.Image(width: 448, height: 224);
    for (var y = 0; y < image.height; y++) {
      for (var x = 0; x < image.width; x++) {
        if (x < 120) {
          image.setPixelRgb(x, y, 255, 0, 0);
        } else if (x < 330) {
          image.setPixelRgb(x, y, 0, 200, 0);
        } else {
          image.setPixelRgb(x, y, 0, 0, 255);
        }
      }
    }
    final runner = _RecordingRunner(_scoresFor(0, 0.8));
    await _classify(directory, image, runner);

    final input = runner.input!;
    expect(input, hasLength(224 * 224 * 3));
    for (final pixelIndex in [0, (112 * 224) + 112, (224 * 224) - 1]) {
      expect(input.skip(pixelIndex * 3).take(3), [0, 200, 0]);
    }
  });

  test('maps every output index to its exact canonical class', () async {
    for (var index = 0; index < foodClassNames.length; index++) {
      final runner = _RecordingRunner(_scoresFor(index, 0.73));
      final result = await _classify(directory, _solidImage(), runner);

      expect(result.food, foodClassNames[index], reason: 'output index $index');
      expect(result.confidence, 0.73, reason: 'output index $index');
    }
  });

  test('selects the maximum score, not the first or last output', () async {
    final scores = List<double>.filled(foodClassNames.length, 0.01);
    scores[2] = 0.42;
    scores[10] = 0.88;
    scores[12] = 0.17;
    final result = await _classify(
      directory,
      _solidImage(),
      _RecordingRunner(scores),
    );

    expect(result.food, 'borek');
    expect(result.confidence, 0.88);
  });

  test('rejects output with the wrong length or non-finite scores', () async {
    for (final scores in [
      List<double>.filled(foodClassNames.length - 1, 0.1),
      List<double>.filled(foodClassNames.length + 1, 0.1),
      [..._scoresFor(0, 0.8)..[4] = double.nan],
      [..._scoresFor(0, 0.8)..[4] = double.infinity],
      [..._scoresFor(0, 0.8)..[4] = double.negativeInfinity],
    ]) {
      await expectLater(
        _classify(directory, _solidImage(), _RecordingRunner(scores)),
        throwsA(isA<StateError>()),
      );
    }
  });

  test(
    'rejects undecodable image data before invoking the model runner',
    () async {
      final file = File('${directory.path}/invalid.png');
      await file.writeAsBytes([1, 2, 3, 4]);
      final runner = _RecordingRunner(_scoresFor(0, 0.8));

      await expectLater(
        OnDeviceFoodClassificationService(modelRunner: runner)
            .classifyImage(file.path),
        throwsA(isA<FormatException>()),
      );
      expect(runner.calls, 0);
    },
  );
}

Future<FoodClassification> _classify(
  Directory directory,
  img.Image source,
  _RecordingRunner runner,
) async {
  final file = File('${directory.path}/synthetic.png');
  await file.writeAsBytes(img.encodePng(source));
  return OnDeviceFoodClassificationService(modelRunner: runner)
      .classifyImage(file.path);
}

img.Image _solidImage() {
  final image = img.Image(width: 256, height: 256);
  for (var y = 0; y < image.height; y++) {
    for (var x = 0; x < image.width; x++) {
      image.setPixelRgb(x, y, 42, 84, 126);
    }
  }
  return image;
}

List<double> _scoresFor(int index, double score) => List<double>.filled(
  foodClassNames.length,
  (1 - score) / (foodClassNames.length - 1),
)..[index] = score;

class _RecordingRunner implements FoodModelRunner {
  _RecordingRunner(this.scores);

  final List<double> scores;
  Float32List? input;
  var calls = 0;

  @override
  Future<List<double>> run(Float32List input) async {
    this.input = input;
    calls++;
    return scores;
  }
}
